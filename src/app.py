from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import pinecone
import os
import torch
import yaml

def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
    - file_path (str): The path to the YAML file containing configuration.

    Returns:
    - dict: A dictionary containing the loaded configuration.
    """
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading configuration: {exc}")
            return None

def load_pdf_files(input_dir):
    """
    Load PDF files from a directory.

    Args:
    - input_dir (str): The directory path containing PDF files.

    Returns:
    - List[Document]: A list of Document objects representing the loaded PDF files.
    """
    loader = PyPDFDirectoryLoader(input_dir)
    return loader.load()

def split_text(data):
    """
    Split extracted data into text chunks.

    Args:
    - data (List[Document]): A list of Document objects.

    Returns:
    - List[TextChunk]: A list of TextChunk objects representing the split text chunks.
    """
    Text_Splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return Text_Splitter.split_documents(data)

def load_embeddings(model_name):
    """
    Load embeddings from Hugging Face.

    Args:
    - model_name (str): The name of the Hugging Face model for embeddings.

    Returns:
    - Embeddings: An embeddings object loaded from Hugging Face.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def initialize_pinecone(api_key, api_env):
    """
    Initialize Pinecone with the provided API key and environment.

    Args:
    - api_key (str): The API key for Pinecone.
    - api_env (str): The environment for Pinecone.

    Returns:
    - None
    """
    pinecone.init(api_key=api_key, environment=api_env)

def setup_pinecone_index(text_chunks, embeddings, index_name):
    """
    Setup a Pinecone index using text chunks and embeddings.

    Args:
    - text_chunks (List[TextChunk]): A list of TextChunk objects.
    - embeddings (Embeddings): An embeddings object.
    - index_name (str): The name of the Pinecone index.

    Returns:
    - PineconeIndex: A Pinecone index object.
    """
    return Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

def setup_hugging_face_pipeline(model, tokenizer):
    """
    Setup a Hugging Face pipeline for text generation.

    Args:
    - model (Model): The Hugging Face model for text generation.
    - tokenizer (Tokenizer): The tokenizer for the model.

    Returns:
    - HuggingFacePipeline: A pipeline object for text generation.
    """
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=30,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )
    return HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.1})

def create_prompt_template():
    """
    Create a prompt template for question answering.

    Returns:
    - PromptTemplate: A template object for question answering prompts.
    """
    instruction = """
    {context}

    Question: {question}
    """
    SYSTEM_PROMPT = """Utilize the provided information to respond to the question. If you are uncertain about the answer, simply acknowledge that you don't know, and refrain from providing speculative answers."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"
    system_prompt = B_SYS + SYSTEM_PROMPT + E_SYS
    template = B_INST + system_prompt + instruction + E_INST
    return PromptTemplate(template=template, input_variables=["context", "question"])

def setup_qa_chain(llm, docsearch, prompt):
    """
    Setup a question answering chain.

    Args:
    - llm (HuggingFacePipeline): The Hugging Face pipeline for text generation.
    - docsearch (PineconeIndex): The Pinecone index for document retrieval.
    - prompt (PromptTemplate): The template for question answering prompts.

    Returns:
    - RetrievalQA: A question answering chain object.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def main():
    """
    The main function for executing the PDF Interaction ChatBot.

    Returns:
    - None
    """
    # Load configuration from the YAML file
    config_path = "../config.yaml"
    config = load_config(config_path)

    # import configuration from yaml
    input_dir = config.get('input_dir')
    modelEmbeddings  = config.get("modelEmbeddings")
    autoTokenizer = config.get("autoTokenizer")
    autoModelForCausalLM = config.get("autoModelForCausalLM")
    SYSTEM_PROMPT = config.get('SYSTEM_PROMPT')

    # Initializing Pinecone
    PINECONE_API_KEY = config.get('PINECONE_API_KEY')
    PINECONE_API_ENV = config.get('PINECONE_API_ENV')
    initialize_pinecone(PINECONE_API_KEY, PINECONE_API_ENV)

    # Load the pdf Files
    data = load_pdf_files(input_dir)

    # Split the extracted data into text chunks
    Docs = split_text(data)

    # Download the embeddings from hugging face
    embeddings = load_embeddings(modelEmbeddings)

    # Create embeddings for each text chunk
    pinecone_index = setup_pinecone_index(Docs, embeddings, index_name="test-1")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(autoTokenizer, token=True)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(autoModelForCausalLM,
                                                device_map='auto',
                                                torch_dtype=torch.float32,
                                                token=True,
                                                load_in_8bit=False
                                                )

    # Set the pipeline
    llm = setup_hugging_face_pipeline(model, tokenizer)

    # Create a prompt template
    prompt_template = create_prompt_template()

    # Create the QA chain
    qa_chain = setup_qa_chain(llm, pinecone_index, prompt_template)

    while True:
        user_input=input(f"prompt:")
        if user_input=='exit':
            print('Exiting')
            sys.exit()
        if user_input=='':
            continue
        result=qa_chain({'query':user_input})
        print(f"Answer:{result['result']}")

if __name__ == "__main__":
    main()