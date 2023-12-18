# PDF-Interaction-with-LangChain-and-Llama2 🦜🦙
![RAG](https://github.com/Niez-Gharbi/PDF-Interaction-with-LangChain-and-Llama2/assets/57814219/f0126bb9-2077-4f7f-9569-84cd0827fab2)

Welcome to the PDF Interaction ChatBot repository! This is an example of Retrieval Augmented Generation, the Chatbot can answer questions related to the PDF files provided, that will be loaded and fed as knowledge to the chatbot.

Technologies Used
* Langchain
* Llama2
* pinecone
* Hugging Face

## Prerequisites 📋
Before running the ChatBot, ensure that you have the required dependencies installed. You can install them using the following command:
```
pip install -r requirements.txt
```

## Configuration ⚙️
The ChatBot uses a configuration file (config.yaml) to specify the input directory for PDF files, Hugging Face model details, Pinecone API key and environment, and other parameters. Make sure to update the configuration file with the appropriate values.

## Running Locally 💻
To run the PDF Interaction ChatBot, execute the following command:

```
cd src
python app.py
```

The ChatBot will prompt you for input. Enter a question, and the ChatBot will utilize the provided information from the PDF documents to generate an answer. If you want to exit the ChatBot, type 'exit'.

