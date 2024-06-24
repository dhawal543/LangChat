README
Chatbot

This is a conversational AI chatbot designed to answer questions based on the content of uploaded PDF documents. It utilizes the LLaMA language model for generating responses and Redis for storing document embeddings to facilitate efficient retrieval of relevant information.
Features

    Document Upload: Upload PDF documents to the chatbot. The text from these documents is extracted, split into chunks, and stored in Redis as embeddings.
    Query Response: Ask questions, and the chatbot retrieves relevant chunks from the uploaded documents and generates a coherent response.
    Cache System: Option to cache responses for faster retrieval of similar queries.
    Conversational Chain: Uses a QA chain to generate context-aware answers.

How It Works

    Upload Documents: Upload PDFs and convert them into text chunks.
    Create Embeddings: Generate embeddings for each text chunk and store them in Redis.
    Ask Questions: Ask questions, and the chatbot retrieves the most relevant document chunks using cosine similarity.
    Generate Responses: Use a conversational chain to generate and return a context-aware answer.

Prerequisites

    Python 3.8 or higher
    Redis server running locally

Installation

    Clone the repository:

    bash


Create and activate a virtual environment:

bash

python -m venv venv
source venv/bin/activate

Install the required packages:

bash

    pip install -r requirements.txt

    Set up environment variables:
        Create a .env file in the project root.
        Add your Redis configuration and any other necessary environment variables.

Usage

    Upload Documents:
        Place your PDF files in the folder specified in the upload_docs method.
        Run the script:

    bash

    python chatbot.py

        The chatbot will process the PDFs, extract text, create embeddings, and store them in Redis.

    Ask Questions:
        Interact with the chatbot through the command line.
        Provide your questions when prompted, and get responses based on the uploaded documents.

