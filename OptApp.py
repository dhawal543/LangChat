import os
import warnings
import time
import glob
import json
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import redis
import numpy as np

class Chatbot:
    
    def __init__(self):
        warnings.filterwarnings("ignore")
        MODEL = "llama2"
        self.model = Ollama(model=MODEL)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.similar_docs = False
        self.chain = False
        self.cache_file = "cache.json"
        self.load_cache()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def list_files_in_folder(self, folder_path):
        files = glob.glob(os.path.join(folder_path, '*'))
        file_names = [file for file in files]
        return file_names

    def get_pdf_text(self, pdf_docs):
        text = " "
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(self, text_chunks):
        vectors = self.embeddings.embed_documents(text_chunks)
        for i, vector in enumerate(vectors):
            self.redis_client.set(f"doc_vector_{i}", json.dumps(vector))
            self.redis_client.set(f"doc_text_{i}", text_chunks[i])
        return len(vectors)

    def get_conversational_chain(self):
        template = """
        Answer the question based on the context below. If you can't
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context","question"])
        chain = load_qa_chain(self.model, chain_type="stuff", prompt=prompt)
        return chain

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_similar_documents(self, user_query, top_k=5):
        query_vector = self.embeddings.embed_query(user_query)
        similarities = []
        for key in self.redis_client.scan_iter("doc_vector_*"):
            vector = np.array(json.loads(self.redis_client.get(key)))
            similarity = self.cosine_similarity(query_vector, vector)
            similarities.append((similarity, key))
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_similar_keys = [key for _, key in similarities[:top_k]]
        docs = [Document(page_content=self.redis_client.get(key.replace(b"vector", b"text")).decode('utf-8')) for key in top_similar_keys]
        return docs

    def user_input(self, user_question, is_cache):
        query_vector = self.embeddings.embed_query(user_question)
        
        # Checking cache first
        if is_cache.lower() == 'y': 
            for cache_key in self.cache:
                cache_item = self.cache[cache_key]
            if 'vector' in cache_item:
                cached_vector = np.array(cache_item['vector'])
                similarity = self.cosine_similarity(query_vector, cached_vector)
                if similarity > 0.7:  
                    print("Using cached response.")
                    return cache_item['response']
            
        

        start_time = time.time()
        if self.similar_docs == False:
            self.similar_docs = self.retrieve_similar_documents(user_question)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time in retrieve_similar_documents function:", elapsed_time, "seconds")
        if self.chain == False:
            self.chain = self.get_conversational_chain()

        response = self.chain(
            {"input_documents": self.similar_docs, "question": user_question},
            return_only_outputs=True
        )

        # Update cache
        self.cache[user_question] = {'response': response, 'vector': query_vector}
        self.save_cache()
        return response


    def upload_docs(self):
        folder_path = '/Users/nikhil/Desktop/AB-InBev/Chatbot1/pdf-folder' 
        files = self.list_files_in_folder(folder_path)
        print("Files in folder:", files)
        text = self.get_pdf_text(files) 
        print("Converted into text")
        start_time = time.time()
        chunks = self.get_text_chunks(text)
        end_time = time.time()
        print("Converted into chunks")
        elapsed_time = end_time - start_time
        print("Elapsed Time in creating chunks:", elapsed_time, "seconds")
        start_time = time.time()
        num_vectors = self.get_vectorstore(chunks)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Stored {num_vectors} vectors into Redis")
        print("Elapsed Time in creating embeddings and store in Redis:", elapsed_time, "seconds")

    def main(self):
        while True:
            user_question = input("HEY!!, I am Wise Bud, Please enter your question: ")
            is_cache = input("Cache Enables, if yes, press Y, otherwise press ENTER:")
            start_time = time.time()
            if user_question == "exit":
                break
            if user_question:
                response = self.user_input(user_question,is_cache)
                if isinstance(response, dict) and 'output_text' in response:
                    response_text = response['output_text']
                else:
                    response_text = response

                if "Based on the provided document" in response_text:
                    answer = response_text.split("Based on the provided document,", 1)[-1].strip(': ').strip()
                else:
                    answer = response_text.strip(': ').strip()

                print(f"Answer: {answer}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Elapsed Time in generating the response for a user query:", elapsed_time, "seconds")

if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.main()
