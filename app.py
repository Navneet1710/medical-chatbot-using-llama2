from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, text_split, load_pdf
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone
import os
from tqdm import tqdm

app = Flask(__name__)

load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
INDEX_NAME = "medical-bot"

def initialize_pinecone():
    """Initialize Pinecone client and return the index"""
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    return pc

def setup_qa_chain():
    """Set up the QA chain with embeddings and LLM"""
    # Initialize embeddings
    embeddings = download_hugging_face_embeddings()
    
    # Initialize Pinecone vector store
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )
    
    # Set up LLM
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={
            'max_new_tokens': 512,
            'temperature': 0.8
        }
    )
    
    # Create prompt template
    PROMPT = PromptTemplate(
        template="""Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:""", 
        input_variables=["context", "question"]
    )
    # Create QA chain using the new invoke method
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# Initialize QA chain
try:
    qa = setup_qa_chain()
except Exception as e:
    print(f"Error setting up QA chain: {e}")
    raise

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form['msg']
    input = msg
    print(input)
    result=qa({"query":input})
    print("Response:",result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)