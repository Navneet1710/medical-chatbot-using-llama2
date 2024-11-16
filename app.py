from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings,text_split,load_pdf
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

import os
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm  # for progress tracking

# Initialize Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)

index_name = "medical-bot"

# Check if index exists, if not create it
if index_name not in pc.list_indexes().names():
    spec = ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
    
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=spec
    )
    # Wait for index to be ready
    time.sleep(20)

# Get the index
index = pc.Index(index_name)

# Create embeddings and upsert to Pinecone in smaller batches
batch_size = 100  # Adjust based on your needs
embeddings_list = []
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
total_vectors = 0

print(f"Total chunks to process: {len(text_chunks)}")

# Process in batches
for i in tqdm(range(0, len(text_chunks), batch_size)):
    batch = text_chunks[i:i + batch_size]
    
    # Create embeddings for the batch
    batch_embeddings = []
    for chunk in batch:
        try:
            vector = embeddings.embed_query(chunk.page_content)
            batch_embeddings.append(vector)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            continue
    
    # Create vectors for the batch
    vectors = list(zip(
        [str(j) for j in range(i, i + len(batch_embeddings))],
        batch_embeddings,
        [{"text": chunk.page_content} for chunk in batch[:len(batch_embeddings)]]
    ))
    
    # Upsert batch
    try:
        index.upsert(vectors=vectors)
        total_vectors += len(vectors)
        print(f"Upserted batch of {len(vectors)} vectors. Total vectors: {total_vectors}")
        
        # Add a small delay between batches
        time.sleep(1)
    except Exception as e:
        print(f"Error upserting batch: {e}")
    
    # Verify the count periodically
    if i % (batch_size * 5) == 0:
        try:
            stats = index.describe_index_stats()
            print(f"Current index stats: {stats}")
        except Exception as e:
            print(f"Error getting index stats: {e}")

# Final verification
try:
    final_stats = index.describe_index_stats()
    print(f"Final index stats: {final_stats}")
except Exception as e:
    print(f"Error getting final stats: {e}")


#Creating Embeddings for Each of The Text Chunks & storing
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"
)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
