from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
print(f"API Key loaded: {'Yes' if PINECONE_API_KEY else 'No'}")
print(f"Environment loaded: {'Yes' if PINECONE_API_ENV else 'No'}")

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("Pinecone API key or environment not found in .env file")

def main():
    # Load and split PDF documents
    extracted_data = load_pdf("data/")
    text_chunks = text_split(extracted_data)
    embeddings = download_hugging_face_embeddings()

    # Initialize Pinecone
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    index_name = "medical-bot"

    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        spec = pinecone.ServerlessSpec(
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
        print("Created new index. Waiting for it to initialize...")
        time.sleep(20)

    # Get the index
    index = pc.Index(index_name)

    # Create embeddings and upsert to Pinecone in smaller batches
    batch_size = 100  # Adjust based on your needs
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

    # Final verification ok
    try:
        final_stats = index.describe_index_stats()
        print(f"Final index stats: {final_stats}")
        print("Vector store creation completed successfully!")
    except Exception as e:
        print(f"Error getting final stats: {e}")

if __name__ == "__main__":
    main()