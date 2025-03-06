from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# i
def download_hugging_face_embeddings():
    """
    Create HuggingFace embeddings using the latest package
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def load_pdf(path):
    """
    Load PDF documents from a directory
    """
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(documents):
    """
    Split documents into text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks