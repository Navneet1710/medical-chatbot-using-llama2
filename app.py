import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pinecone
import os
import time
from datetime import datetime
import base64
from pathlib import Path

# Load environment variables
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

# Function to load and encode local images
def get_base64_of_image(image_path):
    """Get base64 encoded string of an image to embed in HTML"""
    if not os.path.exists(image_path):
        # Return a default image or placeholder if the image doesn't exist
        return ""
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Custom CSS for chat styling
def local_css():
    # Define paths to your local images - REPLACE THESE WITH YOUR ACTUAL LOCAL PATHS
    # If you don't have local images, leave a comment and we'll handle that case
    bot_avatar_path = "assets/bot-avatar.png"  # Replace with your local bot avatar path
    user_avatar_path = "assets/user-avatar.png"  # Replace with your local user avatar path
    
    # Get base64 encoded images or use fallback colors if images don't exist
    bot_avatar_base64 = get_base64_of_image(bot_avatar_path)
    user_avatar_base64 = get_base64_of_image(user_avatar_path)
    
    # Prepare image CSS with fallbacks
    bot_avatar_css = f"background-image: url(data:image/png;base64,{bot_avatar_base64}); background-size: cover;" if bot_avatar_base64 else "background-color: #3a7efc;"
    user_avatar_css = f"background-image: url(data:image/png;base64,{user_avatar_base64}); background-size: cover;" if user_avatar_base64 else "background-color: #58cc71;" 
    
    st.markdown(f"""
    <style>
    .chat-container {{
        padding: 10px;
        border-radius: 15px;
        background-color: rgba(0,0,0,0.4);
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
    }}
    .user-message {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }}
    .bot-message {{
        display: flex;
        justify-content: flex-start;
        margin-bottom: 10px;
    }}
    .msg-content-user {{
        background-color: #58cc71;
        padding: 10px;
        border-radius: 15px;
        max-width: 70%;
        color: white;
    }}
    .msg-content-bot {{
        background-color: rgb(82, 172, 255);
        padding: 10px;
        border-radius: 15px;
        max-width: 70%;
        color: white;
    }}
    .avatar-img {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
        {user_avatar_css}
    }}
    .avatar-img-bot {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
        {bot_avatar_css}
    }}
    .timestamp {{
        font-size: 10px;
        color: rgba(255,255,255,0.5);
        margin-top: 5px;
    }}
    .page-bg {{
        background: linear-gradient(to right, rgb(38, 51, 61), rgb(50, 55, 65), rgb(33, 33, 78));
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }}
    .chat-header {{
        display: flex;
        align-items: center;
        background-color: rgba(0,0,0,0.4);
        padding: 10px;
        border-radius: 15px 15px 0 0;
        color: white;
    }}
    .chat-input {{
        background-color: rgba(0,0,0,0.4) !important;
        border-radius: 0 0 15px 15px !important;
        padding: 10px;
    }}
    .stTextInput input {{
        background-color: rgba(0,0,0,0.3) !important;
        border: none !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 15px !important;
    }}
    .online-indicator {{
        height: 15px;
        width: 15px;
        background-color: #4cd137;
        border-radius: 50%;
        border: 1.5px solid white;
        margin-left: -15px;
        margin-top: 25px;
    }}
    .header-text {{
        margin-left: 15px;
    }}
    .header-title {{
        font-size: 20px;
        font-weight: bold;
    }}
    .header-subtitle {{
        font-size: 12px;
        color: rgba(255,255,255,0.6);
    }}
    .header-avatar {{
        width: 70px; 
        height: 70px;
        border-radius: 50%;
        margin-right: 15px;
        {bot_avatar_css}
    }}
    </style>
    <div class="page-bg"></div>
    """, unsafe_allow_html=True)

def main():
    local_css()
    
    # Create or get session state for chat history and input tracking
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Function to handle message submission
    def handle_input():
        if st.session_state.widget_input and not st.session_state.processing:
            # Set processing flag to prevent multiple submissions
            st.session_state.processing = True
            
            # Get current input and clear the input field
            user_message = st.session_state.widget_input
            st.session_state.widget_input = ""
            
            # Get current time
            current_time = datetime.now().strftime("%H:%M")
            
            # Add user message to chat
            st.session_state.messages.append({
                'role': 'user',
                'content': user_message,
                'time': current_time
            })
            
            # Process with QA chain
            try:
                result = st.session_state.qa_chain({"query": user_message})
                bot_response = result["result"]
            except Exception as e:
                bot_response = f"Sorry, I encountered an error: {str(e)}"
            
            # Add bot response to chat
            st.session_state.messages.append({
                'role': 'bot',
                'content': bot_response,
                'time': current_time
            })
            
            # Reset processing flag
            st.session_state.processing = False
    
    # Chat header
    st.markdown("""
    <div class="chat-header">
        <div style="position: relative;">
            <div class="header-avatar"></div>
            <div class="online-indicator"></div>
        </div>
        <div class="header-text">
            <div class="header-title">Medical Chatbot</div>
            <div class="header-subtitle">Ask me anything!</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="user-message">
                <div class="msg-content-user">
                    {message['content']}
                    <div class="timestamp">{message['time']}</div>
                </div>
                <div class="avatar-img"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <div class="avatar-img-bot"></div>
                <div class="msg-content-bot">
                    {message['content']}
                    <div class="timestamp">{message['time']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input - Using callback for better control
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    st.text_input(
        "Type your message...", 
        key="widget_input", 
        on_change=handle_input
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize QA chain
    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading medical knowledge base..."):
            try:
                st.session_state.qa_chain = setup_qa_chain()
                st.success("Medical bot is ready!")
                time.sleep(1)
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
            except Exception as e:
                st.error(f"Error setting up QA chain: {e}")

# Run the app
if __name__ == "__main__":
    main()