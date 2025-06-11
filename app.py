import streamlit as st
import logging
from pathlib import Path
import shutil
from chat.rag_chat import SalesCounsellorBot
from embeddings.embed_docs import process_uploaded_file, reembed_all_documents
from utils.conversation_manager import ConversationManager

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        # Initialize chatbot and ensure documents are embedded
        try:
            # First re-embed all documents to ensure they're in the vector store
            reembed_all_documents()
            # Then initialize the chatbot
            st.session_state.chatbot = SalesCounsellorBot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            logging.error(f"Error initializing chatbot: {str(e)}")
            st.session_state.chatbot = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
        st.session_state.conversation_manager.start_new_session()

def save_uploaded_file(uploaded_file, data_dir: Path) -> Path:
    """Save an uploaded file to the data directory."""
    try:
        # Create data directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logging.info(f"Saved uploaded file: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        raise

def process_and_embed_file(file_path: Path) -> bool:
    """Process and embed a single file."""
    try:
        # Process the file
        process_uploaded_file(file_path)
        logging.info(f"Successfully processed and embedded file: {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="Sales Counsellor Bot",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Check if chatbot initialization failed
    if st.session_state.chatbot is None:
        st.error("Failed to initialize the chatbot. Please check the logs and try again.")
        return

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.bot {
            background-color: #475063;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Management")
        
        # Document upload section
        st.markdown("### Upload Documents")
        st.markdown("Upload PDF or TXT files to add to the knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:
            data_dir = Path("data")
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            # Save the file
                            file_path = save_uploaded_file(uploaded_file, data_dir)
                            
                            # Process and embed the file
                            if process_and_embed_file(file_path):
                                st.session_state.uploaded_files.add(uploaded_file.name)
                                st.success(f"Successfully processed {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process {uploaded_file.name}")
                                # Remove the file if processing failed
                                file_path.unlink(missing_ok=True)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        logging.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # Display current documents
        st.markdown("### Current Documents")
        data_dir = Path("data")
        if data_dir.exists():
            for file_path in data_dir.glob("*"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(file_path.name)
                with col2:
                    if st.button("🗑️", key=f"delete_{file_path.name}"):
                        try:
                            # Delete the file
                            file_path.unlink()
                            # Remove from session state
                            st.session_state.uploaded_files.discard(file_path.name)
                            # Re-embed remaining documents
                            reembed_all_documents()
                            st.success(f"Deleted {file_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {file_path.name}: {str(e)}")
                            logging.error(f"Error deleting {file_path.name}: {str(e)}")

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chatbot.clear_history()
            # Archive current session and start new one
            st.session_state.conversation_manager.clear_current_session()
            st.rerun()

    # Main chat interface
    st.title("🤖 Sales Counsellor Bot")
    st.markdown("""
        Welcome! I'm your AI sales counsellor assistant. I can help you with:
        - Course information and details
        - Program comparisons
        - Pricing and payment options
        - Career guidance
        - And more!
        
        Feel free to ask any questions about our courses and programs.
    """)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Add to conversation history
        st.session_state.conversation_manager.add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.get_response(
                        prompt,
                        st.session_state.messages[:-1]  # Exclude the current message
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Add to conversation history
                    st.session_state.conversation_manager.add_message("assistant", response)
                except Exception as e:
                    error_message = "I apologize, but I encountered an error. Please try again."
                    st.error(error_message)
                    logging.error(f"Error getting response: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    # Add error to conversation history
                    st.session_state.conversation_manager.add_message("assistant", error_message)

if __name__ == "__main__":
    main() 