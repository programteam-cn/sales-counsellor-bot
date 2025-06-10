import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from config.config import load_config

# Set up logging
logging.basicConfig(
    filename='chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_base_prompt() -> str:
    """Load the base prompt from the prompts folder."""
    try:
        prompt_path = Path("prompts/base_prompt.txt")
        with open(prompt_path, 'r') as f:
            prompt = f.read().strip()
            logging.info("Successfully loaded base prompt from file")
            return prompt
    except Exception as e:
        logging.error(f"Error loading base prompt: {str(e)}")
        raise

class SalesCounsellorBot:
    def __init__(self):
        self.config = load_config()
        self.initialize_chain()
        self.is_first_message = True  # Track if this is the first message in the conversation

    def initialize_chain(self):
        try:
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.config["GOOGLE_API_KEY"],
                temperature=0.7,
                convert_system_message_to_human=True
            )

            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.config["GOOGLE_API_KEY"]
            )

            # Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=self.config["QDRANT_URL"],
                api_key=self.config["QDRANT_API_KEY"]
            )

            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name="sales_counsellor",
                embedding=embeddings
            )

            # Verify vector store connection and collection
            try:
                collection_info = qdrant_client.get_collection("sales_counsellor")
                logging.info(f"Connected to Qdrant collection. Points count: {collection_info.points_count}")
            except Exception as e:
                logging.error(f"Error connecting to Qdrant collection: {str(e)}")
                raise

            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Load base prompt
            base_prompt = load_base_prompt()

            # Define the prompt template using the base prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{base_prompt}\n\nContext: {{context}}\n\nChat History: {{chat_history}}\n\nQuestion: {{question}}\n\nIs this the first message in the conversation? {{is_first_message}}"),
                ("human", "{question}")
            ])

            # Create the chain
            self.chain = prompt | llm

            logging.info("Chat chain initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing chat chain: {str(e)}")
            raise

    def is_basic_conversation(self, question: str) -> bool:
        """Check if the question is basic conversation or greeting."""
        basic_phrases = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "how are you", "how's it going", "what's up",
            "thanks", "thank you", "bye", "goodbye", "see you"
        ]
        question_lower = question.lower().strip()
        return any(phrase in question_lower for phrase in basic_phrases)

    def get_response(self, question: str, chat_history: List[Dict[str, Any]] = None) -> str:
        try:
            if chat_history is None:
                chat_history = []

            # Convert chat history to the format expected by the chain
            formatted_history = []
            for msg in chat_history:
                if msg["role"] == "user":
                    formatted_history.append(("human", msg["content"]))
                else:
                    formatted_history.append(("ai", msg["content"]))

            # For basic conversation, don't require context
            if self.is_basic_conversation(question):
                logging.info("Handling basic conversation query")
                chain_input = {
                    "context": "This is a basic conversation query.",
                    "chat_history": formatted_history,
                    "question": question,
                    "is_first_message": str(self.is_first_message)
                }
            else:
                # Search for relevant documents with logging
                logging.info(f"Searching for relevant documents for question: {question}")
                docs = self.vector_store.similarity_search(question, k=3)
                
                # Log the retrieved documents
                for i, doc in enumerate(docs):
                    logging.info(f"Retrieved document {i+1}:")
                    logging.info(f"Content: {doc.page_content[:200]}...")  # Log first 200 chars
                    if hasattr(doc.metadata, 'source'):
                        logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")

                context = "\n".join(doc.page_content for doc in docs) if docs else "No specific context available."
                logging.info(f"Combined context length: {len(context)} characters")

                # Prepare the input for the chain
                chain_input = {
                    "context": context,
                    "chat_history": formatted_history,
                    "question": question,
                    "is_first_message": str(self.is_first_message)
                }

            # Get response from the chain
            response = self.chain.invoke(chain_input)
            
            # Update first message flag
            self.is_first_message = False
            
            if hasattr(response, 'content'):
                logging.info(f"Generated response: {response.content[:200]}...")  # Log first 200 chars
                return response.content
            else:
                logging.warning(f"Unexpected response type: {type(response)}")
                return str(response)

        except Exception as e:
            logging.error(f"Error getting response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    def clear_history(self):
        """Clear the conversation history and reset first message flag"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.is_first_message = True  # Reset first message flag
