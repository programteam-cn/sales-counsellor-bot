# Coding Ninjas - AI Sales Counsellor Bot

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit%20%7C%20LangChain-orange)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-green)
![Database](https://img.shields.io/badge/VectorDB-Qdrant-red)

This project is an advanced AI-powered chatbot designed to act as **"Alisha,"** an intelligent and personable sales counsellor for Coding Ninjas. It uses a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate, context-aware answers from a custom knowledge base, ensuring that prospective students receive the most relevant and persuasive information about courses, placements, and career outcomes.

The application is built with a user-friendly **Streamlit** interface, enabling real-time chat and dynamic management of the knowledge base through document uploads.

## Key Features

-   **Advanced RAG Architecture**: Goes beyond a standard LLM's knowledge by retrieving information directly from your documents (brochures, sales guides, FAQs) before generating an answer. This minimizes hallucinations and ensures responses are tailored to Coding Ninjas' specific offerings.
-   **Dynamic Knowledge Management**: The Streamlit sidebar allows administrators to upload new PDF or TXT files to the knowledge base on the fly. Documents can also be deleted, and the knowledge base will automatically re-index to reflect the changes.
-   **Persona-Driven Interaction**: The bot embodies the persona of "Alisha," a friendly, enthusiastic, and convincing sales counsellor, as defined by a detailed system prompt. This ensures a consistent and engaging user experience.
-   **Conversational Memory**: Maintains the context of the ongoing conversation, allowing for natural follow-up questions and a coherent dialogue flow.
-   **Natural Information Gathering**: Mimics a real sales counsellor by subtly asking clarifying questions to understand a user's background (e.g., career stage, experience) to provide more personalized and effective advice.

## How It Works: A Deep Dive into the Architecture

This project is not a simple chatbot. It's a sophisticated RAG system that bridges the gap between a powerful language model and your proprietary data.

### The "Why" - Technical Decisions Explained

#### Phase 1: Knowledge Ingestion (`embeddings/embed_docs.py`)

This is the process of teaching the bot. It converts your unstructured documents into a structured, searchable library.

1.  **Document Loading**: We use LangChain's `PyPDFLoader` and `TextLoader`.
    *   **Why?** These are robust, community-vetted tools for parsing text content from the most common document formats, preserving metadata where possible.

2.  **Text Splitting**: The loaded text is split into smaller chunks using `RecursiveCharacterTextSplitter`.
    *   **Why?** LLMs have a limited context window. Sending a whole document is inefficient. This splitter is intelligent—it tries to split text along semantic boundaries (paragraphs, sentences) before making hard cuts. The `chunk_overlap` ensures that context isn't lost between chunks.

3.  **Embedding Creation**: Each text chunk is converted into a high-dimensional vector using Google's `models/embedding-001`.
    *   **Why?** An embedding is a numerical representation of semantic meaning. Text with similar meanings will have vectors that are "close" to each other in vector space. This is the magic that enables semantic search. Google's model is chosen for its performance and its synergy with the Gemini LLM.

4.  **Vector Storage**: The embeddings (vectors) and their corresponding text chunks are stored in **Qdrant**.
    *   **Why Qdrant?** A standard database can't efficiently search for "semantic closeness." Qdrant is a specialized vector database built for extremely fast and scalable similarity searches using algorithms like HNSW (Hierarchical Navigable Small World). It's the engine that powers our retrieval.

#### Phase 2: Inference (`chat/rag_chat.py`)

This is what happens every time a user sends a message.

1.  **Query Embedding**: The user's question is converted into a vector using the *same* `embedding-001` model.
    *   **Why?** To find relevant documents, the query must be in the same vector space as the stored chunks.

2.  **Similarity Search (Retrieval)**: LangChain, via the Qdrant client, takes the query vector and searches for the 'k' most similar document vectors in the database.
    *   **How?** It uses a distance metric (like Cosine Similarity) to find the text chunks that are semantically closest to the user's question. These retrieved chunks form the "context."

3.  **Prompt Augmentation**: This is the most critical step. A master prompt is constructed using:
    *   **The System Persona**: The detailed instructions from `prompts/base_prompt.txt` that tell the LLM how to behave as "Alisha."
    *   **Retrieved Context**: The relevant document chunks found in the previous step.
    *   **Chat History**: The last few turns of the conversation, providing short-term memory.
    *   **The User's Question**: The latest query from the user.

4.  **LLM Generation**: The final, augmented prompt is sent to Google's `gemini-pro` LLM.
    *   **Why?** The LLM now has everything it needs: its personality (`Persona`), the relevant facts (`Context`), the conversation flow (`History`), and the user's need (`Question`). It uses this to generate a response that is both factually grounded in the provided documents and conversationally appropriate.

## Technology Stack

| Component           | Technology                               | Why it was chosen                                                                                                                                                                                            |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **App Framework**   | `Streamlit`                              | Perfect for rapidly building data-centric and interactive web apps with pure Python. Ideal for demos and internal tools with features like chat interfaces and file uploaders built-in.                        |
| **AI Orchestration**| `LangChain`                              | The industry-standard framework for gluing together components of an LLM application. It simplifies the RAG pipeline, from document loading to managing prompts and interacting with the vector store.        |
| **LLM & Embeddings**| `Google Generative AI` (Gemini, embedding-001) | Provides a powerful, state-of-the-art LLM (Gemini) for high-quality text generation and a high-performance model for creating text embeddings. A cohesive and well-supported ecosystem.                |
| **Vector Database** | `Qdrant`                                 | A high-performance, open-source vector database built for speed and scale. It's purpose-built for the fast similarity searches required for RAG and offers advanced filtering capabilities.                  |
| **Configuration**   | `python-dotenv`                          | A simple and effective way to manage secrets and configuration (like API keys) by loading them from a `.env` file, keeping them out of the source code.                                                   |
| **Document Parsing**| `pypdf`                                  | A reliable and pure-Python library for extracting text from PDF documents, which are a primary format for brochures and detailed guides.                                                                     |

## Project Structure

```
.
├── 📄 .env.example          # Example environment variables file
├── 📄 .gitignore
├── 📄 README.md
├── 📄 app.py                # Main Streamlit web application entrypoint
├── 📄 requirements.txt      # List of Python dependencies
├── 📄 run_chat.py           # Entrypoint for the command-line chat interface
├── 📂 chat/
│   └── 📄 rag_chat.py       # Core logic for the RAG chain, memory, and response generation
├── 📂 config/
│   └── 📄 config.py         # Loads and validates environment variables
├── 📂 data/                  # Directory to store your knowledge base documents (PDFs, TXT)
├── 📂 embeddings/
│   └── 📄 embed_docs.py     # Script for processing and embedding documents into Qdrant
└── 📂 prompts/
    └── 📄 base_prompt.txt     # The master system prompt defining the bot's persona and rules
```

## Setup and Installation

Follow these steps to run the Sales Counsellor Bot locally.

### 1. Prerequisites

-   Python 3.9+
-   An active **Google AI API Key**. You can get one from [Google AI Studio](https://makersuite.google.com/).
-   A **Qdrant instance**. You can use the free [Qdrant Cloud tier](https://cloud.qdrant.io/) or run it locally via Docker. You will need the **Cluster URL** and an **API Key**.

### 2. Clone the Repository

```bash
git clone https://github.com/programteam-cn/sales-counsellor-bot.git
cd sales-counsellor-bot
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to avoid package conflicts.

```bash
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install all required packages using pip.

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file by copying the example file.

```bash
cp .env.example .env
```

Now, open the `.env` file and add your secret keys:

```dotenv
# Your API key from Google AI Studio
GOOGLE_API_KEY="AIzaSy..."

# Your Qdrant Cloud URL
QDRANT_URL="https://your-unique-id.cloud.qdrant.io:6333"

# Your Qdrant API Key
QDRANT_API_KEY="your-qdrant-secret-key"
```

### 6. Add Knowledge Documents

Place any PDF or TXT files you want the bot to learn from into the `data/` directory. Create the directory if it doesn't exist.

### 7. Run the Initial Data Ingestion

You must "teach" the bot by embedding your documents. Run the following script. This will process all files in the `data/` directory and store them in Qdrant.

```bash
python embeddings/embed_docs.py
```

Watch the console for logs. You should see it creating a collection named `sales_counsellor` and embedding your documents.

### 8. Run the Application

You have two ways to interact with the bot.

#### Option A: Streamlit Web App (Recommended)

This provides a full graphical interface for chatting and managing documents.

```bash
streamlit run app.py
```

Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

#### Option B: Command-Line Interface

For a simpler, terminal-based chat experience:

```bash
python run_chat.py
```

## How to Use the App

1.  **Chatting**: Simply type your questions in the input box and press Enter. The conversation history is maintained for your session.
2.  **Uploading Documents**: Use the sidebar in the Streamlit app to upload new PDF or TXT files. The file will be automatically processed and added to the knowledge base.
3.  **Deleting Documents**: Click the trash can icon next to a document in the sidebar to delete it. **Important**: Deleting a file triggers a full re-embedding of all remaining documents to ensure the knowledge base is clean and consistent. This may take a few moments.
4.  **Clearing Chat**: Click the "Clear Chat History" button to reset the current conversation.
