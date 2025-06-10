import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_config():
    """Load and return configuration values from environment variables."""
    config = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION", "sales_counsellor")
    }
    
    # Validate required configuration
    missing_keys = [key for key, value in config.items() if value is None]
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return config

# For backward compatibility
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "sales_counsellor")
