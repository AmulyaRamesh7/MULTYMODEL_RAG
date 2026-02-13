import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
GROQ_MODEL = "llama3-70b-8192"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
