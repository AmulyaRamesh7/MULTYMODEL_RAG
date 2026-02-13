import requests
import numpy as np
from config import JINA_API_KEY, JINA_EMBEDDING_URL

def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        JINA_EMBEDDING_URL,
        headers=headers,
        json={
            "model": "jina-embeddings-v4",
            "input": texts
        }
    )

    data = response.json()
    return np.array([item["embedding"] for item in data["data"]])
