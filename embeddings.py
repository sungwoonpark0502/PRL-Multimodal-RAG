import google.generativeai as genai  
import numpy as np  
from PIL import Image  
import pytesseract  
import io 
import config  

def generate_text_embedding(text):
    """
    Generate text embeddings using the Gemini API.
    
    Args:
        text (str): The input text to be embedded.
    
    Returns:
        list: A list representing the generated text embedding.
    """
    response = genai.embed_content(
        model="models/text-embedding-004",  # Specify the embedding model
        content=text  # Provide the input text for embedding
    )
    return response["embedding"]  # Return the generated embedding vector
