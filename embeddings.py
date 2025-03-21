import google.generativeai as genai  
import config  

# Ensure API key is set
genai.configure(api_key=config.GEMINI_API_KEY)

def generate_text_embedding(text):
    """
    Generate text embeddings using the Gemini API and debug the response.
    
    Args:
        text (str): The input text to be embedded.
    
    Returns:
        list: A list representing the generated text embedding.
    """
    text = text.strip()
    if not text:
        return []  # Don't process empty text

    try:
        # Call embed_content using text-embedding-004
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        # Print entire response to see the actual keys (debugging)
        print("\n[DEBUG] embed_content response:", response)

        # The correct key might be 'embeddings', 'embedding', or something else.
        if "embeddings" in response and response["embeddings"]:
            # Usually it’s 'embeddings': [[0.1, 0.2, ...]]
            return response["embeddings"][0]
        elif "embedding" in response and response["embedding"]:
            # Possibly 'embedding': [0.1, 0.2, ...]
            return response["embedding"]
        else:
            print("[DEBUG] No 'embedding' or 'embeddings' found in response.")
            return []
        
    except Exception as e:
        print("[ERROR] generate_text_embedding:", e)
        return []
