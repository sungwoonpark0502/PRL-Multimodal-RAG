# rag_gemini.py
import google.generativeai as genai
from retrieval import TextRetriever
from PIL import Image
import requests
from io import BytesIO

# Configure Gemini API
genai.configure(api_key="-----")

class GeminiRAG:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-pro-vision")
        self.retriever = TextRetriever()
        self.retriever.create_index([
            "Gemini AI is great for text processing.",
            "It also handles images.",
            "RAG improves LLM accuracy."
        ])

    def generate_response(self, query, image_url=None):
        """Generates a response from Gemini using retrieved text and optional image."""
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join(retrieved_docs)

        if image_url:
            # Load image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            input_data = [query, image]
        else:
            input_data = f"Context: {context}\n\nQuestion: {query}"

        response = self.model.generate_content(input_data)
        return response.text

# Example usage
if __name__ == "__main__":
    rag = GeminiRAG()
    
    # Text-based query
    print("Response:", rag.generate_response("How does Gemini process images?"))

  