import chromadb 
import google.generativeai as genai  
from embeddings import generate_text_embedding  

# Initialize ChromaDB persistent client with a local database path
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Get or create a collection named "documents" to store and retrieve embeddings
collection = chroma_client.get_or_create_collection(name="documents")

def retrieve_relevant_data(query_text):
    """
    Retrieve the most relevant document from ChromaDB using text embeddings.
    
    Args:
        query_text (str): The user query for which relevant documents are retrieved.
    
    Returns:
        str: The most relevant document's text if found, otherwise None.
    """
    query_embedding = generate_text_embedding(query_text)  # Generate embedding for the query text

    # Perform a similarity search in ChromaDB
    results = collection.query(  # Searches ChromaDB for the closest matching document embedding
        query_embeddings=[query_embedding],  # Query using the generated embedding
        n_results=3  # Retrieve the top 3 most relevant documents
    )  # ChromaDB uses Approximate Nearest Neighbor (ANN) Search // calculates the cosine similarity

    # Check if any document was retrieved
    if results["ids"]:
        return results["metadatas"][0][0]["raw_text"]  # Extract the stored raw text
    
    return None  # Return None if no relevant data is found

def ask_gemini(query_text, response_mode):
    """
    Retrieve relevant data and decide whether to call Gemini LLM based on user selection.

    Args:
        query_text (str): The user query to process.
        response_mode (str): "db_only" for DB-only responses, "db_gemini" for LLM fallback.

    Returns:
        str: The response generated based on user selection.
    """
    relevant_data = retrieve_relevant_data(query_text)  # Retrieve relevant document text

    # If response mode is "DB Only" and no relevant data is found, return a simple message
    if response_mode == "db_only":
        return relevant_data if relevant_data else "No relevant data found."

    # If response mode is "DB + Gemini", use Gemini if no DB data is found
    if not relevant_data:
        relevant_data = "No relevant document found in the database."

    # Construct a prompt with both the user query and retrieved relevant data
    prompt = f"User query: {query_text}\nRelevant information: {relevant_data}"
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")  
    response = model.generate_content(prompt)  # Generate a response using the prompt

    return response.text  # Return the generated response
