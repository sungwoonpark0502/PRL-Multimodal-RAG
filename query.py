import chromadb
import google.generativeai as genai
from embeddings import generate_text_embedding

# Initialize ChromaDB persistent client with local storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def re_rank_results(query_text, retrieved_chunks):
    """
    Uses Gemini LLM to re-rank retrieved chunks based on relevance.

    Args:
        query_text (str): The user's original query.
        retrieved_chunks (list): List of text chunks retrieved from ChromaDB.

    Returns:
        list: Re-ranked list of retrieved text chunks.
    """
    if not retrieved_chunks:
        return []

    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        # Construct a re-ranking prompt
        prompt = f"Given the query: '{query_text}', rank the following results from most to least relevant:\n"
        for i, chunk in enumerate(retrieved_chunks):
            prompt += f"{i+1}. {chunk}\n"

        response = model.generate_content(prompt)
        if not hasattr(response, "text"):
            # If no text in the response, fallback
            return retrieved_chunks

        ranked_list = response.text.split("\n")
        ranked_chunks = []
        for line in ranked_list:
            line = line.strip()
            try:
                # e.g. "1. ..." => index 0
                index = int(line.split(".")[0]) - 1
                ranked_chunks.append(retrieved_chunks[index])
            except Exception:
                # If we can't parse the line, ignore it
                continue

        # If we failed to parse any lines or it's empty, fallback to original
        return ranked_chunks if ranked_chunks else retrieved_chunks

    except Exception as e:
        print(f"[re_rank_results] Error: {e}")
        return retrieved_chunks

def retrieve_relevant_data(query_text, k=3, initial_retrieval_size=10):
    """
    Retrieve the top-K most relevant chunks from ChromaDB using text embeddings,
    then optionally re-rank them with Gemini.

    Args:
        query_text (str): The user query.
        k (int): Number of top relevant results to return.
        initial_retrieval_size (int): Number of results to retrieve before re-ranking.

    Returns:
        tuple: (query_embedding, top_k_chunks, top_k_embeddings)
    """
    query_embedding = generate_text_embedding(query_text)
    if not query_embedding:
        return [], [], []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=initial_retrieval_size
    )

    retrieved_chunks = []
    retrieved_embeddings = []

    if results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            # Extract the raw text from the metadatas field
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            chunk_text = meta.get("raw_text", "")
            chunk_emb = results["embeddings"][0][i] if results.get("embeddings") else None

            if chunk_text and chunk_emb:
                retrieved_chunks.append(chunk_text)
                retrieved_embeddings.append(chunk_emb)
            else:
                print(f"[WARNING] Skipping retrieval for index {i}: Missing text or embedding.")

    if not retrieved_chunks:
        print("[retrieve_relevant_data] No relevant chunks found.")
        return query_embedding, [], []

    return query_embedding, retrieved_chunks[:k], retrieved_embeddings[:k]

def ask_gemini(query_text, response_mode="db_gemini", k=3):
    """
    Retrieves relevant data from ChromaDB (up to k results), 
    optionally calls Gemini LLM to generate a final answer.

    Args:
        query_text (str): The user's query.
        response_mode (str): "db_only" for DB-only results, 
                             "db_gemini" for an LLM-based final answer.
        k (int): Number of top relevant results to retrieve.

    Returns:
        tuple: (query_embedding, ranked_chunks, ranked_embeddings, final_answer)
    """
    query_embedding, ranked_chunks, ranked_embeddings = retrieve_relevant_data(query_text, k)

    # DB-only mode: Just return the chunk text if any
    if response_mode == "db_only":
        if ranked_chunks:
            return (query_embedding, ranked_chunks, ranked_embeddings, "\n".join(ranked_chunks))
        else:
            return (query_embedding, [], [], "No relevant data found.")

    # DB + Gemini mode
    if not ranked_chunks:
        relevant_data = "No relevant document found in the database."
    else:
        relevant_data = "\n".join(ranked_chunks)

    prompt = f"User query: {query_text}\nRelevant information: {relevant_data}"

    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            final_answer = response.text
        else:
            final_answer = "Unexpected response format."
    except Exception as e:
        final_answer = f"Error: {str(e)}"

    return (query_embedding, ranked_chunks, ranked_embeddings, final_answer)
