from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from process_data import process_and_store_file, process_and_store_text
from query import ask_gemini
import chromadb

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files")
    metadata = request.form.get("metadata", "")

    responses = []
    for file in files:
        if file.filename == "":
            continue
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        extracted_text, chunk_data = process_and_store_file(file_path, metadata)
        responses.append({
            "filename": filename,
            "extracted_text": extracted_text,
            "chunk_data": chunk_data
        })

    return jsonify(responses), 200

@app.route("/upload-text", methods=["POST"])
def upload_text():
    data = request.json
    user_text = data.get("text", "").strip()
    metadata = data.get("metadata", "").strip()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    extracted_text, chunk_data = process_and_store_text(user_text, metadata)
    return jsonify({
        "extracted_text": extracted_text,
        "chunk_data": chunk_data
    }), 200

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query_text = data.get("query", "").strip()
    response_mode = data.get("response_mode", "db_gemini")
    k = int(data.get("k", 3))

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:
        query_embedding, retrieved_chunk, retrieved_embedding, response_text = ask_gemini(query_text, response_mode, k)
        return jsonify({
            "query_embedding": query_embedding,
            "retrieved_chunk": retrieved_chunk,
            "retrieved_embedding": retrieved_embedding,
            "response": response_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reset-db", methods=["POST"])
def reset_db():
    """
    Deletes all stored documents from the ChromaDB "documents" collection.
    """
    try:
        results = collection.get()
        if "ids" in results and results["ids"]:
            all_ids = results["ids"]
            collection.delete(ids=all_ids)
        return jsonify({"message": "Database reset: all documents deleted."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/db-contents", methods=["GET"])
def show_db_contents():
    """
    Returns a list of stored documents from the ChromaDB "documents" collection.
    """
    try:
        results = collection.get()
        docs = []
        if "ids" in results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                docs.append({"id": doc_id, "metadata": meta})
        return jsonify({"documents": docs}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
