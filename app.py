from flask import Flask, request, render_template, jsonify  
import os  
from werkzeug.utils import secure_filename 
import config 
from process_data import process_and_store_file, store_data 
from query import ask_gemini  
from embeddings import generate_text_embedding  

# Initialize Flask application
app = Flask(__name__)

# Define upload folder and ensure it exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define route for homepage with both GET and POST methods
@app.route("/", methods=["GET", "POST"])
def index():
    message = None  # Placeholder for user messages

    if request.method == "POST":
        # Handle file upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)  # Ensure safe file name
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)  # Save path
            file.save(file_path)  # Save uploaded file
            message = process_and_store_file(file_path)  # Process and store file data
        
        # Handle text input
        elif "text_input" in request.form and request.form["text_input"].strip() != "":
            user_text = request.form["text_input"].strip()  # Get user input text
            embedding = generate_text_embedding(user_text)  # Generate embedding for input text
            store_data(f"text_input_{len(user_text)}", user_text, embedding)  # Store data
            message = "Text input processed and stored successfully!"  # Confirmation message

    return render_template("index.html", message=message)  # Render the template with message

# Define API endpoint for handling user queries
@app.route("/query", methods=["POST"])
def query():
    data = request.json  # Get JSON data from request
    query_text = data.get("query")  # Extract query text
    response_mode = data.get("response_mode", "db_gemini")  # Default to "DB + Gemini" if not specified

    if not query_text:  # Return error if no query is provided
        return jsonify({"error": "No query provided"}), 400

    try:
        response_text = ask_gemini(query_text, response_mode)  # Pass response_mode to ask_gemini
        return jsonify({"response": response_text})  # Return response as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle any errors

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for development
