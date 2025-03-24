import os
import chromadb
import pdfminer.high_level
import fitz  # PyMuPDF for handling PDFs
import pytesseract  # OCR for extracting text from images
from PIL import Image  # Image processing
import io  # Handle binary file data
import whisper  # OpenAI Whisper for audio/video transcription
import google.generativeai as genai  # Google Gemini LLM API
from embeddings import generate_text_embedding  # Custom function for text embedding
import re  # Regex for text processing
import uuid  # Generate unique IDs to prevent duplicates
import subprocess  # To convert video to audio

# Initialize ChromaDB client with persistent local storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def store_data(data_id, raw_text, embedding, metadata=""):
    """
    Stores text data in ChromaDB with a unique identifier, vector embeddings, and metadata.
    """
    if not raw_text.strip():
        print(f"Skipping {data_id}: Empty raw text.")
        return

    if not embedding or len(embedding) == 0:
        print(f"Skipping {data_id}: Empty embedding.")
        return

    if not isinstance(embedding[0], list):
        embedding = [embedding]  # Ensure embedding is 2D

    unique_id = f"{data_id}_{uuid.uuid4().hex[:8]}"  # Prevent duplicate IDs
    try:
        collection.add(
            ids=[unique_id],
            embeddings=embedding,
            metadatas=[{"raw_text": raw_text, "metadata": metadata}]
        )
        print(f"Stored data with ID: {unique_id}")
    except Exception as e:
        print(f"Error storing data with ID {unique_id}: {e}")

### FILE PROCESSING FUNCTIONS ###

def extract_text_from_pdf(file_path):
    """ Extracts text from a PDF using PyMuPDF. """
    return pdfminer.high_level.extract_text(file_path)

def extract_text_from_txt(file_path):
    """ Reads text from a .txt file. """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(file_path):
    """ Extracts text from an image using OCR (Tesseract). """
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)

def convert_video_to_audio(video_path):
    """ Converts video to an audio file using ffmpeg. Returns the audio path. """
    audio_path = video_path.replace(".mp4", ".wav")
    try:
        subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"], check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return None

def transcribe_audio(file_path):
    """
    Transcribes an audio file using OpenAI's Whisper model.
    Returns transcribed text.
    """
    model = whisper.load_model("base")  # Load Whisper ASR model
    result = model.transcribe(file_path)
    return result["text"]

def process_and_store_file(file_path, metadata=""):
    """
    Processes a file (PDF, TXT, Image, Audio, Video), extracts text,
    summarizes it using agentic chunking, generates embeddings, and stores in ChromaDB.
    """
    file_extension = file_path.split(".")[-1].lower()
    raw_text = ""

    if file_extension == "txt":
        raw_text = extract_text_from_txt(file_path)
    elif file_extension == "pdf":
        raw_text = extract_text_from_pdf(file_path)
    elif file_extension in ["png", "jpg", "jpeg", "tiff"]:
        raw_text = extract_text_from_image(file_path)
    elif file_extension in ["mp3", "wav"]:
        raw_text = transcribe_audio(file_path)
    elif file_extension in ["mp4", "avi", "mov"]:
        audio_path = convert_video_to_audio(file_path)
        if audio_path:
            raw_text = transcribe_audio(audio_path)

    if not raw_text.strip():
        return "No content extracted from file!", None

    chunks = agentic_chunk_text(raw_text)
    chunk_data = []
    for idx, c in enumerate(chunks):
        emb = generate_text_embedding(c)
        if emb:
            store_data(f"{file_path}_chunk_{idx}", c, emb, metadata)
            chunk_data.append({"chunk": c, "embedding": emb})

    return raw_text, chunk_data

### GEMINI LLM FUNCTIONS ###

def extract_topic(text):
    """
    Extracts the main topic using Gemini LLM.
    If response is generic, uses regex fallback.
    """
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    entity_prompt = (
        "Extract the **main topic** from the following text.\n\n"
        f"{text}"
    )
    try:
        entity_response = model.generate_content(entity_prompt)
        entity_name = entity_response.text.strip() if entity_response else ""
        return entity_name if entity_name else extract_topic_fallback(text)
    except Exception as e:
        print(f"[extract_topic] Error: {e}")
        return extract_topic_fallback(text)

def extract_topic_fallback(text):
    """ Extracts a topic based on capitalized phrases. """
    matches = re.findall(r'\b[A-Z][a-z]*\b(?:\s+[A-Z][a-z]*)*', text)
    return matches[0] if matches else "Unknown Topic"

def agentic_chunk_text(text):
    """
    Uses Gemini LLM to extract topics from the provided text and generate bullet points.
    There are two cases:
      Case 1: If the text revolves around one main topic, each bullet must be in the format:
                <Main Topic>, <Sub Topic>: <Concise summary>
      Case 2: If the text contains multiple independent topics, each bullet must be in the format:
                <Topic>: <Concise summary>
    Only output the bullet points in the specified format.
    """
    text = text.strip()
    if not text:
        return []
    
    summarize_prompt = f"""
    You are an advanced text summarization agent. The user has provided the following text:

    {text}

    First, determine if the text is primarily about one main topic or about multiple independent topics.

    If it is primarily about one main topic (Case 1), generate bullet points using this exact format:
        <Main Topic>, <Sub Topic>: <Summary of key details about that subtopic>
    For example, if the text is about a product called "Apple iPhone 16", your output might look like:
        Apple iPhone 16, iPhone 16: Features the new A18 chip, enhanced camera capabilities, and design refinements.
        Apple iPhone 16, A18 Chip: Provides substantial performance and efficiency improvements for gaming and AI tasks.
        Apple iPhone 16, Camera System: Includes upgraded sensors and computational photography features.
        Apple iPhone 16, Design Refinements (iPhone 16 Pro): Offers thinner display borders and a titanium build.
        Apple iPhone 16, Intelligence: Powers AI processing for overall performance enhancements.

    If the text contains multiple independent topics (Case 2), generate bullet points using this exact format:
        <Topic>: <Summary of key details about that topic>
    For example, if the text states "James likes basketball but Sam likes soccer", your output might be:
        James: James likes basketball.
        Sam: Sam likes soccer.

    Only output the summarized bullet points in the specified format and nothing else.
    """
    
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(summarize_prompt)
        if response and hasattr(response, "text"):
            bullet_text = response.text.strip()
            lines = bullet_text.split("\n")
            cleaned_chunks = []
            for line in lines:
                # Clean the line by removing extra bullet markers and spaces.
                line = line.strip().lstrip("-•*").strip()
                if line:
                    if ":" not in line:
                        line = f"General: {line}"
                    cleaned_chunks.append(line)
            return cleaned_chunks
    except Exception as e:
        print(f"[agentic_chunk_text] Error: {e}")
    
    return naive_sentence_fallback(text)

def naive_sentence_fallback(text):
    import re
    sentences = re.split(r'[.!?]\s+', text.strip())
    return [f"General: {sent.strip()}" for sent in sentences if sent.strip()]


### TEXT PROCESSING FUNCTION ###
def process_and_store_text(user_text, metadata=""):
    """
    Processes text input, dynamically generates bullet-point summaries,
    embeds, and stores in ChromaDB.
    """
    user_text = user_text.strip()
    if not user_text:
        return "No text provided!", None

    chunks = agentic_chunk_text(user_text)
    chunk_data = []
    for idx, c in enumerate(chunks):
        emb = generate_text_embedding(c)
        if emb:
            store_data(f"text_input_chunk_{idx}", c, emb, metadata)
            chunk_data.append({"chunk": c, "embedding": emb})
    return user_text, chunk_data
