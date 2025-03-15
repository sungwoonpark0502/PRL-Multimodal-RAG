import chromadb
import pdfminer.high_level
import fitz  # PyMuPDF for extracting images
import pytesseract
from PIL import Image
import io
from embeddings import generate_text_embedding

# Initialize ChromaDB persistent client with local storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection named "documents" to store embeddings
collection = chroma_client.get_or_create_collection(name="documents")

def store_data(data_id, raw_text, embedding):
    """
    Stores text input or file data in ChromaDB.

    Args:
        data_id (str): Unique identifier for the data.
        raw_text (str): The original text content.
        embedding (list): The generated embedding vector for the text.
    """
    collection.add(
        ids=[data_id],  # Store with a unique identifier
        embeddings=[embedding],  # Store the corresponding embedding vector
        metadatas=[{"raw_text": raw_text}]  # Store the original text as metadata
    )

def extract_images_from_pdf(pdf_path):
    """
    Extracts images from a PDF and applies OCR to extract text.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from images in the PDF.
    """
    extracted_text = ""

    # Open the PDF file
    pdf_doc = fitz.open(pdf_path)

    for page_num in range(len(pdf_doc)):  
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)  # Extract all images on the page

        for img_index, img in enumerate(image_list):
            xref = img[0]  # Image reference ID
            base_image = pdf_doc.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Get image bytes
            image = Image.open(io.BytesIO(image_bytes))  # Convert to PIL image

            # Apply OCR to extract text from the image
            text_from_image = pytesseract.image_to_string(image)
            extracted_text += f"\n[Image {img_index+1} on Page {page_num+1}]: {text_from_image}\n"

    return extracted_text.strip()

def process_and_store_file(file_path):
    """
    Processes a file, extracts its content, generates an embedding, and stores it.

    Args:
        file_path (str): Path to the file being processed.

    Returns:
        str: Success message or an error if the file type is unsupported.
    """
    # Extract file extension to determine processing method
    file_extension = file_path.split(".")[-1].lower()

    # Process text files (.txt)
    if file_extension in ["txt"]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()  # Read the file content
        embedding = generate_text_embedding(raw_text)  # Generate text embedding

    # Process PDF files (.pdf) with both text and image extraction
    elif file_extension in ["pdf"]:
        # Extract text from the PDF
        raw_text = pdfminer.high_level.extract_text(file_path)

        # Extract images and apply OCR if needed
        image_text = extract_images_from_pdf(file_path)

        # Combine extracted text with OCR results
        full_text = raw_text + "\n" + image_text if image_text else raw_text

        # Generate embedding for the combined text
        embedding = generate_text_embedding(full_text)

    # Process image files (.png, .jpg, .jpeg)
    elif file_extension in ["png", "jpg", "jpeg"]:
        from embeddings import generate_image_embedding  # Import image embedding function
        with open(file_path, "rb") as f:
            image_data = f.read()  # Read image as binary data
        embedding = generate_image_embedding(image_data)  # Generate image embedding
        full_text = "Extracted text stored in embedding."  # Placeholder for metadata

    else:
        return "Unsupported file type!"  # Handle unsupported file types

    # Store extracted data and embedding in ChromaDB
    store_data(file_path, full_text, embedding)

    return f"{file_path} processed successfully!"  # Return success message
