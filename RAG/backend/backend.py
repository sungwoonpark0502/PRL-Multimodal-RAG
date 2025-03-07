from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import google.generativeai as genai
from typing import Optional, List
import os
from pydantic import BaseModel
import time
import textwrap

app = FastAPI(max_request_size=1024 * 1024 * 50)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
client = MongoClient("mongodb://admin:password@mongodb:27017/")
db = client["rag_db"]
collection = db["files"]

# Gemini setup
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY"),
    transport="rest",
    client_options={"api_endpoint": "generativelanguage.googleapis.com"}
)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

PROMPT_TEMPLATE = textwrap.dedent("""\
**File Database Context:**
{context}

**Query:** {query}

**Answer Requirements:**
1. Match filenames exactly (case-sensitive)
2. If file exists, show filename AND content
3. Format as: "File: [name]\nContent: [text]"
4. If not found, say "No matching files found\"""")

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    text: Optional[str] = Form(None)
):
    try:
        allowed_types = [
            "application/pdf", "application/octet-stream",
            "image/jpeg", "image/png", "text/plain",
            "video/mp4", "video/quicktime"
        ]
        
        # Validate and correct MIME types
        if file.content_type not in allowed_types:
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                file.content_type = "application/pdf"
            elif filename.endswith(".txt"):
                file.content_type = "text/plain"
            else:
                raise HTTPException(400, "Invalid file type")

        file_content = await file.read()
        
        # Process text content with encoding fallback
        file_text = ""
        if file.content_type == "text/plain":
            try:
                file_text = file_content.decode("utf-8")
            except UnicodeDecodeError:
                file_text = file_content.decode("latin-1", errors="ignore")

        # Structure stored text (priority to file content)
        structured_text = ""
        if file.content_type == "text/plain":
            structured_text = f"File: {file.filename}\nContent: {file_text}"
            if text:
                structured_text += f"\nUser Description: {text}"
        else:
            structured_text = f"File: {file.filename}\nContent: {text or '[Binary file content]'}"

        document = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_content": file_content,
            "text": structured_text,
            "timestamp": time.time()
        }
        collection.insert_one(document)
        return {"message": "File + metadata stored successfully"}
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query")
async def query_gemini(request: QueryRequest):
    try:
        # Retrieve context with text filtering
        cursor = collection.find(
            {"text": {"$exists": True, "$ne": ""}},
            {"text": 1, "timestamp": 1, "_id": 0}
        ).sort("timestamp", -1).limit(15)
        
        # Build context string
        context_items = []
        for doc in cursor:
            if text := doc.get("text", "").strip():
                context_items.append(f"• {text}")
        
        context = "\n".join(context_items)[:10000]

        if not context:
            return {
                "response": "No relevant information found in database",
                "context_used": 0
            }
            
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            query=request.query,
            context=context
        )
        
        # Generate response with timeout
        response = model.generate_content(prompt, request_options={"timeout": 30})
        return {
            "response": response.text,
            "context_used": len(context_items)
        }
    
    except Exception as e:
        raise HTTPException(500, detail=f"Query failed: {str(e)}")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response

@app.get("/check-data")
async def check_data():
    count = collection.count_documents({})
    return {"document_count": count}

@app.delete("/delete-all")
async def delete_all_files():
    result = collection.delete_many({})
    return {"deleted_count": result.deleted_count}