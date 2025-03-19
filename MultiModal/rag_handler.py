import os
import base64
import uuid
import re
import io
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import shutil



CHROMA_PATH = "chroma_db"  # Path where Chroma stores data

load_dotenv()

# Initialize global variables
retriever = None
chain_multimodal_rag = None

def process_pdf(filepath):
    global retriever, chain_multimodal_rag

    # Clear existing ChromaDB data
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    

    # TODO by_title, by_page, by_section, by_paragraph, by_senetence, by_line, by_fixed_length
    # Extract elements
    
    # TODO by_title
    raw_pdf_elements = partition_pdf(
        filename=filepath,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=os.path.dirname(filepath),
    )

    print(raw_pdf_elements)

    # TODO hi_res
    # raw_pdf_elements = partition_pdf(
    #     filename=filepath,
    #     extract_images_in_pdf=True,
    #     infer_table_structure=True,
    #     chunking_strategy="basic",
    #     strategy="hi_res",
    #     max_characters=4000,
    #     new_after_n_chars=3800,
    #     combine_text_under_n_chars=2000,
    #     extract_image_block_types=["Image", "Table"],
    #     extract_image_block_to_payload=False,
    #     image_output_dir_path=os.path.dirname(filepath),
    # )   

    # TODO by_paragraph
    # raw_pdf_elements = partition_pdf(
    #     chunking_strategy="by_paragraph",  
    #     max_characters=4000,              
    #     combine_text_under_n_chars=2000,  
    #     new_after_n_chars=3800,          
    #     filename=filepath
    # )

    # Categorize elements
    texts, tables = [], []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    # Generate summaries
    text_summaries, table_summaries = generate_text_summaries(texts, tables)
    img_base64_list, image_summaries = generate_img_summaries(os.path.dirname(filepath))

    # Create retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    retriever = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list
    )

    # Create RAG chain
    chain_multimodal_rag = multi_modal_rag_chain(retriever)

    return texts, tables

def handle_query(question):
    if not chain_multimodal_rag:
        raise ValueError("Please upload and process a PDF first")
    return chain_multimodal_rag.invoke(question)

def encode_image(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Generate image summary using Gemini"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_base64}"
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    """Generate image summaries"""
    img_base64_list = []
    image_summaries = []
    prompt = """You are an assistant tasked with summarizing images for retrieval. 
    These summaries will be embedded and used to retrieve the raw image. 
    Give a concise summary optimized for retrieval."""

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries

def generate_text_summaries(texts, tables, summarize_texts=False):
    """Generate text summaries using Gemini"""
    prompt_text = """You are an assistant tasked with summarizing content for retrieval.
    Create concise summaries optimized for retrieval. Content: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = texts if not summarize_texts else []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """Create multi-vector retriever"""
    store = InMemoryStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key="doc_id",
    )
    

    def add_docs(summaries, contents):
        doc_ids = [str(uuid.uuid4()) for _ in contents]
        retriever.vectorstore.add_documents([
            Document(page_content=s, metadata={"doc_id": doc_ids[i]})
            for i, s in enumerate(summaries)
        ])
        retriever.docstore.mset(list(zip(doc_ids, contents)))

    if text_summaries: add_docs(text_summaries, texts)
    if table_summaries: add_docs(table_summaries, tables)
    if image_summaries: add_docs(image_summaries, images)

    return retriever

def split_image_text_types(docs):
    """Split images and text"""
    images = []
    texts = []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if looks_like_base64(content) and is_image_data(content):
            images.append(resize_base64_image(content))
        else:
            texts.append(content)
    return {"images": images, "texts": texts}

def img_prompt_func(data_dict):
    """Format prompt for Gemini"""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    content = [
        {"type": "text", "text": f"Question: {data_dict['question']}\nContext:\n{formatted_texts}"}
    ]
    
    for image in data_dict["context"]["images"]:
        content.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}"
        })
    
    return [HumanMessage(content=content)]

def multi_modal_rag_chain(retriever):
    """Create Gemini RAG chain"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # TODO need to know more about this part of the code
    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types), 
         "question": RunnablePassthrough()}
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain

# Utility functions remain the same
def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures)
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")