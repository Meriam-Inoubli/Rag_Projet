import time
import os
import sys
import logging
import numpy as np
import functools
from PyPDF2 import PdfReader
from .config import PROJECT_ID
from langchain.schema import Document
from typing import Generator, List, Tuple
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed



# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_embeddings(logger) -> VertexAIEmbeddings:
    """
    Retrieves VertexAI embeddings.
    """
    try:
        # Initialize VertexAIEmbeddings without directly passing project_id
        embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project=PROJECT_ID)
        logger.info("✅ VertexAI embeddings successfully retrieved (model: textembedding-gecko@latest).")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Error retrieving embeddings: {e}")
        raise


def load_documents_from_local(directory: str, logger: logging.Logger) -> list[Document]:
    """
    Loads PDF documents from a local directory.

    Args:
        directory (str): Path to the directory containing PDF files.
        logger (logging.Logger): Logger to record errors.

    Returns:
        list[Document]: List of documents with content and metadata.
    """
    documents = []
    try:
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if the file is a PDF
            if os.path.isfile(file_path) and filename.endswith(".pdf"):
                try:
                    # Use PyPDF2 to read the PDF file
                    reader = PdfReader(file_path)
                    content = ""
                    
                    # Extract text from each page
                    for page in reader.pages:
                        content += page.extract_text()
                    
                    # Create a LangChain Document object with metadata
                    metadata = {"source": filename} 
                    document = Document(page_content=content, metadata=metadata)
                    documents.append(document)
                    logger.info(f"✅ Document loaded: {filename}")
                except Exception as e:
                    logger.error(f"❌ Error reading file {filename}: {e}")
        
        logger.info(f"✅ {len(documents)} documents loaded from directory '{directory}'.")
    except Exception as e:
        logger.error(f"❌ Error loading documents: {e}")
    
    return documents


def encode_texts_to_embeddings(embedding_model, texts: List[str], logger):
    """Encodes a list of texts into embeddings."""
    try:
        if not texts:
            logger.warning("No text provided to generate embeddings.")
            return [None] * len(texts)
        
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        return [None] * len(texts)


def generate_batches(chunks: List[str], batch_size: int) -> Generator[List[str], None, None]:
    """Yield batches of chunks for processing."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i: i + batch_size]


def encode_text_to_embedding_batched(
    embedding_model: TextEmbeddingModel, 
    chunks: List[str], 
    logger, 
    batch_size: int = 5
) -> Tuple[List[bool], np.ndarray]:
    """Process text batches and return embeddings with success status."""
    embeddings_list: List[List[float]] = []
    is_successful: List[bool] = []
    
    # Split chunks into batches
    batches = list(generate_batches(chunks, batch_size))
    logger.info(f"✅ {len(batches)} batches prepared for processing.")
    
    # Limit request rate (e.g., 10 requests per second)
    seconds_per_job = 1 / 10

    # Process batches in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in batches:
            future = executor.submit(
                functools.partial(encode_texts_to_embeddings, embedding_model, batch, logger)
            )
            futures.append(future)
            time.sleep(seconds_per_job)  # Limit request rate

        # Retrieve results from futures
        for future in as_completed(futures):
            try:
                embeddings = future.result()
                embeddings_list.extend(embeddings)
                is_successful.extend([embedding is not None for embedding in embeddings])
            except Exception as e:
                logger.error(f"❌ Error generating embeddings: {e}")
                is_successful.extend([False] * len(batch))  # Mark all chunks in the batch as failed

    # Filter valid embeddings
    embeddings_list_successful = [embedding for embedding in embeddings_list if embedding is not None]

    # Check if any embeddings were successfully generated
    if not embeddings_list_successful:
        logger.error("❌ No embeddings generated successfully.")
        return is_successful, np.array([])  # Return an empty array

    # Convert the list of embeddings into a NumPy array
    embeddings_array = np.squeeze(np.stack(embeddings_list_successful))

    # Handle the case where there is only one embedding
    if embeddings_array.ndim == 1:
        embeddings_array = [embeddings_array]

    logger.info(f"✅ {len(embeddings_array)} embeddings successfully generated.")
    return is_successful, embeddings_array
