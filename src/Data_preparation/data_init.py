import os 
import sys
import uuid
import logging
import json
import aiohttp
import asyncio
from langchain_core.documents.base import Document
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from lib.metadata import extract_focus_area
from lib.config import TABLE_NAME
from lib.transformer import split_pdfs
from lib.cloud_SQL import (
    create_cloud_sql_database_connection,
    create_table_if_not_exists
)
from lib.embedding import (
    get_embeddings,
    load_documents_from_local,
    generate_batches,
)

DATA_DIRECTORY = r"C:\Users\MSI\Projet_GenAI\Data"

async def main():
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

    # Step 1: Connect to the Cloud SQL database
    try:
        logger.info("Connecting to the Cloud SQL database...")
        engine = create_cloud_sql_database_connection()
        # await create_table_if_not_exists(engine)
    except Exception as e:
        logger.error(f"Error while connecting to the database: {e}")
        return

    # Step 2: Initialize the embedder (VertexAIEmbeddings)
    try:
        logger.info("Initializing VertexAI embedder...")
        embeddings = get_embeddings(logger)
    except Exception as e:
        logger.error(f"Error while initializing the embedder: {e}")
        return

    # Step 3: Load documents from the DATA directory
    documents = load_documents_from_local(r'C:\Users\MSI\Projet_GenAI\Data', logger)
    if not documents:
        logger.error("❌ No documents found in the Data directory.")
        return

    # Step 4: Split documents into chunks
    try:
        logger.info("Splitting documents into chunks...")
        splitted_documents = split_pdfs(documents, chunk_size=1000, chunk_overlap=200)
        batch_size = 5
        batches = list(generate_batches(splitted_documents, batch_size))
        logger.info(f"✅ {len(splitted_documents)} chunks created successfully.")
        logger.info(f"✅ {len(batches)} batches of {batch_size} chunks each.")
    except Exception as e:
        logger.error(f"❌ Error while splitting documents into chunks: {e}")
        return

    # Step 5: Extract chunks (texts) for embeddings
    chunks = [doc.page_content for doc in splitted_documents]

    # Step 6: Create the PostgresVectorStore
    try:
        logger.info("Creating PostgresVectorStore...")
        vector_store = PostgresVectorStore.create_sync(
            engine=engine,
            table_name="MI_RAG",
            embedding_service=embeddings,
        )
        logger.info("✅ PostgresVectorStore created successfully.")
    except Exception as e:
        logger.error(f"❌ Error while creating PostgresVectorStore: {e}")
        sys.exit(1)

    # Step 7: Loop through chunks to insert data
    try:
        logger.info("Processing chunks and inserting into the database...")
        documents_to_insert = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate a UUID for the chunk
                langchain_id = str(uuid.uuid4())

                # Extract the focus area for the chunk
                focus_area = extract_focus_area(chunk, logger)

                # Prepare metadata
                metadata = splitted_documents[i].metadata
                langchain_metadata = {
                    "source": metadata.get("source", "unknown"),
                    "focus_area": focus_area
                }

                # Create a Document object
                document = Document(
                    page_content=chunk,  # Chunk text
                    metadata=langchain_metadata  # Metadata
                )
                documents_to_insert.append(document)
                logger.info(f"✅ Chunk {i} prepared successfully: {langchain_id}")

                # Add a delay between requests to avoid quota limits
                await asyncio.sleep(1) 

                #Add documents 
                logger.info("Adding documents to the vector store...")
                vector_store.add_documents(documents_to_insert)
                logger.info("✅ All chunks inserted successfully.")

            except Exception as e:
                logger.error(f"❌ Error while adding documents: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error while processing chunks: {e}")
        return

if __name__ == "__main__":
    asyncio.run(main())