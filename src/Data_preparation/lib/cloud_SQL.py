import sys
import os 
import logging
import uuid 
from typing import Dict
from sqlalchemy import text
from sqlalchemy import Column
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.documents.base import Document
from langchain_google_cloud_sql_pg import PostgresEngine
from .metadata import extract_focus_area

# Configuration
from .config import (
    PROJECT_ID,
    REGION,
    INSTANCE,
    DATABASE,
    DB_USER,
    DB_PASSWORD,
    TABLE_NAME
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),  # Logs to a file
        logging.StreamHandler(sys.stdout)  # Logs to the console
    ]
)
logger = logging.getLogger(__name__)


def create_cloud_sql_database_connection() -> PostgresEngine:
    """
    Establishes a connection to the Cloud SQL database using SQLAlchemy.
    """
    try:
        engine = PostgresEngine.from_instance(
            project_id=PROJECT_ID,
            instance=INSTANCE,
            region=REGION,
            database=DATABASE,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info(" ✅ Successfully connected to the Cloud SQL database.")
        return engine
    except Exception as e:
        logger.error(f"❌ Error connecting to the database: {e}")
        raise


async def create_table_if_not_exists(engine: PostgresEngine) -> None:
    """
    Creates the table `MI_RAG` in the database if it does not exist.
    Uses the `init_vectorstore_table` method from PostgresEngine.
    """
    try:
        await engine.ainit_vectorstore_table(
            table_name="MI_RAG",
            vector_size=768,
        )
    except ProgrammingError:
        print("Table already created")
        logger.info("✅ Table MI_RAG verified/created successfully.")
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating the table: {e}")      


async def insert_into_sql(engine, data: Dict) -> None:
    """
    Inserts data into the SQL table asynchronously.

    Args:
        engine: Connection to the PostgreSQL database (SQLAlchemy AsyncEngine object).
        data (dict): Dictionary containing the data to be inserted.
            Must include the following keys:
            - langchain_id (str): UUID of the document.
            - content (str): Content of the text chunk.
            - embedding (list): Embedding vector as a list.
            - langchain_metadata (dict): Metadata in JSON format.
    """
    try:
        # SQL query for insertion
        query = text("""
            INSERT INTO MI_RAG_Table (langchain_id, content, embedding, langchain_metadata)
            VALUES (:langchain_id, :content, :embedding, :langchain_metadata)
        """
        )

        # Execute the query asynchronously
        async with engine.connect() as connection:
            await connection.execute(query, {
                "langchain_id": data["langchain_id"],
                "content": data["content"],
                "embedding": data["embedding"],
                "langchain_metadata": data["langchain_metadata"]
            })
            await connection.commit()
        logger.info(f"✅ Document successfully inserted: {data['langchain_id']}")
    except Exception as e:
        logger.error(f"❌ Error inserting data: {e}")


def process_langchain_documents(documents: list[Document], engine, embeddings) -> None:
    """
    Processes a list of LangChain documents and inserts them into the Cloud SQL table.

    Args:
        documents (List[Document]): List of LangChain documents.
        engine: Connection to the PostgreSQL database (SQLAlchemy Engine object).
        embeddings: Embedding model (e.g., VertexAIEmbeddings).
    """
    for doc in documents:
        try:
            # Extract document content and metadata
            content = doc.page_content
            metadata = doc.metadata

            # Generate a UUID for the document
            langchain_id = str(uuid.uuid4())

            # Generate embedding for the content
            embedding = embeddings.embed_documents([content])
            if not embedding:
                logger.warning(f"⚠️ No embedding generated for document: {langchain_id}")
                continue
            embedding = embedding[0]  # Take the first embedding from the list

            # Determine focus_area using KeyBERT
            focus_area = extract_focus_area(content)

            # Prepare metadata
            langchain_metadata = {
                "source": metadata.get("source", "unknown"),  # Document source
                "focus_area": focus_area  # Topic determined by KeyBERT
            }

            # Prepare data for insertion
            data = {
                "langchain_id": langchain_id,
                "content": content,
                "embedding": embedding,
                "langchain_metadata": langchain_metadata
            }

            # Insert data into SQL table
            insert_into_sql(engine, data)
        except Exception as e:
            logger.error(f"❌ Error processing document: {e}")
