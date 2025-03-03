import os
from langchain_google_cloud_sql_pg import PostgresVectorStore, PostgresEngine
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
from config import (
    PROJECT_ID, 
    REGION, 
    INSTANCE, 
    DATABASE, 
    DB_USER
)

load_dotenv()
DB_PASSWORD = os.environ["DB_PASSWORD"]


def get_embedding_model(logger) -> VertexAIEmbeddings:
    """
    Retrieves VertexAI embeddings.
    """
    # Initialize VertexAIEmbeddings without directly passing project_id
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest", project=PROJECT_ID)
    return embeddings
    

def create_cloud_sql_database_connection() -> PostgresEngine:

    engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID,
        instance=INSTANCE,
        region=REGION,
        database=DATABASE,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    return engine



async def get_vector_store(engine: PostgresEngine, embedding: VertexAIEmbeddings) -> PostgresVectorStore:

    vector_store = PostgresVectorStore.create_sync(
        engine=engine,
        table_name="MI_RAG",
        embedding_service=embedding,
    )

    return vector_store

