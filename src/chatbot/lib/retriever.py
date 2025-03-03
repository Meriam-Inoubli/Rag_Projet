import sys
import asyncio
import logging
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_core.documents.base import Document
import aiohttp
from lib.source_retriever import list_top_k_sources  

from lib.config import (
    PROJECT_ID,
    REGION,
    TABLE_NAME,
    INSTANCE
)

from lib.embeddings import (
    create_cloud_sql_database_connection,
    get_embedding_model,
    get_vector_store,
)

async def get_relevant_documents(
    query: str, vector_store: PostgresVectorStore, similarity_threshold: float
) -> list[dict]:
    # Recherche des documents pertinents avec leurs scores de similarité
    relevant_docs_scores = vector_store.similarity_search_with_relevance_scores(
        query=query, k=4
    )

    # Affichez les documents pertinents pour déboguer
    #print("Documents pertinents trouvés :")
    #for doc, score in relevant_docs_scores:
        #print(f"Score : {score}, Contenu : {doc.page_content}")

    # Crée une liste de dictionnaires contenant les informations nécessaires
    relevant_docs = []
    for doc, score in relevant_docs_scores:
        if score >= similarity_threshold:  
            doc_info = {
                "content": doc.page_content,  
                "metadata": doc.metadata,   
                "similarity_score": score,   
                "similarity_type": "cosine"  
            }
            relevant_docs.append(doc_info)

    return relevant_docs

"""async def main():
    # Crée une session client
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
    async with aiohttp.ClientSession() as session:
        # Crée une connexion à la base de données Cloud SQL
        engine = create_cloud_sql_database_connection()

        # Initialise le modèle d'embedding
        embedding = get_embedding_model(logger)

        # Crée le vector store
        vector_store = await get_vector_store(engine, embedding)

        # Récupère les documents pertinents pour une requête
        query = "c'est quoi un cancer DU SEIN?"
        documents = await get_relevant_documents(query, vector_store, 0.1)

        # Affiche les documents pertinents
        print(f"Relevant documents for query: '{query}'")
        for doc_info in documents:
            print("Content:", doc_info["content"])
            print("Metadata:", doc_info["metadata"])
            print("Similarity Score:", doc_info["similarity_score"])
            print("Similarity Type:", doc_info["similarity_type"])
            print("-" * 50)  # Séparateur pour plus de clarté

        if documents:
            # Convertit les dictionnaires en objets Document pour list_top_k_sources
            source_documents = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in documents
            ]
            top_sources = list_top_k_sources(source_documents, k=3)
            print("\nTop Sources:")
            print(top_sources)

if __name__ == "__main__":
    # Exécute le code asynchrone
    asyncio.run(main())"""