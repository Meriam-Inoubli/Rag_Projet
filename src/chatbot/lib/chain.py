from typing import Optional, List
from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA
from langchain_google_cloud_sql_pg import PostgresVectorStore
from pydantic import BaseModel
import logging
import asyncio
import datetime as dt

from lib.embeddings import (
    create_cloud_sql_database_connection, 
    get_embedding_model, 
    get_vector_store
)
from lib.retriever import get_relevant_documents
from lib.model import get_llm
from lib.prompt import get_prompt


class CustomRetriever(BaseRetriever, BaseModel):
    """
    A custom retriever that fetches relevant documents from a PostgresVectorStore
    based on a similarity threshold.
    """
    vector_store: PostgresVectorStore
    similarity_threshold: float

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: A list of relevant documents with similarity scores and types.
        """
        try:
            # Appeler la fonction async de manière synchrone
            relevant_docs = asyncio.run(
                get_relevant_documents(
                    query=query,
                    vector_store=self.vector_store,
                    similarity_threshold=self.similarity_threshold
                )
            )
            # Convertir les dictionnaires en objets Document
            documents = [
                Document(
                    page_content=doc["content"],
                    metadata={
                        **doc["metadata"],
                        "similarity_score": doc["similarity_score"],
                        "similarity_type": doc["similarity_type"]
                    }
                )
                for doc in relevant_docs
            ]
            return documents
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []


async def get_chain(
    vector_store: PostgresVectorStore,
    similarity_threshold: float = 0.5,
    max_output_tokens: int = 716,
    temperature: float = 0.1,
) -> Optional[RetrievalQA]:
    """
    Creates and returns a RetrievalQA chain for answering questions.

    Args:
        vector_store (PostgresVectorStore): The vector store used to retrieve documents.
        similarity_threshold (float): The similarity threshold for filtering documents.
        max_output_tokens (int): The maximum number of tokens for the LLM's response.
        temperature (float): The temperature parameter for the LLM.

    Returns:
        RetrievalQA: A configured RetrievalQA instance.
    """
    try:
        logging.info(f"New agent created at {dt.datetime.now()}")

        # Create a custom retriever
        retriever = CustomRetriever(
            vector_store=vector_store,
            similarity_threshold=similarity_threshold
        )

        # Initialize the language model (LLM)
        llm = get_llm(
            max_output_tokens=max_output_tokens,
            temp=temperature
        )

        # Create the RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": get_prompt()},
            return_source_documents=True,
        )

        return qa
    except Exception as e:
        logging.error(f"Error creating the chain: {e}")
        return None


async def main():
    # Initialize the vector store (replace with your actual initialization code)
    engine = create_cloud_sql_database_connection()
    embeddings = get_embedding_model(engine)
    vector_store = await get_vector_store(engine, embeddings)

    # Create the RetrievalQA chain
    qa_chain = await get_chain(
        vector_store=vector_store,
        similarity_threshold=0.5,
        max_output_tokens=512,
        temperature=0.1
    )

    if not qa_chain:
        print("Failed to create the QA chain.")
        return

    # Define the query
    query = "c'est quoi un cancer du sein?"

    # Use the chain to answer the question
    response = await qa_chain.ainvoke(query)  

    # Display the answer
    print("Answer:", response["result"])

    # Trouver le document avec le score de similarité le plus élevé
    if response["source_documents"]:
        best_doc = max(
            response["source_documents"],
            key=lambda doc: doc.metadata["similarity_score"]
        )
        print("\nBest Source:")
        print(f"- Source: {best_doc.metadata['source']}")
        print(f"- Focus Area: {best_doc.metadata['focus_area']}")
        print(f"- Similarity Score: {best_doc.metadata['similarity_score']}")
        print(f"- Similarity Type: {best_doc.metadata['similarity_type']}")
    else:
        print("\nNo relevant sources found.")


if __name__ == "__main__":
    asyncio.run(main())
