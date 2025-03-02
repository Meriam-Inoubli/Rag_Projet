from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

def split_pdfs(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200, logger: logging.Logger = None) -> list[Document]:
    """
    Splits documents into fixed-size chunks with optional overlap.

    Args:
        documents (list[Document]): List of documents to split.
        chunk_size (int): Maximum size of a chunk (in characters or tokens).
        chunk_overlap (int): Overlap between chunks (in characters or tokens).
        logger (logging.Logger): Logger to record errors.

    Returns:
        list[Document]: List of document chunks.
    """
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       
        chunk_overlap=chunk_overlap, 
        length_function=len,        
        separators=["\n\n", "\n", " ", ""]  
    )
    
    # Split documents into chunks
    splitted_documents = []
    for doc in documents:
        try:
            # Split the document content into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            # Create a new Document for each chunk
            for chunk in chunks:
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata  
                )
                splitted_documents.append(new_doc)
        except Exception as e:
            if logger:
                logger.error(f"❌ Error while splitting the document: {e}")
            continue
    
    if logger:
        logger.info(f"✅ {len(splitted_documents)} chunks successfully created.")
    return splitted_documents


def filter_non_relevant_chunks(chunks):
    """
    Filters out non-relevant chunks (e.g., table of contents, headers, footers).

    Args:
        chunks (list[Document]): List of document chunks.

    Returns:
        list[Document]: Filtered list of relevant chunks.
    """
    relevant_chunks = []
    for chunk in chunks:
        # Ignorer les chunks qui ressemblent à une table des matières
        if "Table des matières" not in chunk.page_content and "Sommaire" not in chunk.page_content:
            relevant_chunks.append(chunk)
    return relevant_chunks