from keybert import KeyBERT
import logging

# Initialize KeyBERT
kw_model = KeyBERT()

def extract_focus_area(content: str, logger: logging.Logger) -> str:
    """
    Uses KeyBERT to extract the most relevant keyword or key phrase.

    Args:
        content (str): The content of the document.
        logger (logging.Logger): Logger to record errors.

    Returns:
        str: The most relevant keyword or key phrase.
    """
    try:
        truncated_content = content[:500]  

        # Extract keywords
        keywords = kw_model.extract_keywords(
            truncated_content,
            keyphrase_ngram_range=(1, 2),  
            stop_words="english", 
            top_n=1  
        )

        if keywords:
            return keywords[0][0]  
        return "general"  
    except Exception as e:
        logger.error(f"❌ Error while extracting focus_area: {e}")
        return "Cancer"