from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableLambda


def get_llm(max_output_tokens: int = 512, temp: float = 0.1):
    """
    Retourne un modèle de langage (LLM) configuré.

    Args:
        max_output_tokens (int): Le nombre maximum de tokens pour la réponse.
        temp (float): Le paramètre de température pour le LLM.

    Returns:
        Un modèle de langage compatible avec LangChain.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=temp,
        max_output_tokens=max_output_tokens,
        timeout=None,
        max_retries=2,
    )
    return llm