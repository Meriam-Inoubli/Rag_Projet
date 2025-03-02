import unittest
from unittest.mock import patch
from langchain_google_genai import ChatGoogleGenerativeAI
from src.chatbot.lib.model import get_llm  

class TestGetLLM(unittest.TestCase):
    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_get_llm_default_params(self, mock_chat_google_genai):
        """
        Teste que la fonction `get_llm` retourne une instance de `ChatGoogleGenerativeAI`
        avec les paramètres par défaut.
        """
        # Appeler la fonction get_llm avec les paramètres par défaut
        llm = get_llm()

        # Vérifier que ChatGoogleGenerativeAI a été appelé avec les bons paramètres
        mock_chat_google_genai.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=0.1,
            max_output_tokens=512,
            timeout=None,
            max_retries=2,
        )

        # Vérifier que la fonction retourne bien une instance de ChatGoogleGenerativeAI
        self.assertIsInstance(llm, ChatGoogleGenerativeAI)

    @patch('langchain_google_genai.ChatGoogleGenerativeAI')
    def test_get_llm_custom_params(self, mock_chat_google_genai):
        """
        Teste que la fonction `get_llm` retourne une instance de `ChatGoogleGenerativeAI`
        avec des paramètres personnalisés.
        """
        max_output_tokens = 1024
        temp = 0.5

        llm = get_llm(max_output_tokens=max_output_tokens, temp=temp)

        # Vérifier que ChatGoogleGenerativeAI a été appelé avec les bons paramètres
        mock_chat_google_genai.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=temp,
            max_output_tokens=max_output_tokens,
            timeout=None,
            max_retries=2,
        )

        self.assertIsInstance(llm, ChatGoogleGenerativeAI)

if __name__ == "__main__":
    unittest.main()