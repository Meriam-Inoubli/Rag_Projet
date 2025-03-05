import os
import uuid
import time
import logging
import asyncio
import vertexai
from PIL import Image
import streamlit as st
from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.services.dataset_service import DatasetServiceClient

# Désactiver l'inspection des modules
st.config.get_option("server.enableCORS")
st.config.get_option("server.enableXsrfProtection")

st.set_page_config(page_title="CareBot", layout="wide", page_icon="🩺")

# Configuration des chemins et des imports locaux
from images import IMAGE_PATH
from lib.feedback import save_feedback, display_feedback_analysis
from lib.callbacks import (
    feedback_callback,
    regenerate_callback,
    initialize_qa_chain,
    get_default_response,
    evaluation_callback,
)
from config import PROJECT_ID, REGION
from eval import display_evaluation_page

# Configuration du logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialisation des services Google Cloud
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

# Chargement du logo
logo = Image.open(os.path.join(IMAGE_PATH, "logo.png")).resize((200, 100))

# Variables de session
if "page" not in st.session_state:
    st.session_state["page"] = "Care Bot"
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Comment puis-je vous aider aujourd'hui ?"}
    ]

if "feedback_data" not in st.session_state:
    st.session_state["feedback_data"] = {}
if "show_feedback_modal" not in st.session_state:
    st.session_state["show_feedback_modal"] = False
if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""
if "last_response" not in st.session_state:
    st.session_state["last_response"] = ""
if "last_duree_reponse" not in st.session_state:
    st.session_state["last_duree_reponse"] = 0

# Fonction asynchrone pour générer une réponse
async def generate_response(qa_chain, prompt):
    try:
        logging.info(f"Prompt envoyé à la chaîne QA : {prompt}")
        response = await qa_chain.ainvoke({"query": prompt})
        logging.info(f"Réponse reçue de la chaîne QA : {response}")
        if not response or not response.get("result"):
            logging.error("La réponse générée est None ou vide.")
            return None
        return response
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse : {e}")
        return None

# Fonction principale asynchrone
async def main():
    # Barre latérale
    with st.sidebar:
        st.image(logo)
        st.markdown("## 🏥 Navigation")
        st.session_state["page"] = st.radio(
            "Sélectionnez une page :",
            ["Care Bot", "Évaluation", "Voir les feedbacks"],
            index=["Care Bot", "Évaluation", "Voir les feedbacks"].index(st.session_state["page"]),
        )
        st.markdown("---")
        st.markdown("### Consignes d'Utilisation 📝")
        st.warning(
            """
            **💡 Conseils pour interagir avec Care Bot :**
            - Les réponses générées sont basées sur des données disponibles et peuvent nécessiter une vérification supplémentaire.
            - Ce chatbot utilise des sources fiables sur le cancer du sein, mais **ne remplace pas un avis médical professionnel**.
            """
        )
        st.caption("Made with ❤️ by CareBot Team")

    # Affichage des pages
    if st.session_state["page"] == "Care Bot":
        st.title("🤖 Care Bot")
        st.markdown(
            "### Bienvenue sur Care Bot, votre assistant médical virtuel spécialisé en oncologie."
        )
        st.markdown("#### Posez-moi vos questions !")
        qa_chain = initialize_qa_chain()

        # Zone d'entrée utilisateur (en bas de la page)
        prompt = st.chat_input("💬 Posez votre question ici...")

        # Affichage des messages du chatbot
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Traitement de la nouvelle question
        if prompt:
            # Ajouter la question à l'historique des messages
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Afficher la question dans l'interface utilisateur
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("CareBot réfléchit..."):
                start_time = time.time()
                response = await generate_response(qa_chain, prompt)

                if response is None:
                    logging.error("La chaîne QA a échoué. Tentative de réinitialisation...")
                    response = await generate_response(qa_chain, prompt)

                if response is not None and response.get("result"):
                    answer = response["result"]
                    duree_reponse = time.time() - start_time

                    if "source_documents" in response and response["source_documents"]:
                        best_doc = max(
                            response["source_documents"],
                            key=lambda doc: doc.metadata.get("similarity_score", 0)
                        )
                        similarity_score = best_doc.metadata.get("similarity_score", 0)

                        # Vérifier si le score de similarité est supérieur ou égal à 0.6
                        if similarity_score >= 0.65:
                            answer += f"\n\n**Source :** {best_doc.metadata.get('source', 'N/A')}\n"
                            answer += f"\n**Focus Area :** {best_doc.metadata.get('focus_area', 'N/A')}\n"
                            answer += f"\n**Similarity Score :** {similarity_score}\n"
                            answer += f"\n**Similarity Type :** {best_doc.metadata.get('similarity_type', 'N/A')}"
                        else:
                            logging.info("\nScore de similarité inférieur à 0.6. Afficher uniquement la réponse du modèle.")
                    else:
                        logging.info("\nNo relevant sources found.")
                else:
                    logging.error("La réponse générée est None ou vide. Vérifiez l'état du chaînage QA.")
                    answer = get_default_response(prompt)
                    duree_reponse = 0

            # Ajouter la réponse à l'historique des messages
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Afficher la réponse dans l'interface utilisateur
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Mettre à jour les variables de session
            st.session_state["last_question"] = prompt
            st.session_state["last_response"] = answer
            st.session_state["last_duree_reponse"] = duree_reponse

            # Afficher les boutons sous la réponse
            col1, col2, col3 = st.columns(3)
            if col1.button("Feedback", key=f"feedback_{uuid.uuid4()}", on_click=feedback_callback):
                pass

            if col2.button("Regénérer la réponse", key=f"regenerate_{uuid.uuid4()}", on_click=regenerate_callback):
                pass

            if col3.button("Évaluation", key=f"evaluation_{uuid.uuid4()}", on_click=evaluation_callback):
                pass

        # Afficher la fenêtre modale pour le feedback
        if st.session_state.get("show_feedback_modal", False):
            with st.form(key="feedback_form"):
                st.markdown("### Donnez votre avis")
                nombre_etoiles = st.slider("Notez CareBot (1 à 5 étoiles) :", 1, 5, 3)
                feedback_text = st.text_area("Laissez un commentaire...")
                if st.form_submit_button("Envoyer"):
                    save_feedback(
                        question=st.session_state["feedback_data"]["question"],
                        reponse=st.session_state["feedback_data"]["reponse"],
                        feedback_text=feedback_text,
                        duree_reponse=st.session_state["feedback_data"]["duree_reponse"],
                        nombre_etoiles=nombre_etoiles
                    )
                    st.success("Merci pour votre feedback !")
                    st.session_state["show_feedback_modal"] = False

    elif st.session_state["page"] == "Évaluation":
        display_evaluation_page()

    elif st.session_state["page"] == "Voir les feedbacks":
        st.title("💬 Feedbacks")
        display_feedback_analysis()

# Point d'entrée pour exécuter l'application
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
