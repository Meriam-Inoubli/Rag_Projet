import os
import uuid
import time
import logging
import asyncio
import vertexai
import nest_asyncio
from PIL import Image
import streamlit as st
st.set_page_config(page_title="CareBot", layout="wide", page_icon="🩺")
from images import IMAGE_PATH
from google.cloud import aiplatform

from lib.feedback import (
    save_feedback, 
    display_feedback_analysis 
)
from lib.callbacks import ( 
    feedback_callback,
    regenerate_callback,
    initialize_qa_chain,
    get_default_response,
    evaluation_callback
) 

from config import PROJECT_ID, REGION
from eval import display_evaluation_page

# Appliquer nest_asyncio pour résoudre les problèmes de boucle d'événements
nest_asyncio.apply()

# Configuration du logging
logging.basicConfig(filename="app.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialisation des services Google Cloud
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)

# Chargement du logo
logo = Image.open(os.path.join(IMAGE_PATH, "logo.png")).resize((200, 100))

# Variables de session
if "page" not in st.session_state:
    st.session_state["page"] = "Care Bot"
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider aujourd'hui ?"}]
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = initialize_qa_chain()
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
        response = await qa_chain.ainvoke({"query": prompt})
        return response
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la réponse : {e}")
        return None

# Fonction pour exécuter une tâche asynchrone dans la boucle d'événements
def run_async_task(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        logging.error(f"Erreur dans la boucle d'événements : {e}")
        return None
    finally:
        loop.close()

# Barre latérale
with st.sidebar:
    st.image(logo)
    st.markdown("## 🏥 Navigation ")
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

    # Zone d'entrée utilisateur (en bas de la page)
    prompt = st.chat_input("💬 Posez votre question ici...")

    # Affichage des messages du chatbot 
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state["qa_chain"]:
            try:
                with st.spinner("CareBot réfléchit..."):
                    start_time = time.time()
                    response = run_async_task(generate_response(st.session_state["qa_chain"], prompt))

                    if response is None:
                        logging.error("La réponse générée est None. Vérifiez l'état du chaînage QA.")
                        answer = get_default_response(prompt)
                        duree_reponse = 0
                    else:
                        answer = response["result"]
                        duree_reponse = time.time() - start_time

                        if "source_documents" in response and response["source_documents"]:
                            best_doc = max(
                                response["source_documents"],
                                key=lambda doc: doc.metadata.get("similarity_score", 0)
                            )
                            answer += f"\n\n**Source :** {best_doc.metadata.get('source', 'N/A')}\n"
                            answer += f"\n**Focus Area :** {best_doc.metadata.get('focus_area', 'N/A')}\n"
                            answer += f"\n**Similarity Score :** {best_doc.metadata.get('similarity_score', 'N/A')}\n"
                            answer += f"\n**Similarity Type :** {best_doc.metadata.get('similarity_type', 'N/A')}"
                        else:
                            logging.info("\nNo relevant sources found.")

            except Exception as e:
                st.error(f"Erreur lors de la génération de la réponse : {e}")
                logging.error(f"Erreur lors de la génération de la réponse : {e}")
                answer = get_default_response(prompt)
                duree_reponse = 0
        else:
            answer = get_default_response(prompt)
            duree_reponse = 0

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

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
