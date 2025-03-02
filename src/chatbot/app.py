import os
import uuid
import logging
import asyncio
import vertexai
import streamlit as st
st.set_page_config(page_title="CareBot", layout="wide", page_icon="🩺")

from datetime import datetime
from google.cloud import aiplatform
from PIL import Image
from images import IMAGE_PATH
import time

# Importation des fonctions personnalisées
from lib.embeddings import (
    create_cloud_sql_database_connection,
    get_embedding_model,
    get_vector_store
)

from lib.chain import get_chain
from lib.model import get_llm
from lib.prompt import get_prompt
from lib.feedback import save_feedback, display_feedback_analysis  # Import des nouvelles fonctions

from config import PROJECT_ID, REGION

from eval import display_evaluation_page
# Configuration de la page

# Configuration du logging
logging.basicConfig(filename="app.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialisation des services Google Cloud
vertexai.init(project=PROJECT_ID, location=REGION)
aiplatform.init(project=PROJECT_ID, location=REGION)



# Chargement du logo
logo = Image.open(os.path.join(IMAGE_PATH, "logo.png")).resize(( 200, 100))

# Variables de session
if "page" not in st.session_state:
    st.session_state["page"] = "Care Bot"
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider aujourd'hui ?"}]
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "feedback_data" not in st.session_state:
    st.session_state["feedback_data"] = {}
if "show_feedback_modal" not in st.session_state:
    st.session_state["show_feedback_modal"] = False  # État pour afficher la fenêtre modale

# Fonction pour initialiser la QA Chain
@st.cache_resource(show_spinner=False)
def initialize_qa_chain():
    try:
        engine = create_cloud_sql_database_connection()
        embeddings = get_embedding_model(engine)
        vector_store = asyncio.run(get_vector_store(engine, embeddings))
        qa_chain = asyncio.run(get_chain(vector_store=vector_store))
        return qa_chain
    except Exception as e:
        st.error(f"Échec de l'initialisation du chatbot : {e}")
        logging.error(f"Échec de l'initialisation du chatbot : {e}")
        return None

def get_default_response(prompt: str) -> str:
    """
    Retourne une réponse par défaut si le chatbot n'est pas initialisé ou en cas d'erreur.
    """
    default_responses = [
        "Je suis désolé, je ne peux pas répondre à votre question pour le moment. Veuillez réessayer plus tard.",
        "Je rencontre des difficultés techniques. Pouvez-vous reformuler votre question ?",
        "Je suis en cours de configuration. Posez-moi votre question plus tard !",
        "Je ne suis pas en mesure de répondre pour le moment. Merci de votre patience.",
    ]
    return default_responses[len(prompt) % len(default_responses)] 

with st.sidebar:
    st.image(logo)
    st.markdown("## Navigation 🏥")
    st.session_state["page"] = st.radio(
        "Sélectionnez une page :",
        ["Care Bot", "Évaluation", "Voir les feedbacks"],
        index=["Care Bot", "Évaluation", "Voir les feedbacks"].index(st.session_state["page"]),
    )
    st.markdown("---")
        # Consignes pour les utilisateurs
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
    # Initialisation de la QA Chain
    if st.session_state["qa_chain"] is None:
        st.session_state["qa_chain"] = initialize_qa_chain()

    # Zone d'entrée utilisateur (en bas de la page)
    prompt = st.text_input("💬 Posez votre question ici...")

    # Affichage des messages du chatbot (au-dessus de la zone de texte)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role_icon = "👤" if message["role"] == "user" else "🤖"
            st.markdown(f"{role_icon} **{message['role'].capitalize()} :** {message['content']}")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Affichage du message utilisateur
        st.markdown(f"👤 **Vous :** {prompt}")
        
        if st.session_state["qa_chain"]:
            try:
                # Générer une réponse avec le QA chain
                with st.spinner("CareBot réfléchit..."):
                    start_time = time.time()
                    response = asyncio.run(st.session_state["qa_chain"].ainvoke({"query": prompt}))
                    answer = response["result"]  # La réponse générée par le modèle
                    duree_reponse = time.time() - start_time

                    # Vérifier si des documents sources sont disponibles
                    if "source_documents" in response and response["source_documents"]:
                        # Trouver le document avec le score de similarité le plus élevé
                        best_doc = max(
                            response["source_documents"],
                            key=lambda doc: doc.metadata.get("similarity_score", 0)  # Utiliser get pour éviter les erreurs si la clé est manquante
                        )
                        
                        # Ajouter des informations supplémentaires à la réponse
                        answer += f"\n\n**Source :** {best_doc.metadata.get('source', 'N/A')}\n"
                        answer += f"**Focus Area :** {best_doc.metadata.get('focus_area', 'N/A')}\n"
                        answer += f"**Similarity Score :** {best_doc.metadata.get('similarity_score', 'N/A')}\n"
                        answer += f"**Similarity Type :** {best_doc.metadata.get('similarity_type', 'N/A')}"

                        # Afficher les informations de débogage dans la console
                        logging.info("\nBest Source:")
                        logging.info(f"- Source: {best_doc.metadata.get('source', 'N/A')}")
                        logging.info(f"- Focus Area: {best_doc.metadata.get('focus_area', 'N/A')}")
                        logging.info(f"- Similarity Score: {best_doc.metadata.get('similarity_score', 'N/A')}")
                        logging.info(f"- Similarity Type: {best_doc.metadata.get('similarity_type', 'N/A')}")
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

        # Ajouter la réponse du chatbot
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"🤖 **CareBot :** {answer}")

        # Afficher les boutons sous la réponse
        col1, col2, col3 = st.columns(3)
        if col1.button("Feedback", key=f"feedback_{uuid.uuid4()}"):
            st.session_state["feedback_data"] = {
                "nature_feedback": "neutre",  
                "question": prompt,
                "reponse": answer,
                "duree_reponse": duree_reponse
            }
            st.session_state["show_feedback_modal"] = True  

        if col2.button("Regénérer la réponse", key=f"regenerate_{uuid.uuid4()}"):
            # Logique pour regénérer la réponse
            st.session_state.messages.pop()  
            st.experimental_rerun()  

        if col3.button("Évaluation", key=f"evaluation_{uuid.uuid4()}"):
            st.session_state["page"] = "Évaluation"  # Rediriger vers la page d'évaluation
            st.experimental_rerun()  # Relancer le script pour mettre à jour la page

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