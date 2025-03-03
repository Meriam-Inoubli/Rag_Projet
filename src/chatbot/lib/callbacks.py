import streamlit as st
import asyncio
import logging
import time 

# Importation des fonctions personnalisées
from lib.embeddings import (
    create_cloud_sql_database_connection,
    get_embedding_model,
    get_vector_store
)

from lib.chain import get_chain


# Callbacks pour les boutons
def feedback_callback():
    st.session_state["feedback_data"] = {
        "nature_feedback": "neutre",
        "question": st.session_state["last_question"],
        "reponse": st.session_state["last_response"],
        "duree_reponse": st.session_state["last_duree_reponse"]
    }
    st.session_state["show_feedback_modal"] = True

def regenerate_callback():
    if st.session_state["last_question"]:
        # Récupérer la dernière question
        prompt = st.session_state["last_question"]

        # Réinitialiser les messages pour éviter les doublons
        st.session_state.messages.pop()

        # Générer une nouvelle réponse
        if st.session_state["qa_chain"]:
            try:
                with st.spinner("CareBot réfléchit..."):
                    start_time = time.time()
                    response = asyncio.run(st.session_state["qa_chain"].ainvoke({"query": prompt}))
                    answer = response["result"]  
                    duree_reponse = time.time() - start_time

                    # Vérifier si des documents sources sont disponibles
                    if "source_documents" in response and response["source_documents"]:
                        best_doc = max(
                            response["source_documents"],
                            key=lambda doc: doc.metadata.get("similarity_score", 0)
                        )

                        # Ajouter des informations supplémentaires à la réponse
                        answer += f"\n\n**Source :** {best_doc.metadata.get('source', 'N/A')}\n"
                        answer += f"\n**Focus Area :** {best_doc.metadata.get('focus_area', 'N/A')}\n"
                        answer += f"\n**Similarity Score :** {best_doc.metadata.get('similarity_score', 'N/A')}\n"
                        answer += f"\n**Similarity Type :** {best_doc.metadata.get('similarity_type', 'N/A')}"

            except Exception as e:
                st.error(f"Erreur lors de la régénération de la réponse : {e}")
                logging.error(f"Erreur lors de la régénération de la réponse : {e}")
                answer = get_default_response(prompt)
                duree_reponse = 0
        else:
            answer = get_default_response(prompt)
            duree_reponse = 0

        # Ajouter la nouvelle réponse au chat
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Mettre à jour les valeurs pour le feedback
        st.session_state["last_response"] = answer

def evaluation_callback():
    st.session_state["page"] = "Évaluation"


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
