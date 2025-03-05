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

async def regenerate_callback():
    if st.session_state["last_question"]:
        prompt = st.session_state["last_question"]
        st.session_state.messages.pop()

        if st.session_state["qa_chain"]:
            try:
                with st.spinner("CareBot réfléchit..."):
                    start_time = time.time()
                    # Utilisez await pour appeler generate_response
                    response = await generate_response(st.session_state["qa_chain"], prompt)
                    if response and response.get("result"):
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

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.chat_message("assistant"):
                            st.markdown(answer)

                        st.session_state["last_response"] = answer
                    else:
                        st.error("Erreur lors de la régénération de la réponse.")
            except Exception as e:
                st.error(f"Erreur lors de la régénération de la réponse : {e}")
                logging.error(f"Erreur lors de la régénération de la réponse : {e}")

def evaluation_callback():
    st.session_state["page"] = "Évaluation"

@st.cache_resource(show_spinner=False)
def initialize_qa_chain():
    try:
        logging.info("Connexion à la base de données...")
        engine = create_cloud_sql_database_connection()
        logging.info("Chargement du modèle d'embedding...")
        embeddings = get_embedding_model(engine)
        logging.info("Création du vector store...")
        vector_store = asyncio.run(get_vector_store(engine, embeddings))  
        logging.info("Création de la chaîne QA...")
        qa_chain = asyncio.run(get_chain(vector_store=vector_store))  
        logging.info("Chaîne QA initialisée avec succès.")
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
