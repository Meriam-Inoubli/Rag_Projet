import os
import ast  
import asyncio
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.outputs import LLMResult, ChatGeneration
import google.auth
from config import PROJECT_ID

# Suppress symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configurer l'API Google Vertex AI
config = {
    "project_id": PROJECT_ID,  
    "chat_model_id": "gemini-1.5-pro",  
    "embedding_model_id": "textembedding-gecko@latest",  
}

# Authentification Google
creds, _ = google.auth.default(quota_project_id=config["project_id"])

# Créer le modèle de langage et les embeddings avec Vertex AI
vertexai_llm = ChatVertexAI(
    credentials=creds,
    model_name=config["chat_model_id"],
)
vertexai_embeddings = VertexAIEmbeddings(
    credentials=creds,
    model_name=config["embedding_model_id"],
)

# Envelopper les modèles pour Ragas
# Create a custom is_finished_parser to capture Gemini generation completion signals
def gemini_is_finished_parser(response: LLMResult) -> bool:
    is_finished_list = []
    for g in response.flatten():
        resp = g.generations[0][0]

        # Check generation_info first
        if resp.generation_info is not None:
            finish_reason = resp.generation_info.get("finish_reason")
            if finish_reason is not None:
                is_finished_list.append(
                    finish_reason in ["STOP", "MAX_TOKENS"]
                )
                continue

        # Check response_metadata as fallback
        if isinstance(resp, ChatGeneration) and resp.message is not None:
            metadata = resp.message.response_metadata
            if metadata.get("finish_reason"):
                is_finished_list.append(
                    metadata["finish_reason"] in ["STOP", "MAX_TOKENS"]
                )
            elif metadata.get("stop_reason"):
                is_finished_list.append(
                    metadata["stop_reason"] in ["STOP", "MAX_TOKENS"] 
                )

        # If no finish reason found, default to True
        if not is_finished_list:
            is_finished_list.append(True)

    return all(is_finished_list)

vertexai_llm = LangchainLLMWrapper(vertexai_llm, is_finished_parser=gemini_is_finished_parser)
vertexai_embeddings = LangchainEmbeddingsWrapper(vertexai_embeddings)

# Configurer le LLM pour Ragas
evaluator_llm = vertexai_llm

# Charger un modèle d'embedding pour la similarité cosinus
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def calculate_cosine_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def calculate_precision_recall_f1(answer, reference):
    """
    Calcule les métriques de précision, rappel et F1-score.
    """
    answer_tokens = set(answer.lower().split())
    reference_tokens = set(reference.lower().split())

    true_positives = len(answer_tokens.intersection(reference_tokens))
    false_positives = len(answer_tokens - reference_tokens)
    false_negatives = len(reference_tokens - answer_tokens)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def convert_to_list(context):
    """
    Convertit une chaîne de caractères en liste si nécessaire.
    """
    if isinstance(context, str):
        if context.startswith("[") and context.endswith("]"):
            try:
                return ast.literal_eval(context)
            except (ValueError, SyntaxError):
                return [context]
        else:
            return [context]
    return context

async def evaluate_sample(sample):
    """
    Évalue un échantillon avec Ragas et retourne les résultats.
    """
    try:
        # Créer un dataset avec l'échantillon
        dataset = EvaluationDataset([sample])

        # Initialiser la métrique avec le LLM
        context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

        # Obtenir l'évaluation
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        # Convertir le résultat en dictionnaire
        result_df = result.to_pandas()
        result_dict = result_df.to_dict(orient="records")[0]

        # Calculer la précision du contexte
        context_precision_score = await context_precision.single_turn_ascore(HumanMessage(content=sample.user_input))

        return {
            "Faithfulness": result_dict["faithfulness"],
            "Answer Relevancy": result_dict["answer_relevancy"],
            "Context Precision": context_precision_score
        }
    except Exception as e:
        return {"error": str(e)}

def display_evaluation_page():
    st.title("📊 Évaluation du système RAG")
    st.markdown("### Évaluez les performances de votre système RAG en utilisant des métriques standards et spécifiques.")

    # Charger les données d'évaluation
    st.markdown("#### Charger les données d'évaluation")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV contenant les questions, les réponses générées, les réponses de référence et les contextes", type=["csv"])

    if uploaded_file is not None:
        evaluation_data = pd.read_csv(uploaded_file)
        evaluation_data["retrieved_contexts"] = evaluation_data["retrieved_contexts"].apply(convert_to_list)

        st.write("Aperçu des données chargées :")
        st.dataframe(evaluation_data.head())

        required_columns = ["input", "response", "reference", "retrieved_contexts"]
        if all(col in evaluation_data.columns for col in required_columns):
            if evaluation_data[required_columns].isnull().any().any():
                st.error("Le fichier CSV contient des valeurs manquantes. Veuillez vérifier les données.")
            else:
                st.success("Données chargées avec succès !")

                sample_size = st.slider("#### 🎯 Sélectionnez le nombre de questions à évaluer", min_value=1, max_value=len(evaluation_data), value=5)
                sampled_data = evaluation_data.sample(n=sample_size, random_state=42)

                for index, row in sampled_data.iterrows():
                    st.markdown(f"---")
                    st.markdown(f"### Question : **{row['input']}**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Réponse générée :**")
                        st.write(row["response"])
                    with col2:
                        st.markdown("**Réponse de référence :**")
                        st.write(row["reference"])

                    # Calculer les métriques basiques
                    similarity_score = calculate_cosine_similarity(row["reference"], row["response"])
                    precision, recall, f1 = calculate_precision_recall_f1(row["response"], row["reference"])

                    metrics_table = pd.DataFrame({
                        "Métrique": ["Cosine Similarity", "Precision", "Recall", "F1 Score"],
                        "Valeur": [similarity_score, precision, recall, f1]
                    })

                    # Afficher le tableau des métriques (toujours visible)
                    st.markdown("#### Métriques")
                    st.table(metrics_table)

                    # Préparer l'échantillon Ragas
                    sample = SingleTurnSample(
                        user_input=row["input"],
                        reference=row["reference"],
                        retrieved_contexts=row["retrieved_contexts"],
                        response=row["response"],
                    )

                    # Essayer de calculer et afficher les métriques Ragas
                    try:
                        with st.spinner("Calcul des métriques Ragas..."):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            ragas_metrics = loop.run_until_complete(evaluate_sample(sample))

                            if "error" in ragas_metrics:
                                st.error(f"Une erreur s'est produite lors du calcul des métriques Ragas : {ragas_metrics['error']}")
                            else:
                                # Afficher le graphique Ragas
                                st.markdown("#### Métriques Ragas")
                                fig, ax = plt.subplots()
                                bars = ax.bar(ragas_metrics.keys(), ragas_metrics.values(), color=["blue", "green", "orange"])
                                ax.set_ylabel("Score")
                                ax.set_title("Métriques Ragas")
                                ax.set_ylim(0, 1)

                                for bar in bars:
                                    height = bar.get_height()
                                    ax.annotate(f'{height:.2f}',
                                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                                xytext=(0, 3),
                                                textcoords="offset points",
                                                ha='center', va='bottom')

                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Une erreur s'est produite lors de la génération du graphique Ragas : {str(e)}")
        else:
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {required_columns}")
