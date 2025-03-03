import csv
import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Chemin du fichier CSV pour stocker les feedbacks
FEEDBACK_FILE = "feedbacks.csv"

# En-têtes du fichier CSV
CSV_HEADERS = [
    "timestamp", "nature_feedback", "question", "reponse", 
    "feedback_text", "duree_reponse", "nombre_etoiles"
]

def save_feedback(question: str, reponse: str, feedback_text: str, duree_reponse: float, nombre_etoiles: int):
    """
    Sauvegarde le feedback dans un fichier CSV.
    """
    if nombre_etoiles in [1, 2]:
        nature_feedback = "negative"
    elif nombre_etoiles == 3:
        nature_feedback = "neutre"
    else:  # 4 ou 5 étoiles
        nature_feedback = "positive"

    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([ 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            nature_feedback,
            question,
            reponse,
            feedback_text,
            duree_reponse,
            nombre_etoiles
        ])

def load_feedbacks():
    """
    Charge les feedbacks depuis le fichier CSV.
    """
    if os.path.exists(FEEDBACK_FILE):
        feedbacks = pd.read_csv(FEEDBACK_FILE)
        required_columns = ["timestamp", "nature_feedback", "feedback_text", "nombre_etoiles"]
        missing_columns = [col for col in required_columns if col not in feedbacks.columns]
        
        if missing_columns:
            st.error(f"❗ Les colonnes suivantes sont manquantes dans le fichier CSV : {missing_columns}")
            return pd.DataFrame(columns=CSV_HEADERS)
        return feedbacks
    return pd.DataFrame(columns=CSV_HEADERS)

def display_feedback_analysis():
    """
    Affiche les graphiques en temps réel des feedbacks sous forme de dashboard.
    """
    st.markdown("### 📊 Analyse des feedbacks")
    
    # Charger les feedbacks
    feedbacks = load_feedbacks()
    
    if feedbacks.empty:
        st.warning("⚠️ Aucun feedback disponible pour le moment.")
        return
    
    try:
        import matplotlib.pyplot as plt

        # Créer un dashboard avec plusieurs colonnes
        col1, col2 = st.columns(2)  # Deux colonnes côte à côte

        # Graphique 1 : Répartition des types de feedback (camembert)
        with col1:
            st.markdown("#### 🔄 Répartition des types de feedback")
            feedback_counts = feedbacks["nature_feedback"].value_counts()
            fig1, ax1 = plt.subplots()  # Nouvelle figure pour le graphique 1
            ax1.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal') 
            st.pyplot(fig1)

        # Graphique 2 : Distribution des notes (barres)
        with col2:
            st.markdown("#### ⭐ Distribution des notes")
            fig2, ax2 = plt.subplots()  # Nouvelle figure pour le graphique 2
            feedbacks["nombre_etoiles"].value_counts().sort_index().plot(kind="bar", color="skyblue", ax=ax2)
            ax2.set_xlabel("Nombre d'étoiles")
            ax2.set_ylabel("Nombre de feedbacks")
            st.pyplot(fig2)  # Afficher le graphique

        # Icône : Nombre total de réponses
        st.markdown("#### 📈 Nombre total de réponses")
        total_reponses = len(feedbacks)  # Calcul du nombre total de réponses
        st.metric(label="💬 Total des réponses", value=total_reponses)

    except ImportError:
        st.error("⚠️ La bibliothèque `matplotlib` n'est pas installée. Veuillez l'installer pour afficher les graphiques.")
        st.code("pip install matplotlib")

    # Afficher les commentaires récents
    st.markdown("#### 📝 Commentaires récents")
    st.dataframe(feedbacks[["timestamp", "nature_feedback", "feedback_text", "nombre_etoiles"]].tail(10))

# Exemple d'utilisation
if __name__ == "__main__":
    display_feedback_analysis()
