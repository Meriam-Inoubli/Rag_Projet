import pandas as pd
import asyncio
from lib.chain import get_chain  
from lib.embeddings import (
    create_cloud_sql_database_connection, 
    get_embedding_model, 
    get_vector_store)

# Exemples de questions et contextes spécifiques au cancer du sein
questions = [
    "Qu'est-ce que le cancer du sein ?",
    "Quels sont les symptômes du cancer du sein ?",
    "Comment diagnostique-t-on le cancer du sein ?",
    "Quels sont les facteurs de risque du cancer du sein ?",
    "Quels sont les traitements pour le cancer du sein ?",
    "Qu'est-ce que la mammographie ?",
    "Qu'est-ce qu'une biopsie mammaire ?",
    "Quels sont les types de cancer du sein ?",
    "Qu'est-ce que la chirurgie conservatrice du sein ?",
    "Quels sont les effets secondaires de la chimiothérapie pour le cancer du sein ?",
    "Qu'est-ce que l'hormonothérapie pour le cancer du sein ?",
    "Comment prévenir le cancer du sein ?",
    "Qu'est-ce que la reconstruction mammaire ?",
    "Quels sont les stades du cancer du sein ?",
    "Qu'est-ce que la radiothérapie pour le cancer du sein ?"
]

contextes = [
    "Le cancer du sein est l'un des cancers les plus fréquents chez les femmes.",
    "Les symptômes du cancer du sein peuvent varier selon le stade de la maladie.",
    "Le diagnostic précoce du cancer du sein améliore les chances de guérison.",
    "Les facteurs de risque du cancer du sein incluent des éléments génétiques et environnementaux.",
    "Les traitements du cancer du sein sont adaptés en fonction du stade et du type de cancer.",
    "La mammographie est un outil essentiel pour le dépistage du cancer du sein.",
    "Une biopsie mammaire est souvent nécessaire pour confirmer un diagnostic de cancer du sein.",
    "Il existe plusieurs types de cancer du sein, chacun ayant des caractéristiques spécifiques.",
    "La chirurgie conservatrice du sein est une option pour les patientes atteintes d'un cancer à un stade précoce.",
    "Les effets secondaires de la chimiothérapie peuvent varier selon les patientes.",
    "L'hormonothérapie est souvent utilisée pour les cancers du sein hormonodépendants.",
    "La prévention du cancer du sein passe par un dépistage régulier et un mode de vie sain.",
    "La reconstruction mammaire est une étape importante pour de nombreuses patientes après une mastectomie.",
    "Le stade du cancer du sein détermine le traitement et le pronostic.",
    "La radiothérapie est souvent utilisée après une chirurgie pour réduire le risque de récidive."
]

# Générer un fichier CSV avec des données d'évaluation
async def generate_evaluation_data(num_samples=15):
    engine = create_cloud_sql_database_connection()
    embeddings = get_embedding_model(engine)
    vector_store = await get_vector_store(engine, embeddings)

    # Initialiser la chaîne RAG
    qa_chain = await get_chain(vector_store=vector_store)

    data = {
        "user_input": [],
        "response": [],
        "reference_answer": [],
        "retrieved_contexts": []
    }

    for i in range(num_samples):
        question = questions[i]
        reponse = responses[i]
        context = contextes[i]
        # Générer une réponse avec la chaîne RAG
        response = await qa_chain.ainvoke({"query": question})
        generated_answer = response["result"]

        # Ajouter les données au dictionnaire
        data["user_input"].append(question)
        data["response"].append(generated_answer)
        data["reference_answer"].append(response)  
        data["retrieved_contexts"].append(context)

    # Créer un DataFrame et sauvegarder en CSV
    df = pd.DataFrame(data)
    df.to_csv("evaluation_data.csv", index=False)
    print("Fichier CSV généré avec succès : evaluation_data.csv")

# Exécuter la fonction asynchrone
if __name__ == "__main__":
    asyncio.run(generate_evaluation_data(num_samples=15))