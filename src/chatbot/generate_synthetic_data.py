import pandas as pd
import asyncio
import time
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

# Réponses de référence
reference_answers = [
    "Le cancer du sein est une tumeur maligne qui se développe à partir des cellules mammaires.",
    "Les symptômes courants incluent une masse dans le sein, des changements de la peau ou du mamelon, et des écoulements.",
    "Le diagnostic repose sur la mammographie, l'échographie, et parfois une biopsie.",
    "Les facteurs de risque incluent l'âge, les antécédents familiaux, et certaines mutations génétiques comme BRCA1 et BRCA2.",
    "Les traitements incluent la chirurgie, la radiothérapie, la chimiothérapie, et l'hormonothérapie.",
    "La mammographie est une radiographie du sein utilisée pour détecter des anomalies.",
    "Une biopsie mammaire consiste à prélever un échantillon de tissu pour analyse.",
    "Les types incluent le carcinome canalaire, le carcinome lobulaire, et le cancer triple négatif.",
    "La chirurgie conservatrice vise à retirer la tumeur tout en préservant le sein.",
    "Les effets secondaires incluent la fatigue, la perte de cheveux, et des nausées.",
    "L'hormonothérapie bloque les hormones qui stimulent la croissance des cellules cancéreuses.",
    "La prévention inclut un mode de vie sain, un dépistage régulier, et parfois des médicaments préventifs.",
    "La reconstruction mammaire est une chirurgie pour restaurer l'apparence du sein après une mastectomie.",
    "Les stades vont de 0 (non invasif) à IV (métastatique).",
    "La radiothérapie utilise des rayons pour détruire les cellules cancéreuses résiduelles."
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
        context = contextes[i]
        reference_answer = reference_answers[i]

        # Générer une réponse avec la chaîne RAG
        response = await qa_chain.ainvoke({"query": question})
        generated_answer = response["result"]

        # Ajouter les données au dictionnaire
        data["user_input"].append(question)
        data["response"].append(generated_answer)
        time.sleep(10)
        data["reference_answer"].append(reference_answer)
        data["retrieved_contexts"].append(context)

    # Créer un DataFrame et sauvegarder en CSV
    df = pd.DataFrame(data)
    df.to_csv("evaluation_data.csv", index=False)
    print("Fichier CSV généré avec succès : evaluation_data.csv")

# Exécuter la fonction asynchrone
if __name__ == "__main__":
    asyncio.run(generate_evaluation_data(num_samples=15))
