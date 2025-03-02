# 🧠 Projet_GenAI : Système RAG pour l'Analyse des Données sur le Cancer du Sein
Ce projet est un système RAG (Retrieval-Augmented Generation) conçu pour répondre à des questions en combinant la recherche d'informations et la génération de texte. Il est spécialisé dans le traitement et l'analyse de données sur le cancer du sein, en utilisant des documents PDF, des embeddings, et des services cloud comme Google Cloud Vertex AI. Le système inclut également un module d'évaluation pour mesurer les performances en utilisant des métriques standard et spécifiques. 

## 🚀 Fonctionnalités

### **Système RAG**
- **Gestion des Données PDF** : Extraction, découpage en chunks, et génération d'embeddings.
- **Stockage dans Cloud SQL** : Les embeddings et métadonnées sont stockés dans une table Cloud SQL pour une récupération rapide.
- **Récupération de Contexte** : Utilisation de l'API Gemini pour récupérer des documents pertinents à partir des données stockées.
- **Génération de Réponses** : Reformulation des réponses en utilisant des modèles de langage avancés.
- **Intégration avec Google Cloud et Vertex AI** : Utilise des services cloud pour le stockage et le traitement des données. 

### **Feedback Utilisateur**
- **Soumission de Feedback** : Les utilisateurs peuvent soumettre un feedback sur chaque réponse.
- **Consultation des Feedbacks** : Les utilisateurs peuvent voir les feedbacks des autres utilisateurs sous formes des graphiques . 
- **Regénération de Réponses** : Possibilité de regénérer une réponse si la première réponse n'est pas pertinente.

### **Évaluation du Système**
- **Métriques Standard** : Calcule la similarité cosinus, la précision, le rappel et le F1-score.
- **Métriques Ragas** : Évalue la fidélité (`faithfulness`), la pertinence de la réponse (`answer_relevancy`) et la précision du contexte (`context_precision`).
- **Visualisation des Résultats** : Affiche les résultats sous forme de tableaux et de graphiques à barres.

## 📂 Structure du Projet

```plaintext
Projet_GenAI/
├── Data/                        # Dossier contenant les fichiers sources et PDF
│   ├── documents/               # Fichiers PDF pour la base de connaissances
│   └── evaluation.csv           # Fichier CSV pour l'évaluation du système
├── notebooks/                   # Notebooks pour Cloud SQL
│   └── cloud_sql_notebook.ipynb
├── src/                         # Code source du projet
│   ├── chatbot/                 # Module de chatbot
│   │   ├── lib/                 # Bibliothèques internes
│   │   │   ├── chain.py         # Logique de chaîne de traitement
│   │   │   ├── retrieve.py      # Logique de récupération des contextes
│   │   │   ├── embeddings.py    # Gestion des embeddings
│   │   │   ├── error_handler.py # Gestion des erreurs
│   │   │   ├── feedback.py      # Gestion des feedbacks utilisateur
│   │   │   ├── model.py         # Modèles de langage
│   │   │   ├── prompt.py        # Gestion des prompts
│   │   │   └── source_retriever.py # Récupération des sources
│   │   ├── app.py               # Point d'entrée de l'application Streamlit
│   │   ├── eval.py              # Module d'évaluation du système
│   │   └── generate-synthetic-data.py # Génération de données synthétiques
│   ├── data_preparation/        # Préparation des données
│   │   ├── lib/                 # Bibliothèques internes
│   │   │   ├── cloud_SQL.py     # Intégration avec Cloud SQL
│   │   │   ├── embeddings.py    # Gestion des embeddings
│   │   │   ├── metadata.py      # Gestion des métadonnées
│   │   │   └── transformer.py   # Transformation des données
│   │   └── data_init.py         # Initialisation des données
├── test/                        # Tests unitaires
│   └── test_get_llm.py           # Test pour la récupération des LLm
├── requirements.txt             # Liste des dépendances
├── README.md                    # Documentation du projet
└── .env                         # Fichier de configuration des variables d'environnement
```

## 📦 Installation

### **Prérequis**
- Python 3.9 ou supérieur 
- Un compte Google Cloud avec Vertex AI activé
- Une clé API OpenAI (Pour Ragas) 

### **Étapes d'Installation**
1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/Meriam-Inoubli/Rag_Projet.git
   cd Projet_GenAI
   ```

2. **Installer les dépendances :**
   Utilisez `pip` pour installer les dépendances listées dans `requirements.txt` :
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer les variables d'environnement :**
   Créez un fichier `.env` à la racine du projet et ajoutez vos données 
   

4. **Configurer Google Cloud :**
   - Assurez-vous que Vertex AI est activé sur votre projet Google Cloud.
   - Configurez l'authentification avec `gcloud auth application-default login`.

5. **Lancer l'application :**
   ```bash
   streamlit run src/chatbot/app.py
   ```

## 🛠 Utilisation

### **Utilisation du Système RAG**
1. **Charger des Documents :**
   - Ajoutez des documents PDF ou texte à la base de connaissances (`Data/documents/`).
   - Le système indexera automatiquement les documents pour la recherche.

2. **Poser des Questions :**
   - Entrez une question dans l'interface utilisateur.
   - Le système RAG récupérera les contextes pertinents et générera une réponse.

3. **Feedback et Regénération :**
   - Les utilisateurs peuvent soumettre un feedback sur chaque réponse.
   - Les utilisateurs peuvent voir les feedbacks des autres utilisateurs.
   - Possibilité de regénérer une réponse si la première réponse n'est pas pertinente.

### **Évaluation du Système**
1. **Charger les Données d'Évaluation :**
   - Téléchargez un fichier CSV contenant les colonnes suivantes :
     - `user_input` : La question posée par l'utilisateur.
     - `response` : La réponse générée par le système RAG.
     - `reference` : La réponse de référence (réponse attendue).
     - `retrieved_contexts` : Les contextes récupérés par le système RAG (sous forme de liste).

2. **Évaluer les Performances :**
   - Sélectionnez le nombre de questions à évaluer.
   - Visualisez les métriques standard et Ragas sous forme de tableaux et de graphiques.

3. **Analyser les Résultats :**
   - Identifiez les points forts et les points faibles du système RAG.
   - Améliorez le système en fonction des résultats.

## 📊 Métriques d'Évaluation
- **Similarité Cosinus** : Mesure la similarité entre la réponse générée et la réponse de référence.
- **Précision, Rappel, F1-Score** : Évalue la qualité de la réponse générée par rapport à la référence.
- **Fidélité (Faithfulness)** : Mesure si la réponse est fidèle au contexte récupéré.
- **Pertinence de la Réponse (Answer Relevancy)** : Mesure si la réponse est pertinente par rapport à la question.
- **Précision du Contexte (Context Precision)** : Mesure la précision du contexte récupéré par rapport à la réponse de référence.

## 🤝 Contribution
Les contributions sont les bienvenues !

## 📜 Licence
Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

