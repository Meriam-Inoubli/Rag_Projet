# Model Card: RAG Medical - Cancer du Sein

## Informations générales
- **Nom du modèle** : RAG Medical - Cancer du Sein  
- **Version** : v1.0  
- **Date de création** : 28 fevrier 2025  
- **Auteurs** : Meriam Inoubli 
- **Contact** : meriam.inoubli@dauphine.tn 

---

## Description du modèle
Le modèle **RAG Medical** est conçu pour fournir des informations médicales précises et fiables sur le cancer du sein en français. Il utilise une architecture RAG (Retrieval-Augmented Generation) combinant un retriever basé sur PostgresVectorStore et un modèle de langage Gemini-1.5-pro.

### Domaines d'application
- Oncologie  
- Cancer du sein  

### Public cible
- Professionnels de santé  
- Patients  
- Chercheurs  

---

## Données d'entraînement et de test
### Sources des données
- Documents médicaux  
- Articles scientifiques  
- Guides de pratique clinique  

### Pré-traitement des données
- Découpage en chunks de 1000 caractères avec un chevauchement de 200 caractères.  
- Filtrage des contenus non pertinents (ex : tables des matières).  

### Taille des données
- 4 documents  
- 29 000 chunks  

### Biais potentiels
- Sur-représentation de certaines sources.  
  

---

## Métriques de performance
### Métriques utilisées
- **Similarité cosinus** : Mesure la similarité entre la réponse générée et la réponse de référence.  
- **F1-score** : Évalue la précision et le rappel des réponses.  
- **Faithfulness** : Mesure à quel point la réponse est fidèle aux contextes récupérés.  
- **Answer Relevancy** : Évalue la pertinence de la réponse par rapport à la question.  
- **Context Precision** : Mesure la précision des contextes récupérés.  

### Résultats
| Métrique               | Valeur | Description                                   |
|------------------------|--------|-----------------------------------------------|
| Similarité cosinus      | 0.75   | Similarité moyenne entre les réponses.        |
| F1-score               | 0.55   | Score F1 moyen sur l'ensemble de test.        |
| Faithfulness           | 0.0   | Fidélité des réponses aux contextes.          |
| Answer Relevancy       | 0.0   | Pertinence des réponses par rapport aux questions. |
| Context Precision      | 0.0   | Précision des contextes récupérés.            |

---

## Utilisation du modèle
### Cas d'utilisation
- Répondre à des questions sur les symptômes, les traitements, et le diagnostic du cancer du sein.  

### Instructions d'utilisation
- Poser des questions en français.  
- Éviter les questions hors contexte.  

### Exemples de questions/réponses
- **Question** : Quels sont les symptômes du cancer du sein ?  
  **Réponse** : Les symptômes incluent une masse dans le sein, des changements de la peau, et des écoulements du mamelon.  

- **Question** : Comment diagnostique-t-on le cancer du sein ?  
  **Réponse** : Le diagnostic repose sur la mammographie, l'échographie, et parfois une biopsie.  

---

## Éthique et responsabilité
### Considérations éthiques
- Ce modèle ne doit pas remplacer un avis médical professionnel.  

### Limites de responsabilité
- Ne pas utiliser pour des décisions médicales critiques.  

### Transparence
- Les contextes récupérés sont affichés pour chaque réponse.  

---

## Maintenance et mise à jour
### Fréquence des mises à jour
- Mises à jour trimestrielles.  

### Processus de mise à jour
- Ajout de nouvelles données.  
- Réévaluation des performances.  
  

---

## Licence
- **Licence** : MIT  
