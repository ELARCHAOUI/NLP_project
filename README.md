# NLP_project
# Projet d'Analyse de Sentiment Avancée pour AeroPulse Corp. (NLP avec BERT)

## 🎯 Contexte du Projet & Objectif Métier

Ce projet a été mené pour **AeroPulse Corp.**, un acteur majeur dans l'aéronautique et les systèmes de défense, qui fait face à un volume colossal de données textuelles non structurées (rapports de maintenance, journaux de bord, retours clients, etc.). L'objectif était de développer un système d'**analyse de sentiment avancé** pour :

-   **Détection Précoce des Problèmes :** Identifier rapidement les signaux faibles d'incidents ou de défaillances dans les rapports textuels pour une maintenance prédictive et proactive.
-   **Amélioration du Support Client :** Classer et prioriser les retours clients, optimiser la gestion des demandes et identifier les tendances d'insatisfaction.
-   **Compréhension de la Voix du Client :** Extraire des insights précieux sur la perception des produits et services à partir de flux textuels.

## 🚀 Approche Technique : Modèle de Transformeur Pré-entraîné (BERT)

Nous avons mis en œuvre une solution de pointe en Traitement du Langage Naturel (NLP) basée sur les **Transformeurs Pré-entraînés**, qui sont à la base des modèles de langage les plus performants actuels.

### **1. Acquisition & Préparation des Données Textuelles**

*   **Source :** Dataset "Twitter US Airline Sentiment" de Kaggle (`tweets.csv`), simulant des retours clients courts et informels.
*   **Nettoyage Robuste (avec SpaCy) :**
    *   Conversion en minuscules, suppression des URLs, mentions (@), hashtags (#), ponctuation et chiffres.
    *   **Tokenisation & Lemmatisation :** Réduction des mots à leur forme de base (`running` -> `run`) et suppression des mots vides de sens (`stop words`) pour normaliser le texte et réduire le bruit.
    *   Gestion des valeurs manquantes et des doublons.
*   **Variable Cible :** Simplification en classification binaire (`binary_sentiment` : `1` pour `négatif`, `0` pour `non-négatif` (positif/neutre)), priorisant la détection des problèmes.

### **2. Vectorisation du Texte avec le Tokenizer de BERT**

Contrairement aux méthodes traditionnelles comme TF-IDF, les Transformeurs utilisent leur propre mécanisme de représentation du langage :
*   **Tokenizer de BERT :** Le texte nettoyé est converti en séquences d'identifiants numériques (`input_ids`) et de masques d'attention (`attention_mask`), préparés spécifiquement pour l'architecture de BERT. Ce processus gère également le padding et la troncation des séquences.

### **3. Modélisation : Fine-tuning d'un Modèle BERT**

*   **Architecture :** Utilisation de `BertForSequenceClassification` (basé sur `bert-base-uncased`), un modèle de Transformeur pré-entraîné de Hugging Face.
*   **Apprentissage par Transfert :** Le modèle, déjà entraîné sur des quantités massives de texte pour comprendre la langue, a été **ajusté finement (fine-tuned)** sur notre dataset de tweets pour la tâche spécifique d'analyse de sentiment.
*   **Optimisation :** Entraînement sur GPU (Google Colab), utilisant `AdamW` comme optimiseur et un faible taux d'apprentissage.

### **4. Évaluation des Performances : Des Résultats de Pointe**
Le modèle a été évalué sur un ensemble de validation dédié (20% des données), simulant des données non vues.

*   **Classe d'intérêt (Sentiment Négatif) :**
    *   **Précision : 84%** (84% des tweets prédits "négatifs" sont réellement négatifs - faible fausse alarme).
    *   **Rappel : 89%** (89% des tweets réellement "négatifs" ont été détectés - peu de problèmes manqués).
    *   **F1-Score : 87%** (Excellent équilibre entre précision et rappel).
*   **Capacité de Discrimination :**
    *   **AUC-ROC : 0.90** (Excellente capacité à distinguer les sentiments négatifs des autres).
    *   **PR-AUC : 0.93** (Performance exceptionnelle pour la détection de la classe d'intérêt dans un contexte de déséquilibre).

Ces métriques placent notre modèle parmi les solutions de pointe, capable de détecter les problèmes avec une grande fiabilité.

### **5. IA Explicable (XAI) : Comprendre les Décisions du Modèle**

Pour démystifier la "boîte noire" de BERT, nous avons exploré l'IA explicable (XAI) :
*   **Objectif :** Comprendre quels mots influencent le plus la prédiction du sentiment.
*   **Méthode :** Utilisation des **SHAP values** (KernelExplainer pour une vue globale et Force Plot pour une explication individuelle). Bien que gourmande en ressources, cette approche fournit des insights précieux.
*   **Insights Clés :** Nous avons pu identifier des mots spécifiques (`delayed`, `cancelled`, `lost`, `worst`) qui poussent fortement la prédiction vers le négatif, et d'autres (`thanks`, `great`) vers le non-négatif. Ces informations sont cruciales pour AeroPulse Corp. pour comprendre la nature des problèmes signalés.

## 🚀 **Déploiement & Impact Opérationnel**

Le modèle entraîné et son tokenizer ont été sauvegardés, prêts pour le déploiement.

*   **API Flask (Démonstrateur) :** Une API web simple a été créée (`app.py`), permettant d'envoyer un texte (tweet) et de recevoir une prédiction de sentiment en temps réel. Cette API peut être intégrée aux systèmes existants d'AeroPulse Corp.
*   **Bénéfices Concrets pour AeroPulse Corp. :**
    *   **Alertes Automatisées :** Détecter et prioriser les rapports critiques.
    *   **Analyse de Tendances :** Comprendre les évolutions du sentiment client sur leurs produits/services.
    *   **Optimisation des Ressources :** Acheminer les demandes de support vers les bonnes équipes plus rapidement.
    *   **Amélioration Produit/Service :** Utiliser les retours négatifs ciblés pour l'amélioration continue.

## 🛠️ **Technologies Utilisées**

*   **Langage :** Python
*   **Librairies :** Pandas, NumPy, Matplotlib, Seaborn, SpaCy, Hugging Face Transformers, PyTorch, scikit-learn, Flask, SHAP.

---
