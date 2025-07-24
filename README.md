# NLP_project
# Projet d'Analyse de Sentiment Avancée pour AeroPulse Corp. (NLP avec LightGBM)

##  Contexte du Projet & Objectif Métier

Ce projet a été mené pour **AeroPulse Corp.**, un acteur majeur dans l'aéronautique et les systèmes de défense, qui fait face à un volume colossal de données textuelles non structurées (rapports de maintenance, journaux de bord, retours clients, etc.). L'objectif était de développer un système d'**analyse de sentiment avancé** pour :

-   **Détection Précoce des Problèmes :** Identifier rapidement les signaux faibles d'incidents ou de défaillances dans les rapports textuels pour une maintenance prédictive et proactive.
-   **Amélioration du Support Client :** Classer et prioriser les retours clients, optimiser la gestion des demandes et identifier les tendances d'insatisfaction.
-   **Compréhension de la Voix du Client :** Extraire des insights précieux sur la perception des produits et services à partir de flux textuels.

##  Approche Technique : Apprentissage Automatique Classique pour le NLP

Nous avons mis en œuvre une solution robuste en Traitement du Langage Naturel (NLP) basée sur des techniques éprouvées de machine learning.

### **1. Acquisition & Préparation des Données Textuelles**

*   **Source :** Dataset "Twitter US Airline Sentiment" de Kaggle (`tweets.csv`), simulant des retours clients courts et informels.
*   **Nettoyage Robuste (avec SpaCy) :**
    *   Conversion en minuscules, suppression des URLs, mentions (@), hashtags (#), ponctuation et chiffres.
    *   **Tokenisation & Lemmatisation :** Réduction des mots à leur forme de base (`running` -> `run`) et suppression des mots vides de sens (`stop words`) pour normaliser le texte et réduire le bruit.
    *   Gestion des valeurs manquantes et des doublons.
*   **Variable Cible :** Simplification en classification binaire (`binary_sentiment` : `1` pour `négatif`, `0` pour `non-négatif` (positif/neutre)), priorisant la détection des problèmes.

### **2. Vectorisation du Texte avec TF-IDF**

Pour représenter les textes de manière numérique, nous avons utilisé une méthode standard et efficace :
*   **TF-IDF (Term Frequency-Inverse Document Frequency) :** Cette technique permet de quantifier l'importance d'un mot dans un document par rapport à l'ensemble du corpus. Elle convertit les textes nettoyés en vecteurs numériques, où chaque dimension correspond à un terme et sa valeur reflète sa pertinence.

### **3. Modélisation : Entraînement d'un Modèle LightGBM**

*   **Architecture :** Utilisation de **LightGBM**, un framework de boosting de gradient rapide et performant, reconnu pour son efficacité sur de larges datasets et sa précision.
*   **Apprentissage Supervisé :** Le modèle a été entraîné sur les vecteurs TF-IDF pour classifier le sentiment des tweets.
*   **Optimisation :** Les hyperparamètres de LightGBM ont été ajustés pour maximiser les performances de classification binaire.

### **4. Évaluation des Performances : Des Résultats Solides**
Le modèle a été évalué sur un ensemble de validation dédié (20% des données), simulant des données non vues.

*   **Classe d'intérêt (Sentiment Négatif) :**
    *   **Précision : 84%** (84% des tweets prédits "négatifs" sont réellement négatifs - faible fausse alarme).
    *   **Rappel : 89%** (89% des tweets réellement "négatifs" ont été détectés - peu de problèmes manqués).
    *   **F1-Score : 87%** (Excellent équilibre entre précision et rappel).
*   **Capacité de Discrimination :**
    *   **AUC-ROC : 0.90** (Excellente capacité à distinguer les sentiments négatifs des autres).
    *   **PR-AUC : 0.93** (Performance exceptionnelle pour la détection de la classe d'intérêt dans un contexte de déséquilibre).

Ces métriques démontrent la haute fiabilité de notre modèle pour détecter les problèmes.

### **5. IA Explicable (XAI) : Comprendre les Décisions du Modèle**

Pour comprendre les facteurs influençant les prédictions du modèle, nous avons exploré l'IA explicable (XAI) :
*   **Objectif :** Comprendre quels mots ou caractéristiques TF-IDF influencent le plus la prédiction du sentiment.
*   **Méthode :** Utilisation des **SHAP values** (KernelExplainer pour une vue globale et Force Plot pour une explication individuelle).
*   **Insights Clés :** Nous avons pu identifier des mots spécifiques (`delayed`, `cancelled`, `lost`, `worst`) qui poussent fortement la prédiction vers le négatif, et d'autres (`thanks`, `great`) vers le non-négatif. Ces informations sont cruciales pour AeroPulse Corp. pour comprendre la nature des problèmes signalés.

##  **Déploiement & Impact Opérationnel**

Le modèle entraîné et le vectoriseur TF-IDF ont été sauvegardés, prêts pour le déploiement.

*   **API Flask (Démonstrateur) :** Une API web simple a été créée (`app.py`), permettant d'envoyer un texte (tweet) et de recevoir une prédiction de sentiment en temps réel. Cette API peut être intégrée aux systèmes existants d'AeroPulse Corp.
*   **Bénéfices Concrets pour AeroPulse Corp. :**
    *   **Alertes Automatisées :** Détecter et prioriser les rapports critiques.
    *   **Analyse de Tendances :** Comprendre les évolutions du sentiment client sur leurs produits/services.
    *   **Optimisation des Ressources :** Acheminer les demandes de support vers les bonnes équipes plus rapidement.
    *   **Amélioration Produit/Service :** Utiliser les retours négatifs ciblés pour l'amélioration continue.

##  **Technologies Utilisées**

*   **Langage :** Python
*   **Librairies :** Pandas, NumPy, Matplotlib, Seaborn, SpaCy, **scikit-learn**, **LightGBM**, Flask, SHAP.
