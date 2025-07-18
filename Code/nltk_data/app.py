# app.py (fichier à créer localement pour le modèle TF-IDF/LightGBM)
from flask import Flask, request, jsonify
import joblib # Pour charger le modèle LightGBM et le TfidfVectorizer
import os     # Pour gérer les chemins de fichiers

# --- Configuration du Modèle et des Chemins ---
# ADAPTER CE CHEMIN pour qu'il pointe vers le dossier LOCAL où vous avez sauvegardé vos modèles TF-IDF/LightGBM
# Exemple: si app.py est dans 'mon_projet_nlp/' et les modèles dans 'mon_projet_nlp/model_tfidf_lgbm/'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_tfidf_lgbm') # Utilise le chemin relatif au fichier app.py
# Ou un chemin absolu si vous préférez:
# MODEL_DIR = 'C:/chemin/vers/votre/Projet_NLP/model_tfidf_lgbm' # <-- ADAPTEZ CE CHEMIN

LGBM_MODEL_FILENAME = os.path.join(MODEL_DIR, 'lgbm_sentiment_model.pkl')
VECTORIZER_FILENAME = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Charger le Modèle LightGBM et le TfidfVectorizer (une seule fois au démarrage de l'API)
print("Chargement du Modèle LightGBM et du TfidfVectorizer au démarrage de l'API...")
try:
    LGBM_MODEL = joblib.load(LGBM_MODEL_FILENAME)
    VECTORIZER = joblib.load(VECTORIZER_FILENAME)
    print("Modèle LightGBM et TfidfVectorizer chargés avec succès.")
except Exception as e:
    print(f"ERREUR lors du chargement du modèle/vectoriseur : {e}")
    print("Assurez-vous que les chemins sont corrects et que les fichiers .pkl existent.")
    exit() # Quitter l'application si le modèle ne peut pas être chargé


# Créer l'application Flask
app = Flask(__name__)

# --- Fonction de Prédiction ---
def predict_sentiment_for_text(text: str) -> dict:
    """
    Prédit le sentiment d'un texte donné en utilisant le modèle TF-IDF/LightGBM.
    """
    # 1. Pré-traitement du texte (similaire à ce que nous avons fait dans le notebook)
    # Dans un déploiement réel, ces fonctions seraient importées ou recodées ici.
    # Pour simplifier, nous allons recoder les étapes essentielles ici.
    import re
    # (NLTK n'est pas utilisé directement ici pour éviter les dépendances lourdes du service API)
    # Nous devrions idéalement utiliser Spacy pour la production ici si c'était le pipeline complet.
    # Pour cet exemple simple d'API, nous allons faire un nettoyage très basique pour le TF-IDF.

    # Nettoyage très basique pour l'API (sans NLTK/SpaCy complets pour la simplicité)
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^a-z\s]', '', cleaned_text) # Garde lettres et espaces
    cleaned_text = re.sub(r'http\S+|www.\S+', '', cleaned_text) # Supprime URLs
    cleaned_text = re.sub(r'@\S+|#\S+', '', cleaned_text) # Supprime mentions/hashtags
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Gère les espaces

    # 2. Vectorisation du texte avec le TfidfVectorizer chargé
    # Le vectoriseur attend une liste de chaînes.
    text_vectorized = VECTORIZER.transform([cleaned_text])

    # 3. Prédiction par le modèle LightGBM
    # predicted_proba retourne les probabilités [prob_classe0, prob_classe1]
    probabilities = LGBM_MODEL.predict_proba(text_vectorized)[0]
    
    # Déduire la classe prédite
    predicted_class_id = LGBM_MODEL.predict(text_vectorized)[0] # 0 ou 1
    
    sentiment_labels = {0: 'non-negative', 1: 'negative'}
    predicted_sentiment = sentiment_labels[predicted_class_id]

    return {
        "text_original": text,
        "predicted_sentiment": predicted_sentiment,
        "probability_negative": float(probabilities[1]), # Probabilité que ce soit 'negative'
        "probability_non_negative": float(probabilities[0]) # Probabilité que ce soit 'non-negative'
    }

# --- Route de l'API Flask ---
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_api():
    if request.method == 'POST':
        data = request.get_json() # Récupérer les données envoyées en JSON
        text = data.get('text', '') # Extraire le texte de la requête JSON

        if not text:
            return jsonify({"error": "No 'text' provided in the request."}), 400

        try:
            prediction_result = predict_sentiment_for_text(text)
            return jsonify(prediction_result)
        except Exception as e:
            # Gérer les erreurs de prédiction
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --- Exécution de l'Application Flask ---
if __name__ == '__main__':
    print("Démarrage de l'API Flask...")
    app.run(host='0.0.0.0', port=5000, debug=False)