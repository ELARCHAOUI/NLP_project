# test_api.py (ou une nouvelle cellule de votre notebook)
import requests
import json

# L'URL de votre API Flask locale
API_URL = "http://127.0.0.1:5000/predict_sentiment" # Ou "http://localhost:5000/predict_sentiment"

# Données à envoyer (des tweets de test)
test_tweet_1 = "I am so disappointed with my flight delay, this is unacceptable!"
test_tweet_2 = "Great flight, on time and amazing service!"
test_tweet_3 = "My flight was okay, nothing special."
test_tweet_4 = "Worst customer service ever, my bag is lost!"




headers = {'Content-Type': 'application/json'}

def send_test_request(tweet):
    data = {'text': tweet}
    print(f"\n--- Envoi de la requête pour : \"{tweet}\" ---")
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)

        result = response.json()
        print("\nRéponse de l'API :")
        print(json.dumps(result, indent=2)) # Affiche le JSON formaté

    except requests.exceptions.ConnectionError:
        print("\nERREUR : La connexion à l'API a échoué.")
        print("Assurez-vous que votre 'app.py' est bien en cours d'exécution dans le terminal de VS Code.")
    except requests.exceptions.RequestException as e:
        print(f"\nErreur lors de l'appel de l'API : {e}")

send_test_request(test_tweet_1)
send_test_request(test_tweet_2)
send_test_request(test_tweet_3)
send_test_request(test_tweet_4)



print("\n--- Tests de l'API terminés ---")