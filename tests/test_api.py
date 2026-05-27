from fastapi.testclient import TestClient

from api.main import app


# Initialisation du client de test FastAPI
# Simule des requêtes HTTP sans démarrer réellement le serveur Uvicorn
client = TestClient(app)


# Test de l'endpoint /health
def test_health_check_returns_ok():
    """
    Vérifie que l'API répond correctement sur /health.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# Test de l'endpoint /ask
def test_ask_returns_rag_response():
    """
    Vérifie que l'endpoint /ask retourne :
    - une réponse HTTP valide
    - une réponse générée
    - les sources utilisées par le système RAG
    """

    payload = {
        "question": "Quels événements culturels sont disponibles à Montpellier ?"
    }

    response = client.post("/ask", json=payload)

    data = response.json()

    assert response.status_code == 200

    assert "question" in data
    assert "answer" in data
    assert "sources" in data

    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)


# Test des questions vides
def test_ask_rejects_empty_question():
    """
    Vérifie que l'API refuse les questions vides.
    """

    payload = {
        "question": "   "
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "La question ne peut pas être vide."


# Test de sécurisation de /rebuild
def test_rebuild_rejects_missing_api_key():
    """
    Vérifie que l'endpoint /rebuild refuse les requêtes sans clé API.
    """

    response = client.post("/rebuild")

    assert response.status_code == 422


def test_rebuild_rejects_invalid_api_key():
    """
    Vérifie que l'endpoint /rebuild refuse une clé API invalide.
    """

    response = client.post(
        "/rebuild",
        headers={"x-api-key": "wrong_key"}
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Clé API invalide."