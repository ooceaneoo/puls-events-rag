from src.rag.rag_chain import (
    load_vectorstore,
    format_documents,
    ask_question
)


# Test du chargement de l'index FAISS
def test_load_vectorstore_returns_faiss_instance():
    """
    Vérifie que l'index vectoriel FAISS sauvegardé localement peut être correctement rechargé.

    Contrôles effectués :
    - le chargement ne provoque pas d'erreur
    - un objet vectorstore est bien retourné
    """

    vectorstore = load_vectorstore()

    assert vectorstore is not None


# Test du formatage des documents
def test_format_documents_returns_string():
    """
    Vérifie que les documents récupérés depuis FAISS
    sont correctement transformés en contexte textuel
    exploitable par le modèle de langage.

    Contrôles effectués :
    - les résultats FAISS sont bien récupérés
    - le contexte formaté est une chaîne de caractères
    - le contexte n'est pas vide
    """

    # Chargement de l'index FAISS
    vectorstore = load_vectorstore()

    # Recherche sémantique de documents proches de la requête
    docs = vectorstore.similarity_search(
        "événements culturels à Montpellier",
        k=2
    )

    # Construction du contexte textuel
    formatted = format_documents(docs)

    # Vérification du type retourné
    assert isinstance(formatted, str)

    # Vérification que le contexte contient du texte
    assert len(formatted) > 0


# Test du pipeline RAG complet
def test_ask_question_returns_response():
    """
    Vérifie le bon fonctionnement du pipeline RAG complet.

    Le test valide les étapes suivantes :
    - recherche des documents pertinents dans FAISS
    - génération d'une réponse via Mistral
    - retour des sources utilisées

    Contrôles effectués :
    - la fonction retourne un dictionnaire
    - les clés attendues sont présentes
    - la réponse générée n'est pas vide
    - les sources sont bien retournées sous forme de liste
    """

    # Question de test simulant une requête utilisateur
    result = ask_question(
        "Quels événements culturels peux-tu recommander à Montpellier ?"
    )

    # Vérification du type global retourné
    assert isinstance(result, dict)

    # Vérification de la structure de réponse
    assert "question" in result
    assert "answer" in result
    assert "sources" in result

    # Vérification du contenu de la réponse générée
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0

    # Vérification des sources retournées
    assert isinstance(result["sources"], list)