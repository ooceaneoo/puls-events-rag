import pandas as pd

from langchain_core.documents import Document

from src.vectorstore.build_faiss_index import (
    load_events_dataset,
    create_documents,
    split_documents
)


# ============================================================
# TEST DU CHARGEMENT DU DATASET
# ============================================================

def test_load_events_dataset_returns_dataframe():
    """
    Vérifie que le dataset nettoyé est correctement chargé.

    Contrôles effectués :
    - le résultat est bien un DataFrame pandas
    - le dataset contient au moins une ligne
    - la colonne utilisée pour les embeddings existe
    """

    df = load_events_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "text_for_embedding" in df.columns


# ============================================================
# TEST DE CRÉATION DES DOCUMENTS LANGCHAIN
# ============================================================

def test_create_documents_returns_langchain_documents():
    """
    Vérifie que les lignes du DataFrame sont correctement transformées en documents LangChain.

    Contrôles effectués :
    - la fonction retourne une liste
    - un document est créé
    - chaque élément est bien un objet Document
    - les métadonnées sont correctement conservées
    """

    # Création d'un faux dataset minimal pour le test
    fake_df = pd.DataFrame([
        {
            "title": "Concert test",
            "city": "Montpellier",
            "address": "Place de la Comédie",
            "start_date": "2025-10-01",
            "end_date": "2025-10-02",
            "keywords": "musique",
            "text_for_embedding": "Concert de musique à Montpellier"
        }
    ])

    documents = create_documents(fake_df)

    # Vérification du type retourné
    assert isinstance(documents, list)

    # Vérification du nombre de documents créés
    assert len(documents) == 1

    # Vérification du type LangChain
    assert isinstance(documents[0], Document)

    # Vérification des métadonnées
    assert documents[0].metadata["title"] == "Concert test"


# ============================================================
# TEST DU DÉCOUPAGE EN CHUNKS
# ============================================================

def test_split_documents_returns_chunks():
    """
    Vérifie que les documents longs sont correctement découpés en plusieurs chunks.

    Contrôles effectués :
    - la fonction retourne une liste
    - plusieurs chunks sont générés
    - chaque chunk reste un Document LangChain
    """

    # Création d'un document artificiellement long
    documents = [
        Document(
            page_content="A" * 2000,
            metadata={"title": "Document long"}
        )
    ]

    chunks = split_documents(documents)

    # Vérification du type retourné
    assert isinstance(chunks, list)

    # Vérification qu'il y a bien plusieurs chunks
    assert len(chunks) > 1

    # Vérification du type des chunks générés
    for chunk in chunks:
        assert isinstance(chunk, Document)