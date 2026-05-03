import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

DATA_PATH = Path("data/processed/events_montpellier.csv")
INDEX_PATH = Path("data/vectorstore/faiss_events_montpellier")


# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================

load_dotenv()


# ============================================================
# CHARGEMENT DU DATASET NETTOYÉ
# ============================================================

def load_events_dataset(data_path=DATA_PATH):
    """
    Chargement du fichier CSV généré lors de l'étape de pré-processing.

    Paramètre :
        data_path (Path) : chemin du fichier CSV nettoyé

    Retour :
        pandas.DataFrame : dataset des événements
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    df = pd.read_csv(data_path)

    if "text_for_embedding" not in df.columns:
        raise ValueError("La colonne text_for_embedding est absente du dataset.")

    df = df.dropna(subset=["text_for_embedding"])

    return df


# ============================================================
# CRÉATION DES DOCUMENTS LANGCHAIN
# ============================================================

def create_documents(df):
    """
    Conversion des lignes du dataset en documents LangChain.

    Chaque événement devient un document contenant :
    - le texte à vectoriser
    - les métadonnées utiles à restituer dans les réponses
    """

    documents = []

    for index, row in df.iterrows():
        metadata = {
            "event_id": int(index),
            "title": row.get("title", ""),
            "city": row.get("city", ""),
            "address": row.get("address", ""),
            "start_date": row.get("start_date", ""),
            "end_date": row.get("end_date", ""),
            "keywords": row.get("keywords", "")
        }

        document = Document(
            page_content=row["text_for_embedding"],
            metadata=metadata
        )

        documents.append(document)

    return documents


# ============================================================
# DÉCOUPAGE DES TEXTES EN CHUNKS
# ============================================================

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Découpage des documents en chunks avant vectorisation.

    Ce découpage améliore la recherche sémantique lorsque les textes
    sont longs, tout en conservant les métadonnées associées.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# CONSTRUCTION DE L'INDEX FAISS
# ============================================================

def build_faiss_index(chunks, index_path=INDEX_PATH):
    """
    Vectorisation des chunks avec Mistral puis indexation dans FAISS.

    Résultat :
        création d'un index vectoriel local réutilisable.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))

    return vectorstore


# ============================================================
# TEST DE RECHERCHE SÉMANTIQUE
# ============================================================

def test_similarity_search(vectorstore, query="Quels événements culturels sont disponibles à Montpellier ?", k=3):
    """
    Test rapide de recherche dans l'index FAISS.

    Paramètres :
        vectorstore : index FAISS chargé en mémoire
        query (str) : question de test
        k (int) : nombre de résultats retournés
    """

    results = vectorstore.similarity_search(query, k=k)

    print("\nRésultats de recherche :")

    for i, doc in enumerate(results, start=1):
        print(f"\nRésultat {i}")
        print(f"Titre : {doc.metadata.get('title')}")
        print(f"Ville : {doc.metadata.get('city')}")
        print(f"Date : {doc.metadata.get('start_date')}")
        print(f"Extrait : {doc.page_content[:300]}...")


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":

    df = load_events_dataset()
    print(f"{len(df)} événements chargés depuis le dataset nettoyé")

    documents = create_documents(df)
    print(f"{len(documents)} documents LangChain créés")

    chunks = split_documents(documents)
    print(f"{len(chunks)} chunks générés pour la vectorisation")

    vectorstore = build_faiss_index(chunks)
    print(f"Index FAISS sauvegardé dans : {INDEX_PATH}")

    test_similarity_search(vectorstore)