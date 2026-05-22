import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from src.rag.rag_chain import ask_question
from src.vectorstore.build_faiss_index import (
    load_events_dataset,
    create_documents,
    split_documents,
    build_faiss_index
)

# Chargement des variables d'environnement
load_dotenv()

# ============================================================
# INITIALISATION DE L'API FASTAPI
# ============================================================

# Configurations (titre, descriptions, version)
app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API REST locale permettant d'interroger un système RAG "
        "basé sur LangChain, FAISS et Mistral."
    ),
    version="1.0.0"
)


# ============================================================
# SCHÉMAS DE REQUÊTE ET DE RÉPONSE
# ============================================================

class AskRequest(BaseModel):
    """
    Requête reçue par l'endpoint /ask. Corps attendu pour interroger le système RAG.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question posée par l'utilisateur."
    )


class Source(BaseModel):
    """
    Structure des sources utilisées par le système RAG pour générer la réponse.
    """

    title: str | None = None
    city: str | None = None
    address: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class AskResponse(BaseModel):
    """
    Réponse retournée par le système RAG.
    """

    question: str
    answer: str
    sources: list[Source]


class RebuildResponse(BaseModel):
    """
    Réponse retournée après reconstruction de l'index FAISS.
    """

    message: str
    documents_count: int
    chunks_count: int


# ============================================================
# ENDPOINT DE VÉRIFICATION DE L'API
# ============================================================

@app.get("/health")
def health_check():
    """
    Vérifie que l'API est démarrée correctement.
    """

    return {
        "status": "ok",
        "message": "API Puls-Events opérationnelle"
    }


# ============================================================
# ENDPOINT D'INTERROGATION DU RAG
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Interroge le système RAG à partir d'une question utilisateur.

    Étapes :
    - vérifie que la question n'est pas vide
    - appelle la chaîne RAG
    - retourne la réponse générée et les sources utilisées
    """

    # Suppression des espaces inutiles
    question = request.question.strip()

    # Refus des questions vides
    if not question:
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide."
        )

    try:
        # Appel du pipeline RAG principal
        result = ask_question(question)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        }

    # Gestion des cas où l'index FAISS est absent
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Index FAISS introuvable : {error}"
        )

    # Gestion des erreurs liées aux données ou à la configuration
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    # Gestion des erreurs inattendues
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse : {error}"
        )


# ============================================================
# ENDPOINT DE RECONSTRUCTION DE L'INDEX VECTORIEL
# ============================================================

@app.post("/rebuild", response_model=RebuildResponse)
def rebuild_index(x_api_key: str = Header(...)):
    """
    Reconstruit l'index FAISS à partir du dataset nettoyé.

    Cet endpoint permet de mettre à jour la base vectorielle lorsque le fichier de données a été modifié.
    - recharge les données nettoyées
    - recrée les documents LangChain
    - recalcule les embeddings
    - sauvegarde un nouvel index FAISS
    """
    
    # Vérification de la clé admin
    admin_api_key = os.getenv("ADMIN_API_KEY")

    if x_api_key != admin_api_key:
        raise HTTPException(
            status_code=403,
            detail="Clé API invalide."
        )

    try:
        # Chargement du dataset nettoyé
        df = load_events_dataset()

        # Création des documents LangChain
        documents = create_documents(df)

        # Découpage en chunks pour la vectorisation
        chunks = split_documents(documents)

        # Reconstruction de l'index FAISS
        build_faiss_index(chunks)

        return {
            "message": "Index FAISS reconstruit avec succès.",
            "documents_count": len(documents),
            "chunks_count": len(chunks)
        }
    
    # Gestion du cas où le dataset est absent ou inaccessible
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset introuvable : {error}"
        )

    # Gestion des erreurs liées aux données ou à la configuration
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    # Gestion des erreurs inattendues
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la reconstruction de l'index : {error}"
        )