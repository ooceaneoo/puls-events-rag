import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

# Dossier contenant les fichiers FAISS générés à l'étape précédente :
# - index.faiss : vecteurs
# - index.pkl : documents et métadonnées
INDEX_PATH = Path("data/vectorstore/faiss_events_montpellier")


# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================

# Chargement du fichier .env afin de récupérer la clé API Mistral.
load_dotenv()


# ============================================================
# CHARGEMENT DU MODÈLE D'EMBEDDINGS
# ============================================================

def get_embeddings_model():
    """
    Initialisation du modèle d'embeddings Mistral.

    Le même modèle est utilisé pour :
    - construire l'index FAISS
    - vectoriser les questions utilisateur lors de la recherche
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    # Sécurité : arrêt explicite si la clé API est absente.
    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )


# ============================================================
# CHARGEMENT DU MODÈLE DE GÉNÉRATION
# ============================================================

def get_llm():
    """
    Initialisation du LLM Mistral.

    Ce modèle génère la réponse finale à partir :
    - de la question utilisateur
    - du contexte récupéré dans FAISS
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.2  # limitation des réponses inventées
    )


# ============================================================
# CHARGEMENT DE L'INDEX FAISS
# ============================================================

def load_vectorstore(index_path=INDEX_PATH):
    """
    Chargement de la base vectorielle FAISS sauvegardée localement.

    LangChain recharge à la fois :
    - l'index vectoriel
    - les documents associés 
    - les métadonnées des événements
    """

    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")

    embeddings = get_embeddings_model()

    vectorstore = FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# FORMATAGE DU CONTEXTE
# ============================================================

def format_documents(documents):
    """
    Préparation des documents FAISS avant envoi au LLM.

    Les documents récupérés sont transformés en contexte lisible.
    Les métadonnées sont ajoutées pour permettre au chatbot de citer les titres, dates et lieux dans sa réponse.
    """

    formatted_docs = []

    for doc in documents:
        metadata = doc.metadata

        # Création d'un bloc texte clair pour chaque événement retrouvé.
        formatted_doc = (
            f"Titre : {metadata.get('title', 'Non renseigné')}\n"
            f"Ville : {metadata.get('city', 'Non renseignée')}\n"
            f"Adresse : {metadata.get('address', 'Non renseignée')}\n"
            f"Date de début : {metadata.get('start_date', 'Non renseignée')}\n"
            f"Date de fin : {metadata.get('end_date', 'Non renseignée')}\n"
            f"Contenu : {doc.page_content}"
        )

        formatted_docs.append(formatted_doc)

    # Séparation explicite des événements pour améliorer la lisibilité du contexte.
    return "\n\n---\n\n".join(formatted_docs)


# ============================================================
# PROMPT DU SYSTÈME RAG
# ============================================================

def get_prompt():
    """
    Création du prompt utilisé par la chaîne RAG.

    Le prompt encadre le comportement du chatbot :
    - réponse basée sur les sources
    - pas d'invention 
    - formulation claire pour l'utilisateur final
    """

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
Tu es l'assistant intelligent de Puls-Events.
Ton rôle est d'aider les utilisateurs à trouver des événements culturels à Montpellier.

Règles :
- Réponds uniquement à partir du contexte fourni.
- Si le contexte ne permet pas de répondre, indique que l'information n'est pas disponible.
- Propose des recommandations claires et utiles.
- Mentionne les titres, dates et lieux lorsque ces informations sont disponibles.
- Réponds dans la langue de l'utilisateur.
"""
        ),
        (
            "human",
            """
Question utilisateur :
{question}

Événements retrouvés dans la base :
{context}

Réponse :
"""
        )
    ])


# ============================================================
# QUESTION-RÉPONSE RAG
# ============================================================

def ask_question(question, k=4):
    """
    Exécution du pipeline RAG complet.

    Pipeline :
    1. Chargement de FAISS
    2. Recherche sémantique des événements pertinents
    3. Construction du contexte
    4. Génération de la réponse par Mistral
    5. Retour de la réponse avec les sources
    """

    vectorstore = load_vectorstore()

    # Création du retriever LangChain
    # k correspond au nombre de chunks retournés par FAISS
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Recherche des événements/chunks les plus proches de la question
    documents = retriever.invoke(question)

    # Transformation des documents retrouvés en contexte textuel
    context = format_documents(documents)

    llm = get_llm()
    prompt = get_prompt()

    # Chaîne LangChain : prompt -> modèle Mistral -> sortie texte simple
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question": question,
        "context": context
    })

    sources = []

    # Conservation des sources pour transparence et API
    for doc in documents:
        sources.append({
            "title": doc.metadata.get("title"),
            "city": doc.metadata.get("city"),
            "address": doc.metadata.get("address"),
            "start_date": doc.metadata.get("start_date"),
            "end_date": doc.metadata.get("end_date")
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":

    # Question de démonstration pour valider le fonctionnement du RAG en local
    question = "Quels événements culturels peux-tu me recommander à Montpellier ?"

    result = ask_question(question)

    print("\nQuestion :")
    print(result["question"])

    print("\nRéponse générée :")
    print(result["answer"])

    print("\nSources utilisées :")
    for source in result["sources"]:
        print(f"- {source['title']} | {source['start_date']} | {source['address']}")