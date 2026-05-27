import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig


# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag.rag_chain import ask_question, load_vectorstore


INPUT_PATH = PROJECT_ROOT / "data" / "test" / "ragas_questions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "test" / "ragas_evaluation_results.csv"

RETRIEVAL_K = 3


# Chargement des variables d'environnement
load_dotenv()


# Modèles utilisés pour l'évaluation Ragas
def get_ragas_llm():
    """
    Initialise le modèle Mistral utilisé par Ragas pour évaluer
    automatiquement les réponses générées par le système RAG.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0
    )

def get_ragas_embeddings():
    """
    Initialise le modèle d'embeddings Mistral utilisé par Ragas.

    Les embeddings servent notamment à évaluer la pertinence
    des contextes récupérés.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("La variable MISTRAL_API_KEY est absente du fichier .env.")

    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=api_key
    )


# Récupération des contextes FAISS
def retrieve_contexts(question, k=RETRIEVAL_K):
    """
    Récupère les documents les plus pertinents dans FAISS.

    Ces contextes sont fournis à Ragas pour évaluer :
    - la fidélité de la réponse au contexte
    - la précision du contexte
    - le rappel du contexte
    """

    vectorstore = load_vectorstore()
    documents = vectorstore.similarity_search(question, k=k)

    return [doc.page_content for doc in documents]


# Construction du dataset Ragas
def build_ragas_dataset():
    """
    Construit le dataset attendu par Ragas.

    Colonnes utilisées :
    - question : question utilisateur
    - answer : réponse générée par le pipeline RAG
    - contexts : documents récupérés dans FAISS
    - ground_truth : réponse attendue annotée
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"question", "expected_answer"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Le fichier ragas_questions.csv doit contenir les colonnes "
            "'question' et 'expected_answer'."
        )

    rows = []

    for index, row in df.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]

        print(f"\nQuestion {index + 1}/{len(df)}")
        print(f"Question : {question}")

        # Génération de la réponse par le pipeline RAG réel
        rag_result = ask_question(question)
        generated_answer = rag_result["answer"]

        # Récupération des contextes utilisés pour l'évaluation
        contexts = retrieve_contexts(question)

        rows.append({
            "question": question,
            "answer": generated_answer,
            "contexts": contexts,
            "ground_truth": expected_answer
        })

    return Dataset.from_list(rows)


# Évaluation automatique avec Ragas
def evaluate_rag_with_ragas():
    """
    Lance l'évaluation automatique du système RAG avec Ragas.

    Métriques retenues :
    - faithfulness : vérifie si la réponse reste fidèle au contexte
    - context_precision : vérifie si les contextes récupérés sont pertinents
    - context_recall : vérifie si les contextes couvrent les informations attendues

    Les métriques utilisées ici sont les métriques historiques de Ragas,
    compatibles avec LangChain et le modèle Mistral utilisé dans le projet.
    """

    dataset = build_ragas_dataset()

    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=2
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config
    )

    results_df = result.to_pandas()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nÉvaluation Ragas terminée.")
    print(f"Résultats sauvegardés dans : {OUTPUT_PATH}")

    print("\nScores moyens :")
    print(result)


# Exécution du script
if __name__ == "__main__":
    evaluate_rag_with_ragas()