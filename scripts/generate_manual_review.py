import sys
from pathlib import Path

import pandas as pd


# Configuration des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag.rag_chain import ask_question


INPUT_PATH = PROJECT_ROOT / "data" / "test" / "manual_questions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "test" / "manual_review_results.csv"


# Génération des réponses pour revue manuelle
def generate_manual_review():
    """
    Génère les réponses du système RAG pour toutes les questions du jeu de test manuel.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"question", "expected_answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Le fichier doit contenir les colonnes 'question' et 'expected_answer'."
        )

    rows = []

    for index, row in df.iterrows():
        question = row["question"]
        expected_answer = row["expected_answer"]

        print(f"\nQuestion {index + 1}/{len(df)}")
        print(question)

        result = ask_question(question)

        generated_answer = result["answer"]
        sources = result["sources"]

        formatted_sources = " || ".join([
            f"{source.get('title')} | {source.get('start_date')} | {source.get('address')}"
            for source in sources
        ])

        rows.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "sources": formatted_sources,
            "manual_evaluation": "",
            "manual_comments": ""
        })

    results_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nFichier de revue manuelle généré.")
    print(f"Résultats sauvegardés dans : {OUTPUT_PATH}")


# Exécution du script
if __name__ == "__main__":
    generate_manual_review()