import re
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path


# ============================================================
# CONFIGURATION DE L'API OPENAGENDA (OpenDataSoft)
# ============================================================

BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"


# ============================================================
# NETTOYAGE DES DONNÉES TEXTUELLES
# ============================================================

def clean_text(text):
    """
    Nettoyage des champs textuels issus de l'API.

    Objectifs :
    - suppression des balises HTML
    - normalisation des espaces
    - suppression des retours à la ligne

    Paramètre :
        text (str) : texte brut issu de l'API

    Retour :
        str : texte nettoyé
    """

    if not text:
        return ""

    text = re.sub(r"<.*?>", "", str(text))  # suppression HTML
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# EXTRACTION DES DONNÉES DEPUIS L'API
# ============================================================

def fetch_events(city="Montpellier", limit=100):
    """
    Extraction des événements OpenAgenda via API REST.

    Fonctionnalités :
    - filtrage par ville
    - filtrage temporel (événements sur les 12 derniers mois + à venir)
    - gestion de la pagination pour récupérer l'intégralité des données

    Paramètres :
        city (str) : ville ciblée
        limit (int) : nombre de résultats par appel API

    Retour :
        list : liste brute des événements
    """

    today = date.today()
    one_year_ago = today - timedelta(days=365)

    all_events = []
    offset = 0

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "where": (
                f'location_city="{city}" '
                f'AND firstdate_begin >= "{one_year_ago}"'
            )
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(response.url)
            print(response.text)
            response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        # Arrêt lorsque l'API ne renvoie plus de données
        if not results:
            break

        all_events.extend(results)
        offset += limit

    return all_events


# ============================================================
# TRANSFORMATION ET STRUCTURATION DES DONNÉES
# ============================================================

def clean_events(events):
    """
    Transformation des données brutes en dataset structuré.

    Sélection des champs pertinents pour le système RAG :
    - titre
    - description
    - localisation
    - dates
    - mots-clés

    Construction d'un champ textuel dédié aux embeddings (indexation future).

    Paramètre :
        events (list) : données brutes issues de l'API

    Retour :
        pandas.DataFrame : dataset nettoyé
    """

    cleaned = []

    for event in events:
        title = clean_text(event.get("title_fr") or event.get("title"))
        description = clean_text(event.get("description_fr") or event.get("description"))

        city = event.get("location_city")
        address = clean_text(event.get("location_address"))

        start_date = event.get("firstdate_begin")
        end_date = event.get("lastdate_end")

        keywords = event.get("keywords_fr")

        # Exclusion des événements incomplets
        if not title or not description:
            continue

        text_for_embedding = (
            f"Titre : {title}. "
            f"Description : {description}. "
            f"Lieu : {city}, {address}. "
            f"Date de début : {start_date}. "
            f"Date de fin : {end_date}. "
            f"Mots-clés : {keywords}."
        )

        cleaned.append({
            "title": title,
            "description": description,
            "city": city,
            "address": address,
            "start_date": start_date,
            "end_date": end_date,
            "keywords": keywords,
            "text_for_embedding": text_for_embedding
        })

    return pd.DataFrame(cleaned)


# ============================================================
# SAUVEGARDE DU DATASET
# ============================================================

def save_events(df, output_path="data/processed/events_montpellier.csv"):
    """
    Sauvegarde du dataset structuré au format CSV.

    Création automatique du dossier cible si nécessaire.

    Paramètres :
        df (DataFrame) : données nettoyées
        output_path (str) : chemin de sauvegarde

    Retour :
        Path : chemin du fichier généré
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")

    return output_path


# ============================================================
# POINT D'ENTRÉE DU SCRIPT
# ============================================================

if __name__ == "__main__":

    CITY = "Montpellier"

    # Extraction des données
    events = fetch_events(city=CITY, limit=100)
    print(f"{len(events)} événements bruts récupérés pour {CITY}")

    # Transformation des données
    df = clean_events(events)
    print(f"{len(df)} événements exploitables après nettoyage")

    # Sauvegarde
    path = save_events(df)

    print(f"Dataset sauvegardé dans : {path}")