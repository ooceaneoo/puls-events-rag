import pandas as pd

from src.data.fetch_events import clean_text, clean_events, fetch_events


# ============================================================
# TEST DU NETTOYAGE DES TEXTES
# ============================================================

def test_clean_text_removes_html_and_extra_spaces():
    text = "<p>Concert   à Montpellier</p>\nAvec artistes locaux."

    cleaned = clean_text(text)

    assert cleaned == "Concert à Montpellier Avec artistes locaux."


# ============================================================
# TEST DE L'EXTRACTION API
# ============================================================

def test_fetch_events_returns_list():
    events = fetch_events(city="Montpellier", limit=5)

    assert isinstance(events, list)


# ============================================================
# TEST DE LA STRUCTURE DU DATASET FINAL
# ============================================================

def test_clean_events_returns_dataframe_with_expected_columns():
    fake_events = [
        {
            "title_fr": "Festival test",
            "description_fr": "Un événement culturel à Montpellier.",
            "location_city": "Montpellier",
            "location_address": "Place de la Comédie",
            "firstdate_begin": "2025-10-01",
            "lastdate_end": "2025-10-02",
            "keywords_fr": ["culture", "festival"]
        }
    ]

    df = clean_events(fake_events)

    expected_columns = [
        "title",
        "description",
        "city",
        "address",
        "start_date",
        "end_date",
        "keywords",
        "text_for_embedding"
    ]

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == expected_columns
    assert len(df) == 1


# ============================================================
# TEST DES DONNÉES MANQUANTES
# ============================================================

def test_clean_events_removes_events_without_title_or_description():
    fake_events = [
        {
            "title_fr": "",
            "description_fr": "Description présente",
            "location_city": "Montpellier"
        },
        {
            "title_fr": "Titre présent",
            "description_fr": "",
            "location_city": "Montpellier"
        }
    ]

    df = clean_events(fake_events)

    assert len(df) == 0


# ============================================================
# TEST DU TEXTE PRÊT POUR INDEXATION FUTURE
# ============================================================

def test_text_for_embedding_is_created():
    fake_events = [
        {
            "title_fr": "Exposition test",
            "description_fr": "Une exposition artistique.",
            "location_city": "Montpellier",
            "location_address": "Musée Fabre",
            "firstdate_begin": "2025-11-01",
            "lastdate_end": "2025-11-15",
            "keywords_fr": ["art", "exposition"]
        }
    ]

    df = clean_events(fake_events)

    text = df.loc[0, "text_for_embedding"]

    assert isinstance(text, str)
    assert "Exposition test" in text
    assert "Une exposition artistique" in text
    assert "Montpellier" in text