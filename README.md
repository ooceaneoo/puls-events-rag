# Puls Events RAG

## Contexte du projet

Ce projet a été réalisé dans le cadre d’un projet de système RAG (Retrieval-Augmented Generation).

L’objectif est de construire un assistant capable de répondre à des questions sur des événements culturels à Montpellier à partir d’une base documentaire vectorisée.

Le système combine :
- la recherche documentaire via FAISS
- la génération de réponses avec Mistral AI
- une API REST FastAPI
- des tests automatisés et une évaluation RAG avec Ragas

---

# Objectifs du projet

Le projet permet de :

- collecter et nettoyer des données d’événements
- construire une base vectorielle FAISS
- interroger les données avec un système RAG
- exposer le système via une API REST
- évaluer automatiquement les réponses générées
- conteneuriser l’application avec Docker

---

# Technologies utilisées

## Backend et API
- Python 3.13
- FastAPI
- Uvicorn

## RAG et IA
- LangChain
- FAISS
- Mistral AI
- Mistral AI Embeddings (mistral-embed)

## Data
- Pandas

## Tests et évaluation
- Pytest
- Ragas

## Conteneurisation
- Docker

## CI/CD
- GitHub Actions

---

# Architecture du projet

```text
Utilisateur
   ↓
API FastAPI (/ask)
   ↓
Chaîne RAG LangChain
   ↓
Recherche vectorielle FAISS
   ↓
Récupération du contexte pertinent
   ↓
Mistral AI
   ↓
Réponse générée
```

---

# Fonctionnalités

## API REST

### GET `/health`
Vérifie que l’API fonctionne correctement.

### POST `/ask`
Interroge le système RAG à partir d’une question utilisateur.

### POST `/rebuild`
Reconstruit l’index vectoriel FAISS à partir des données nettoyées.

Cet endpoint est protégé par une clé API d’administration.

---

# Installation du projet

## Cloner le dépôt

```bash
git clone https://github.com/ooceaneoo/puls-events-rag.git
cd puls-events-rag
```

---

## Créer un environnement virtuel

### Windows

```bash
python -m venv venv
source venv/Scripts/activate
```

---

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Variables d’environnement

Créer un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=your_mistral_api_key
ADMIN_API_KEY=puls_events_admin_key
```

---

# Lancement local de l’API

## Démarrer FastAPI

```bash
PYTHONPATH=. uvicorn api.main:app --reload
```

---

## Documentation Swagger

Une fois l’API lancée :

```text
http://127.0.0.1:8000/docs
```

---

# Utilisation de l’API

## Endpoint `/ask`

Exemple de requête :

```json
{
  "question": "Quels événements culturels sont disponibles à Montpellier ?"
}
```

---

## Endpoint `/rebuild`

Header requis :

```text
x-api-key: puls_events_admin_key
```

Cet endpoint :
- recharge les données
- reconstruit les embeddings
- recrée l’index FAISS

---

# Docker

## Build de l’image Docker

```bash
docker build -t puls-events-rag-api .
```

---

## Lancement du conteneur

```bash
docker run --env-file .env -p 8000:8000 puls-events-rag-api
```

---

## Important

Le dossier `data/vectorstore/` est exclu du conteneur via `.dockerignore`.

Après le démarrage du conteneur, il est donc nécessaire d’appeler :

```text
POST /rebuild
```

avant d’utiliser `/ask`.

---

# Tests

## Tests unitaires

```bash
PYTHONPATH=. pytest
```

Les tests couvrent :
- le chargement des données
- la création de la base vectorielle
- la chaîne RAG
- les endpoints API FastAPI

---

# Évaluation du système RAG

Le projet utilise Ragas pour évaluer automatiquement la qualité des réponses générées.

Métriques utilisées :
- Faithfulness
- Context Precision
- Context Recall

Scripts disponibles :

## Génération des réponses pour revue manuelle

```bash
PYTHONPATH=. python scripts/generate_manual_review.py
```

## Évaluation automatique avec Ragas

```bash
PYTHONPATH=. python scripts/evaluate_rag.py
```

Les résultats sont sauvegardés dans :

```text
data/test/
```

---

# GitHub Actions

Le projet inclut un workflow GitHub Actions permettant d’exécuter automatiquement certains tests Pytest lors des push et pull requests.

Fichier :

```text
.github/workflows/tests.yml
```

---

# Structure du projet

```text
puls-events-rag/
│
├── api/
│   └── main.py
│
├── data/
│   ├── processed/
│   └── test/
│
├── scripts/
│   ├── evaluate_rag.py
│   └── generate_manual_review.py
│
├── src/
│   ├── rag/
│   └── vectorstore/
│
├── tests/
│   ├── test_api.py
│   ├── test_fetch_events.py
│   ├── test_rag_chain.py
│   └── test_vectorstore.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Résultats observés

Le système RAG développé permet de générer des réponses contextualisées à partir des événements culturels récupérés via l’API OpenAgenda.

L’évaluation du système a été réalisée en deux étapes :

* une revue manuelle des réponses générées ;
* une évaluation automatique avec la bibliothèque Ragas.

Un jeu de questions a été créé afin de tester la pertinence des réponses produites par le pipeline RAG.

### Revue manuelle

Les réponses générées ont été analysées manuellement afin de vérifier :

* la cohérence des réponses
* la pertinence des événements proposés
* la fidélité des informations retournées
* la présence éventuelle d’hallucinations

Les résultats de cette revue ont été sauvegardés dans :

```bash
data/test/manual_review_results.csv
```

### Évaluation automatique avec Ragas

Les métriques suivantes ont été utilisées :

| Métrique          | Score obtenu (k=3) |
| ----------------- | ------------------ |
| Faithfulness      | 0.8296             |
| Context Precision | 0.2569             |
| Context Recall    | 0.3750             |

### Interprétation des résultats

* Le score élevé de **faithfulness** montre que les réponses générées restent globalement cohérentes avec les informations réellement présentes dans les documents récupérés.
* Le score de **context recall** indique que le système parvient à retrouver une partie importante des informations utiles pour répondre aux questions utilisateur.
* Le score plus faible de **context precision** montre que certains documents récupérés par FAISS ne sont pas toujours totalement pertinents par rapport à la question posée.

Plusieurs valeurs du paramètre `k` ont été testées lors de la recherche vectorielle (`k=3`, `k=4`, etc.).

Le choix de `k=3` a finalement été retenu car il offre un meilleur compromis entre :

* pertinence des documents récupérés
* fidélité des réponses générées
* limitation du bruit dans le contexte envoyé au modèle

### Validation globale du système

Les différents tests réalisés montrent :

* un bon fonctionnement global du pipeline RAG
* une API REST stable et exploitable
* une reconstruction correcte de l’index FAISS
* une intégration fonctionnelle avec Docker
* une automatisation des tests via GitHub Actions

Le système reste toutefois sensible à la qualité des données OpenAgenda et peut encore produire certaines réponses imprécises.

---

## Perspectives d’amélioration

Plusieurs améliorations pourraient être ajoutées :
- déploiement cloud
- authentification plus robuste
- cache des embeddings
- ajout d’un frontend utilisateur
- optimisation des performances
- CI/CD plus avancée
- monitoring des requêtes API
