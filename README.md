# 🔍 Verbatim Analyzer

Alternative open-source à Cobbai pour l'analyse automatisée des verbatims clients.

## Features

- **Import CSV** — Upload avec mapping automatique des colonnes
- **Clustering IA** — Détection automatique des thématiques via BERTopic (multilingue)
- **Analyse de sentiment** — Score par verbatim via HuggingFace (multilingue)
- **Bubble Map** — Cartographie interactive des thèmes (taille=volume, couleur=sentiment)
- **Top/Flop** — Priorisation des sujets positifs et négatifs
- **Filtres** — Par sentiment, score, date, et toutes les métadonnées du CSV
- **Drill-down** — Clic sur un thème → voir les verbatims associés
- **Évolution temporelle** — Volume et sentiment dans le temps
- **Word Cloud** — Nuage de mots par thème ou global
- **Export CSV** — Exporter les résultats filtrés
- **Persistance** — Sauvegarder et recharger les analyses

## Déploiement rapide

### Option 1 : Docker (recommandé pour la prod)

```bash
docker build -t verbatim-analyzer .
docker run -p 8501:8501 -v $(pwd)/data:/app/data verbatim-analyzer
```

Accéder à http://localhost:8501

### Option 2 : Local (dev)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 3 : Streamlit Community Cloud (gratuit)

1. Push ce repo sur GitHub
2. Aller sur https://share.streamlit.io
3. Connecter le repo → Deploy

## Format CSV attendu

Le CSV doit contenir au minimum une colonne de texte (verbatim).
Les colonnes optionnelles sont auto-détectées :

| Colonne | Détection auto | Exemple |
|---------|---------------|---------|
| Verbatim/commentaire | `verbatim`, `comment`, `avis`, `feedback`, `text` | "Le voyage était génial" |
| Score/NPS | `score`, `nps`, `note`, `rating` | 4.5 |
| Date | `date`, `created`, `timestamp` | 2024-03-15 |
| Métadonnées | Tout le reste | destination, agence, marché... |

## Stack technique

- **Streamlit** — Interface web
- **BERTopic** — Clustering de topics (sentence-transformers + HDBSCAN + UMAP)
- **HuggingFace Transformers** — Sentiment analysis multilingue
- **Plotly** — Visualisations interactives
- **pandas** — Manipulation de données

## Limites du POC

- Pas d'authentification (à ajouter via Streamlit auth ou reverse proxy)
- Pas d'alerting Slack (v2)
- Pas de validation/merge manuelle des clusters (v2)
- Stockage en fichier pickle (pas de BDD)
- Performance ~30s pour 5000 verbatims sur CPU

## Roadmap v2

- [ ] Auth basique (Streamlit secrets / OAuth)
- [ ] Merge / rename / split clusters via UI
- [ ] Alerting Slack (webhooks)
- [ ] Import automatisé (cron / API)
- [ ] PostgreSQL + pgvector pour la persistance
- [ ] Comparaison entre deux périodes
