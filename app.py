"""
Verbatim Analyzer — POC Streamlit
Alternative open-source à Cobbai pour l'analyse de verbatims clients.
Clustering BERTopic + Sentiment Analysis + Dashboard interactif.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import pickle
import os
import json
import math
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Verbatim Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea10, #764ba210);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Load multilingual sentiment analysis pipeline on MPS (Apple Silicon GPU)."""
    import torch
    from transformers import pipeline
    device = "mps" if torch.backends.mps.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device,
        truncation=True,
        max_length=512,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load multilingual sentence-transformer for BERTopic."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def run_sentiment_analysis(texts: list[str], progress_bar=None) -> list[dict]:
    """Run sentiment analysis using ONNX-optimized pipeline."""
    pipe = load_sentiment_pipeline()
    sentiment_map = {1: "très négatif", 2: "négatif", 3: "neutre", 4: "positif", 5: "très positif"}
    results = []
    total = len(texts)
    batch_size = 64
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        preds = pipe(batch)
        for p in preds:
            stars = int(p["label"].split()[0])
            results.append({
                "label": sentiment_map.get(stars, "neutre"),
                "score": p["score"],
                "stars": stars,
                "numeric": (stars - 3) / 2,
            })
        if progress_bar is not None:
            done = min(i + batch_size, total)
            progress_bar.progress(done / total, text=f"Sentiment : {done}/{total} verbatims traités")
    return results


def _multilingual_stop_words() -> list[str]:
    """Return a combined stop words list for FR, EN, ES, DE, IT, PT, NL."""
    return list({
        # FR
        "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
        "au", "aux", "ce", "ces", "cet", "cette", "mon", "ma", "mes",
        "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre",
        "vos", "leur", "leurs",
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
        "me", "te", "se", "lui", "y", "en", "qui", "que", "qu", "quoi",
        "dont", "où", "quel", "quelle", "quels", "quelles",
        "à", "a", "et", "est", "en", "dans", "pour", "par", "sur", "avec",
        "sans", "sous", "entre", "vers", "chez", "mais", "ou", "donc",
        "or", "ni", "car", "si", "ne", "pas", "plus", "moins", "très",
        "trop", "aussi", "bien", "mal", "peu", "beaucoup", "tout", "tous",
        "toute", "toutes", "autre", "autres", "même", "mêmes",
        "être", "avoir", "faire", "dire", "aller", "voir", "pouvoir",
        "vouloir", "falloir", "devoir", "été", "fait", "ai", "avons",
        "avez", "ont", "suis", "es", "sommes", "sont", "était", "c",
        "j", "n", "s", "m", "t",
        "alors", "après", "avant", "comme", "comment", "encore", "déjà",
        "là", "ici", "quand", "depuis", "pendant", "jusqu", "jusque",
        "jamais", "toujours", "souvent", "vraiment", "peut", "cela",
        "celui", "celle", "ceux", "celles", "ça", "ci",
        # EN
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "this", "that", "these", "those", "it", "its", "we", "they",
        "i", "you", "he", "she", "my", "your", "his", "her", "our",
        "their", "not", "no", "so", "very", "just", "about", "also",
        # ES
        "el", "los", "las", "unos", "unas", "del", "al", "o", "es",
        "por", "con", "para", "no", "su", "sus", "lo", "como", "más",
        "pero", "muy", "sin", "sobre", "cuando", "también", "fue", "ha",
        "yo", "mi", "ti", "le", "les", "ser", "estar", "haber", "tener",
        "hacer", "poder", "decir", "ir", "ver", "este", "esta", "estos",
        "estas", "ese", "esa", "esos", "esas", "todo", "toda", "todos",
        "todas", "otro", "otra", "otros", "otras", "mucho", "mucha",
        "muchos", "muchas", "aquí", "ahí", "allí", "donde", "cual",
        "quien", "ya", "hay", "sido",
        # DE
        "der", "die", "das", "den", "dem", "ein", "eine", "einen",
        "einem", "einer", "und", "oder", "aber", "im", "an", "am",
        "auf", "aus", "bei", "mit", "nach", "von", "vom", "zu", "zum",
        "zur", "für", "über", "unter", "vor", "hinter", "zwischen",
        "ist", "sind", "war", "waren", "hat", "haben", "hatte", "wird",
        "werden", "kann", "können", "muss", "müssen", "soll", "wollen",
        "nicht", "auch", "noch", "schon", "sehr", "nur", "wenn", "als",
        "wie", "da", "er", "sie", "wir", "ich", "du", "mein", "dein",
        "sein", "ihr", "unser", "euer", "kein", "keine", "sich", "man",
        "was", "wer", "wo", "hier", "dort", "dass", "diese", "dieser",
        "dieses", "jeder", "jede", "jedes", "alle",
        # IT
        "il", "lo", "gli", "uno", "una", "di", "dello", "della",
        "dei", "degli", "delle", "allo", "alla", "ai", "agli", "alle",
        "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
        "nel", "nello", "nella", "nei", "negli", "nelle", "sul", "sullo",
        "sulla", "sui", "sugli", "sulle", "con", "per", "tra", "fra",
        "e", "ma", "che", "chi", "è", "sono", "era", "ho", "hai",
        "hanno", "non", "più", "molto", "tutto", "tutti", "questo",
        "questa", "questi", "queste", "quello", "quella", "quelli",
        "io", "lei", "noi", "voi", "loro", "vi", "dove", "perché",
        "ancora", "essere", "avere", "andare", "vedere", "potere",
        # PT
        "o", "os", "as", "um", "uma", "uns", "umas", "do", "dos",
        "das", "na", "nas", "sem", "sob", "são", "foi", "tem", "há",
        "não", "sim", "muito", "também", "já", "ainda",
        "eu", "ele", "ela", "nós", "vós", "eles", "elas", "meu",
        "minha", "seu", "sua", "nosso", "nossa", "esse", "essa",
        "aquele", "aquela", "toda", "estar", "haver",
        # NL
        "het", "een", "van", "dat", "op", "te", "voor", "wordt",
        "door", "ook", "naar", "bij", "uit", "tot", "niet", "nog",
        "wel", "er", "al", "dan", "dit", "zo", "meer", "veel", "heel",
        "ik", "je", "hij", "zij", "wij", "ze", "hun", "ons", "mijn",
        "jouw", "haar", "kan", "moet", "wil", "zal", "heeft", "hebben",
        "worden", "werd", "wat", "wie", "waar", "hoe",
    })


def run_clustering(texts: list[str], min_cluster_size: int = 15):
    """Run BERTopic clustering. Returns (topic_model, topics, probs)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    embedding_model = load_embedding_model()
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 3),
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=_multilingual_stop_words(),
        min_df=2,
        ngram_range=(1, 2),
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        verbose=False,
        nr_topics=None,
    )
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


def save_analysis(name: str, data: dict):
    """Persist analysis: DataFrame as Parquet, BERTopic model natively, metadata as JSON."""
    analysis_dir = DATA_DIR / name
    analysis_dir.mkdir(exist_ok=True)

    # Save DataFrame as Parquet (10-20x smaller than pickle)
    data["df"].to_parquet(analysis_dir / "verbatims.parquet", index=False)

    # Save BERTopic model using its native serialization
    topic_model = data.get("topic_model")
    if topic_model is not None:
        model_dir = analysis_dir / "topic_model"
        topic_model.save(model_dir, serialization="pickle", save_ctfidf=True, save_embedding_model=False)

    # Save lightweight metadata as JSON
    meta = {
        "topic_names": data.get("topic_names", {}),
        "meta_cols": data.get("meta_cols", []),
        "created_at": datetime.now().isoformat(),
        "row_count": len(data["df"]),
    }
    # topic_info as separate parquet (small)
    topic_info = data.get("topic_info")
    if topic_info is not None:
        topic_info.to_parquet(analysis_dir / "topic_info.parquet", index=False)

    with open(analysis_dir / "meta.json", "w") as f:
        # Convert dict keys to strings for JSON compatibility
        meta["topic_names"] = {str(k): v for k, v in meta["topic_names"].items()}
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_analysis(name: str) -> dict | None:
    """Load persisted analysis from Parquet + BERTopic native format."""
    analysis_dir = DATA_DIR / name

    # Support legacy pickle format
    legacy_path = DATA_DIR / f"{name}.pkl"
    if legacy_path.exists() and not analysis_dir.exists():
        with open(legacy_path, "rb") as f:
            return pickle.load(f)

    if not analysis_dir.exists():
        return None

    # Load DataFrame from Parquet
    parquet_path = analysis_dir / "verbatims.parquet"
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)

    # Load BERTopic model
    topic_model = None
    model_dir = analysis_dir / "topic_model"
    if model_dir.exists():
        try:
            from bertopic import BERTopic
            topic_model = BERTopic.load(model_dir)
        except Exception:
            pass  # Model loading is optional for dashboard display

    # Load metadata
    meta = {}
    meta_path = analysis_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Load topic_info
    topic_info = None
    topic_info_path = analysis_dir / "topic_info.parquet"
    if topic_info_path.exists():
        topic_info = pd.read_parquet(topic_info_path)

    # Restore topic_names with int keys
    topic_names = {int(k): v for k, v in meta.get("topic_names", {}).items()}

    return {
        "df": df,
        "topic_model": topic_model,
        "topic_info": topic_info,
        "topic_names": topic_names,
        "meta_cols": meta.get("meta_cols", []),
    }


def list_analyses() -> list[str]:
    """List saved analyses (both new Parquet format and legacy pickle)."""
    analyses = []
    # New format: directories with verbatims.parquet
    for d in DATA_DIR.iterdir():
        if d.is_dir() and (d / "verbatims.parquet").exists():
            analyses.append(d.name)
    # Legacy pickle format
    for f in DATA_DIR.glob("*.pkl"):
        if f.stem not in analyses:
            analyses.append(f.stem)
    return sorted(analyses, reverse=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def _load_analysis_into_session(name):
    """Load a saved analysis into session state."""
    data = load_analysis(name)
    if data:
        st.session_state["df"] = data["df"]
        st.session_state["topic_model"] = data.get("topic_model")
        st.session_state["topic_info"] = data.get("topic_info")
        st.session_state["topic_names"] = data.get("topic_names", {})
        st.session_state["meta_cols"] = data.get("meta_cols", [])
        st.session_state["analysis_name"] = name
        return True
    return False


def _get_analysis_label(name):
    """Build a human-readable label for an analysis entry."""
    meta_path = DATA_DIR / name / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_info = json.load(f)
        date_str = meta_info.get("created_at", "")[:16]
        rows = meta_info.get("row_count", "?")
        return f"{date_str} — {rows} lignes"
    return name


NEW_ANALYSIS_KEY = "➕ Nouvelle analyse"

with st.sidebar:
    st.title("🔍 Verbatim Analyzer")
    st.caption("Alternative open-source à Cobbai")
    st.divider()

    # --- Analysis selector dropdown ---
    existing = list_analyses()

    # Auto-load latest on first visit
    if "df" not in st.session_state and existing and "_auto_loaded" not in st.session_state:
        if _load_analysis_into_session(existing[0]):
            st.session_state["_auto_loaded"] = True
            st.rerun()

    # Build dropdown options: "New analysis" + saved analyses
    options = [NEW_ANALYSIS_KEY] + existing
    # Format function for display
    def _format_option(opt):
        if opt == NEW_ANALYSIS_KEY:
            return NEW_ANALYSIS_KEY
        return f"📂 {_get_analysis_label(opt)}"

    # Determine default index: current loaded analysis, or latest, or new
    current_name = st.session_state.get("analysis_name", "")
    if current_name in existing:
        default_idx = options.index(current_name)
    elif existing:
        default_idx = 1  # latest saved
    else:
        default_idx = 0  # new analysis

    selected_analysis = st.selectbox(
        "Analyse",
        options,
        index=default_idx,
        format_func=_format_option,
        key="_analysis_selector",
    )

    # Switch analysis when dropdown changes
    if selected_analysis != NEW_ANALYSIS_KEY:
        if st.session_state.get("analysis_name") != selected_analysis:
            if _load_analysis_into_session(selected_analysis):
                st.rerun()
    else:
        # Switched to "New analysis" — clear current data
        if "df" in st.session_state and st.session_state.get("analysis_name") != "":
            for key in ["df", "topic_model", "topic_info", "topic_names", "meta_cols", "analysis_name", "_uploaded_df"]:
                st.session_state.pop(key, None)
            st.rerun()

    is_new_analysis = selected_analysis == NEW_ANALYSIS_KEY
    has_analysis = "df" in st.session_state

    st.divider()

    # --- CSV file uploader (always visible) ---
    if is_new_analysis:
        st.markdown("##### Importer un CSV")
    else:
        st.markdown("##### Ajouter des verbatims")
        if has_analysis:
            st.caption("Les données seront fusionnées avec l'analyse en cours.")

    uploaded = st.file_uploader("CSV de feedback", type=["csv"], key="csv_upload")

    # Quick-load sample data
    sample_path = Path("sample_data.csv")
    if sample_path.exists():
        btn_label = "📋 Charger les données d'exemple" if is_new_analysis else "📋 Ajouter les données d'exemple"
        if st.button(btn_label, use_container_width=True):
            st.session_state["_uploaded_df"] = pd.read_csv(sample_path, encoding="utf-8")
            st.rerun()

    if uploaded:
        try:
            df = pd.read_csv(uploaded, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="latin-1")
        except pd.errors.ParserError:
            uploaded.seek(0)
            df = pd.read_csv(
                uploaded,
                encoding="utf-8",
                on_bad_lines="skip",
                engine="python",
                quoting=1,  # QUOTE_ALL
            )
            st.warning("⚠️ Certaines lignes mal formatées ont été ignorées.")
        st.session_state["_uploaded_df"] = df

    if "_uploaded_df" in st.session_state:
        df = st.session_state["_uploaded_df"]
    else:
        df = None

    if df is not None:
        if has_analysis and not is_new_analysis:
            existing_count = len(st.session_state["df"])
            st.success(f"✅ {len(df)} nouvelles lignes — {existing_count} existantes")
        else:
            st.success(f"✅ {len(df)} lignes importées")
        st.dataframe(df.head(3), use_container_width=True, height=140)

        # ------ Column mapping ------
        cols = list(df.columns)
        st.markdown("##### Mapping des colonnes")

        verbatim_col = st.selectbox(
            "Colonne verbatim",
            cols,
            index=next((i for i, c in enumerate(cols) if any(k in c.lower() for k in ["verbatim", "comment", "avis", "feedback", "text"])), 0),
        )
        score_col = st.selectbox(
            "Colonne score (NPS / étoiles)",
            ["(aucun)"] + cols,
            index=next((i + 1 for i, c in enumerate(cols) if any(k in c.lower() for k in ["score", "nps", "note", "rating", "star"])), 0),
        )
        date_col = st.selectbox(
            "Colonne date",
            ["(aucun)"] + cols,
            index=next((i + 1 for i, c in enumerate(cols) if any(k in c.lower() for k in ["date", "created", "timestamp"])), 0),
        )

        # Metadata columns = everything else
        meta_cols = [c for c in cols if c not in [verbatim_col, score_col if score_col != "(aucun)" else None, date_col if date_col != "(aucun)" else None] and c is not None]
        selected_meta = st.multiselect("Colonnes métadonnées (filtres)", meta_cols, default=meta_cols[:5])

        # --- ENRICHMENT MODE: add to existing analysis ---
        if has_analysis and not is_new_analysis:
            if st.button("🔄 Fusionner et ré-analyser", use_container_width=True, type="primary"):
                existing_df = st.session_state["df"]
                existing_count = len(existing_df)

                # Prepare new data
                df = df.dropna(subset=[verbatim_col])
                df = df[df[verbatim_col].str.strip().str.len() > 5].reset_index(drop=True)
                df["_verbatim"] = df[verbatim_col].astype(str)

                if score_col != "(aucun)":
                    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
                else:
                    df["_score"] = np.nan

                if date_col != "(aucun)":
                    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
                else:
                    df["_date"] = pd.NaT

                df["_meta_cols"] = json.dumps(selected_meta)

                # Deduplicate
                existing_verbatims = set(existing_df["_verbatim"].tolist())
                new_rows = df[~df["_verbatim"].isin(existing_verbatims)].copy()
                n_dupes = len(df) - len(new_rows)
                if n_dupes > 0:
                    st.info(f"ℹ️ {n_dupes} doublons ignorés")

                if len(new_rows) == 0:
                    st.warning("⚠️ Aucun nouveau verbatim à ajouter (tous des doublons).")
                else:
                    # Sentiment on new rows only
                    st.markdown(f"**Étape 1/3 — Sentiment sur {len(new_rows)} nouveaux verbatims**")
                    sentiment_bar = st.progress(0, text=f"Sentiment : 0/{len(new_rows)}")
                    new_texts = new_rows["_verbatim"].tolist()
                    sentiments = run_sentiment_analysis(new_texts, progress_bar=sentiment_bar)
                    new_rows["_sentiment_label"] = [s["label"] for s in sentiments]
                    new_rows["_sentiment_score"] = [s["numeric"] for s in sentiments]
                    new_rows["_sentiment_stars"] = [s["stars"] for s in sentiments]
                    sentiment_bar.progress(1.0, text=f"Sentiment : {len(new_rows)}/{len(new_rows)} ✅")

                    # Merge
                    st.markdown("**Étape 2/3 — Fusion des données**")
                    internal_cols = ["_verbatim", "_score", "_date", "_meta_cols",
                                     "_sentiment_label", "_sentiment_score", "_sentiment_stars"]
                    existing_meta_cols = st.session_state.get("meta_cols", [])
                    all_meta = sorted(set(existing_meta_cols) | set(selected_meta))
                    keep_cols = [c for c in internal_cols + all_meta if c in existing_df.columns or c in new_rows.columns]

                    merged = pd.concat([existing_df[keep_cols], new_rows[keep_cols]], ignore_index=True)
                    st.info(f"📊 {existing_count} + {len(new_rows)} = **{len(merged)} verbatims**")

                    # Re-cluster everything
                    st.markdown(f"**Étape 3/3 — Re-clustering sur {len(merged)} verbatims**")
                    cluster_bar = st.progress(0, text="Clustering : génération des embeddings...")
                    all_texts = merged["_verbatim"].tolist()
                    min_cs = max(5, len(all_texts) // 100)
                    cluster_bar.progress(0.3, text="Clustering : regroupement en cours...")
                    topic_model, topics, probs = run_clustering(all_texts, min_cluster_size=min_cs)
                    cluster_bar.progress(0.9, text="Clustering : attribution des topics...")
                    merged["_topic"] = topics
                    topic_info = topic_model.get_topic_info()
                    cluster_bar.progress(1.0, text="Clustering terminé ✅")

                    topic_names = {}
                    for _, row in topic_info.iterrows():
                        tid = row["Topic"]
                        if tid == -1:
                            topic_names[tid] = "Non classé"
                        else:
                            name = row.get("Name", f"Topic {tid}")
                            parts = str(name).split("_")
                            if len(parts) > 1 and parts[0].isdigit():
                                topic_names[tid] = ", ".join(parts[1:4])
                            else:
                                topic_names[tid] = name
                    merged["_topic_name"] = merged["_topic"].map(topic_names)

                    st.session_state["df"] = merged
                    st.session_state["topic_model"] = topic_model
                    st.session_state["topic_info"] = topic_info
                    st.session_state["topic_names"] = topic_names
                    st.session_state["meta_cols"] = all_meta

                    analysis_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    save_analysis(analysis_name, {
                        "df": merged,
                        "topic_model": topic_model,
                        "topic_info": topic_info,
                        "topic_names": topic_names,
                        "meta_cols": all_meta,
                    })
                    st.session_state["analysis_name"] = analysis_name

                    if "_uploaded_df" in st.session_state:
                        del st.session_state["_uploaded_df"]

                    st.success(f"✅ Analyse enrichie ! {len(merged)} verbatims au total.")
                    st.rerun()

        # --- NEW ANALYSIS MODE ---
        else:
            if st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary"):
                df = df.dropna(subset=[verbatim_col])
                df = df[df[verbatim_col].str.strip().str.len() > 5].reset_index(drop=True)
                df["_verbatim"] = df[verbatim_col].astype(str)

                if score_col != "(aucun)":
                    df["_score"] = pd.to_numeric(df[score_col], errors="coerce")
                else:
                    df["_score"] = np.nan

                if date_col != "(aucun)":
                    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
                else:
                    df["_date"] = pd.NaT

                df["_meta_cols"] = json.dumps(selected_meta)

                texts = df["_verbatim"].tolist()

                st.markdown("**Étape 1/2 — Analyse de sentiment**")
                sentiment_bar = st.progress(0, text=f"Sentiment : 0/{len(texts)} verbatims traités")
                sentiments = run_sentiment_analysis(texts, progress_bar=sentiment_bar)
                df["_sentiment_label"] = [s["label"] for s in sentiments]
                df["_sentiment_score"] = [s["numeric"] for s in sentiments]
                df["_sentiment_stars"] = [s["stars"] for s in sentiments]
                sentiment_bar.progress(1.0, text=f"Sentiment : {len(texts)}/{len(texts)} ✅")

                st.markdown("**Étape 2/2 — Clustering des thématiques**")
                cluster_bar = st.progress(0, text="Clustering : génération des embeddings...")
                min_cs = max(5, len(texts) // 100)
                cluster_bar.progress(0.3, text="Clustering : embeddings générés, regroupement en cours...")
                topic_model, topics, probs = run_clustering(texts, min_cluster_size=min_cs)
                cluster_bar.progress(0.9, text="Clustering : attribution des topics...")
                df["_topic"] = topics
                topic_info = topic_model.get_topic_info()
                cluster_bar.progress(1.0, text="Clustering terminé ✅")

                topic_names = {}
                for _, row in topic_info.iterrows():
                    tid = row["Topic"]
                    if tid == -1:
                        topic_names[tid] = "Non classé"
                    else:
                        name = row.get("Name", f"Topic {tid}")
                        parts = str(name).split("_")
                        if len(parts) > 1 and parts[0].isdigit():
                            topic_names[tid] = ", ".join(parts[1:4])
                        else:
                            topic_names[tid] = name
                df["_topic_name"] = df["_topic"].map(topic_names)

                st.session_state["df"] = df
                st.session_state["topic_model"] = topic_model
                st.session_state["topic_info"] = topic_info
                st.session_state["topic_names"] = topic_names
                st.session_state["meta_cols"] = selected_meta
                st.session_state["verbatim_col"] = verbatim_col

                analysis_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}"
                save_analysis(analysis_name, {
                    "df": df,
                    "topic_model": topic_model,
                    "topic_info": topic_info,
                    "topic_names": topic_names,
                    "meta_cols": selected_meta,
                })
                st.session_state["analysis_name"] = analysis_name

                if "_uploaded_df" in st.session_state:
                    del st.session_state["_uploaded_df"]

                st.success("✅ Analyse terminée !")
                st.rerun()

    st.divider()
    st.caption("Built with BERTopic + HuggingFace + Streamlit")


# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.markdown("## 🔍 Verbatim Analyzer")
    st.markdown("### Importez un CSV de feedback pour commencer l'analyse")
    st.info("👈 Utilisez la barre latérale pour importer vos données ou charger une analyse existante.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 Clustering IA")
        st.markdown("Détection automatique des thématiques récurrentes via BERTopic")
    with col2:
        st.markdown("#### 💬 Sentiment")
        st.markdown("Analyse du sentiment multilingue par verbatim et par cluster")
    with col3:
        st.markdown("#### 🎯 Priorisation")
        st.markdown("Top/flop sujets avec drill-down vers les verbatims")
    st.stop()

df = st.session_state["df"]
meta_cols = st.session_state.get("meta_cols", json.loads(df.iloc[0].get("_meta_cols", "[]")) if "_meta_cols" in df.columns else [])

# ---------------------------------------------------------------------------
# Filters in sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Filtres")

    # Sentiment filter
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["très positif", "positif", "neutre", "négatif", "très négatif"],
        default=None,
    )

    # Score filter
    if "_score" in df.columns and df["_score"].notna().any():
        score_range = st.slider(
            "Score (NPS / étoiles)",
            float(df["_score"].min()),
            float(df["_score"].max()),
            (float(df["_score"].min()), float(df["_score"].max())),
        )
    else:
        score_range = None

    # Date filter
    if "_date" in df.columns and df["_date"].notna().any():
        min_d = df["_date"].min().date()
        max_d = df["_date"].max().date()
        date_range = st.date_input("Période", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    else:
        date_range = None

    # Metadata filters
    meta_filters = {}
    for col in meta_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:  # only show filter if reasonable cardinality
                selected = st.multiselect(f"{col}", sorted(unique_vals.astype(str)), default=None)
                if selected:
                    meta_filters[col] = selected

# Apply filters
filtered = df.copy()
if sentiment_filter:
    filtered = filtered[filtered["_sentiment_label"].isin(sentiment_filter)]
if score_range:
    filtered = filtered[(filtered["_score"] >= score_range[0]) & (filtered["_score"] <= score_range[1])]
if date_range and len(date_range) == 2:
    has_date = filtered["_date"].notna()
    filtered = filtered[
        ~has_date | ((filtered["_date"].dt.date >= date_range[0]) & (filtered["_date"].dt.date <= date_range[1]))
    ]
for col, vals in meta_filters.items():
    filtered = filtered[filtered[col].astype(str).isin(vals)]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
st.markdown(f"## 🔍 Dashboard — {len(filtered)} verbatims")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Verbatims", f"{len(filtered):,}")
with kpi2:
    n_topics = filtered[filtered["_topic"] != -1]["_topic"].nunique()
    st.metric("Thématiques", n_topics)
with kpi3:
    avg_sent = filtered["_sentiment_score"].mean()
    st.metric("Sentiment moyen", f"{avg_sent:+.2f}", delta=None)
with kpi4:
    pct_pos = (filtered["_sentiment_label"].isin(["positif", "très positif"]).sum() / len(filtered) * 100) if len(filtered) > 0 else 0
    st.metric("% Positif", f"{pct_pos:.0f}%")
with kpi5:
    pct_neg = (filtered["_sentiment_label"].isin(["négatif", "très négatif"]).sum() / len(filtered) * 100) if len(filtered) > 0 else 0
    st.metric("% Négatif", f"{pct_neg:.0f}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_map, tab_topics, tab_verbatims, tab_evolution = st.tabs([
    "🫧 Carte des thèmes", "📊 Top / Flop sujets", "💬 Verbatims", "📈 Évolution"
])

# ===== TAB 1: BUBBLE MAP =====
with tab_map:
    # Build topic summary
    topic_summary = (
        filtered[filtered["_topic"] != -1]
        .groupby(["_topic", "_topic_name"])
        .agg(
            count=("_verbatim", "size"),
            avg_sentiment=("_sentiment_score", "mean"),
            avg_stars=("_sentiment_stars", "mean"),
        )
        .reset_index()
    )
    topic_summary["occurrence"] = (topic_summary["count"] / len(filtered) * 100).round(1)

    if not topic_summary.empty:
        fig_bubble = px.scatter(
            topic_summary,
            x="avg_sentiment",
            y="occurrence",
            size="count",
            color="avg_sentiment",
            color_continuous_scale=["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71", "#27ae60"],
            range_color=[-1, 1],
            hover_name="_topic_name",
            hover_data={"count": True, "occurrence": ":.1f", "avg_sentiment": ":.2f"},
            size_max=60,
            text="_topic_name",
            labels={
                "avg_sentiment": "Sentiment moyen",
                "occurrence": "Occurrence (%)",
                "count": "Nb verbatims",
            },
        )
        fig_bubble.update_traces(textposition="top center", textfont_size=11)
        fig_bubble.update_layout(
            height=550,
            coloraxis_colorbar_title="Sentiment",
            xaxis_title="← Négatif — Sentiment — Positif →",
            yaxis_title="Occurrence (%)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_bubble.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        st.plotly_chart(fig_bubble, use_container_width=True)

        # Click to drill down
        selected_topic = st.selectbox(
            "🔎 Drill-down sur un thème",
            ["(tous)"] + sorted(topic_summary["_topic_name"].tolist()),
        )
        if selected_topic != "(tous)":
            drilldown = filtered[filtered["_topic_name"] == selected_topic]
            st.markdown(f"**{len(drilldown)} verbatims** dans le thème « {selected_topic} »")

            # Show sentiment distribution for this topic
            c1, c2 = st.columns([1, 2])
            with c1:
                sent_dist = drilldown["_sentiment_label"].value_counts()
                fig_pie = px.pie(
                    values=sent_dist.values,
                    names=sent_dist.index,
                    color=sent_dist.index,
                    color_discrete_map={
                        "très positif": "#27ae60", "positif": "#2ecc71",
                        "neutre": "#f1c40f", "négatif": "#e67e22", "très négatif": "#e74c3c"
                    },
                    hole=0.4,
                )
                fig_pie.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                display_cols = ["_verbatim", "_sentiment_label"]
                if "_score" in drilldown.columns:
                    display_cols.append("_score")
                st.dataframe(
                    drilldown[display_cols].head(20).rename(columns={
                        "_verbatim": "Verbatim", "_sentiment_label": "Sentiment", "_score": "Score"
                    }),
                    use_container_width=True,
                    height=250,
                )
    else:
        st.info("Aucun cluster trouvé avec les filtres actuels.")


# ===== TAB 2: TOP / FLOP =====
with tab_topics:
    if not topic_summary.empty:
        col_top, col_flop = st.columns(2)

        sorted_topics = topic_summary.sort_values("avg_sentiment", ascending=False)

        with col_top:
            st.markdown("### ✅ Top sujets (positifs)")
            top5 = sorted_topics.head(7)
            fig_top = px.bar(
                top5, y="_topic_name", x="avg_sentiment", orientation="h",
                color="avg_sentiment",
                color_continuous_scale=["#f1c40f", "#2ecc71", "#27ae60"],
                text="count",
                labels={"_topic_name": "", "avg_sentiment": "Sentiment", "count": "Volume"},
            )
            fig_top.update_layout(height=350, showlegend=False, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_flop:
            st.markdown("### ❌ Flop sujets (négatifs)")
            flop5 = sorted_topics.tail(7).sort_values("avg_sentiment")
            fig_flop = px.bar(
                flop5, y="_topic_name", x="avg_sentiment", orientation="h",
                color="avg_sentiment",
                color_continuous_scale=["#e74c3c", "#e67e22", "#f1c40f"],
                text="count",
                labels={"_topic_name": "", "avg_sentiment": "Sentiment", "count": "Volume"},
            )
            fig_flop.update_layout(height=350, showlegend=False, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_flop, use_container_width=True)

        # Volume distribution
        st.markdown("### 📊 Répartition par volume")
        vol_sorted = topic_summary.sort_values("count", ascending=True).tail(15)
        fig_vol = px.bar(
            vol_sorted, y="_topic_name", x="count", orientation="h",
            color="avg_sentiment",
            color_continuous_scale=["#e74c3c", "#f1c40f", "#27ae60"],
            range_color=[-1, 1],
            text="occurrence",
            labels={"_topic_name": "", "count": "Nombre de verbatims", "occurrence": "Occurrence %"},
        )
        fig_vol.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_vol.update_layout(height=450, yaxis=dict(autorange="reversed"), coloraxis_colorbar_title="Sentiment")
        st.plotly_chart(fig_vol, use_container_width=True)

        # Sentiment distribution overall
        st.markdown("### 🎭 Distribution du sentiment")
        sent_order = ["très négatif", "négatif", "neutre", "positif", "très positif"]
        sent_counts = filtered["_sentiment_label"].value_counts().reindex(sent_order, fill_value=0)
        fig_sent = px.bar(
            x=sent_counts.index, y=sent_counts.values,
            color=sent_counts.index,
            color_discrete_map={
                "très positif": "#27ae60", "positif": "#2ecc71",
                "neutre": "#f1c40f", "négatif": "#e67e22", "très négatif": "#e74c3c"
            },
            labels={"x": "Sentiment", "y": "Nombre de verbatims"},
        )
        fig_sent.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)


# ===== TAB 3: VERBATIMS =====
with tab_verbatims:
    st.markdown("### 💬 Explorer les verbatims")

    # Search
    search = st.text_input("🔍 Rechercher dans les verbatims", "")

    display_df = filtered.copy()
    if search:
        display_df = display_df[display_df["_verbatim"].str.contains(search, case=False, na=False)]

    # Topic filter for this tab
    topic_filter = st.multiselect(
        "Filtrer par thème",
        sorted(filtered["_topic_name"].dropna().unique()),
        default=None,
    )
    if topic_filter:
        display_df = display_df[display_df["_topic_name"].isin(topic_filter)]

    st.markdown(f"**{len(display_df)} verbatims**")

    # Display
    show_cols = ["_verbatim", "_topic_name", "_sentiment_label"]
    if "_score" in display_df.columns and display_df["_score"].notna().any():
        show_cols.append("_score")
    for mc in meta_cols[:3]:  # Show first 3 metadata cols
        if mc in display_df.columns:
            show_cols.append(mc)

    rename_map = {
        "_verbatim": "Verbatim",
        "_topic_name": "Thème",
        "_sentiment_label": "Sentiment",
        "_score": "Score",
    }

    st.dataframe(
        display_df[show_cols].rename(columns=rename_map),
        use_container_width=True,
        height=500,
        column_config={
            "Verbatim": st.column_config.TextColumn(width="large"),
            "Sentiment": st.column_config.TextColumn(width="small"),
        },
    )

    # Export
    csv_export = display_df[show_cols].rename(columns=rename_map).to_csv(index=False)
    st.download_button("📥 Exporter en CSV", csv_export, "verbatims_export.csv", "text/csv")


# ===== TAB 4: EVOLUTION =====
with tab_evolution:
    if "_date" in filtered.columns and filtered["_date"].notna().any():
        st.markdown("### 📈 Évolution temporelle")

        dated = filtered.dropna(subset=["_date"]).copy()
        dated["_month"] = dated["_date"].dt.to_period("M").dt.to_timestamp()

        # Volume over time
        vol_time = dated.groupby("_month").size().reset_index(name="count")
        fig_vol_time = px.line(vol_time, x="_month", y="count", markers=True, labels={"_month": "", "count": "Verbatims"})
        fig_vol_time.update_layout(height=300)
        st.plotly_chart(fig_vol_time, use_container_width=True)

        # Sentiment over time
        sent_time = dated.groupby("_month")["_sentiment_score"].mean().reset_index()
        fig_sent_time = px.line(
            sent_time, x="_month", y="_sentiment_score", markers=True,
            labels={"_month": "", "_sentiment_score": "Sentiment moyen"},
        )
        fig_sent_time.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
        fig_sent_time.update_layout(height=300)
        st.plotly_chart(fig_sent_time, use_container_width=True)

        # Top topics over time
        st.markdown("### 📊 Thèmes dans le temps")
        top_topics = topic_summary.nlargest(8, "count")["_topic_name"].tolist()
        topic_time = (
            dated[dated["_topic_name"].isin(top_topics)]
            .groupby(["_month", "_topic_name"])
            .size()
            .reset_index(name="count")
        )
        if not topic_time.empty:
            fig_topic_time = px.line(
                topic_time, x="_month", y="count", color="_topic_name",
                markers=True,
                labels={"_month": "", "count": "Mentions", "_topic_name": "Thème"},
            )
            fig_topic_time.update_layout(height=400)
            st.plotly_chart(fig_topic_time, use_container_width=True)
    else:
        st.info("Pas de colonne date disponible pour l'analyse temporelle.")

    # Word cloud (bonus)
    st.markdown("### ☁️ Nuage de mots")
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        en_col = "verbatim_en" if "verbatim_en" in filtered.columns else "_verbatim"
        text_all = " ".join(filtered[en_col].dropna().tolist())
        wc = WordCloud(
            width=800, height=300,
            background_color="white",
            colormap="viridis",
            max_words=80,
            collocations=False,
            stopwords=set(_multilingual_stop_words()),
        ).generate(text_all)
        fig_wc, ax = plt.subplots(figsize=(10, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
    except Exception:
        st.info("Wordcloud non disponible.")
