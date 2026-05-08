#
# Streamlit Transcript Scanner — 
# Candidate Fragments → Suggested Relations
#
# semantic_transcript_scanner.py
#++++++++++++++++++++++++++++++++++++++++++
# URUCHOMIENIE
# streamlit run semantic_transcript_scanner.py
#+++++++++++++++++++++++++++++++++++++++++
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import chardet

import re

POLISH_LABELS = {

    "autonomous_operation":
        "Autonomiczna operacja",

    "continuous_transport":
        "Transport ciągły",

    "labor_costs":
        "Koszty pracy",

    "social_acceptance":
        "Akceptacja społeczna",

    "technological_trust":
        "Zaufanie technologiczne",

    "traffic_efficiency":
        "Efektywność ruchu",

    "signal_visibility":
        "Czytelność oznakowania",

    "supporting_infrastructure":
        "Infrastruktura wspomagająca",

    "infrastructure_optimization":
        "Optymalizacja infrastruktury",

    "regulatory_governance":
        "Regulacje i nadzór",

    "regulatory_instability":
        "Niestabilność regulacyjna",

    "cross_border_interoperability":
        "Interoperacyjność transgraniczna",

    "training_requirements":
        "Wymagania szkoleniowe",

    "consumer_safety":
        "Bezpieczeństwo konsumenta",

    "energy_source":
        "Źródło energii",

    "energy_demand":
        "Zapotrzebowanie energetyczne",

    "deployment_risk":
        "Ryzyko wdrożenia",

    "operational_efficiency":
        "Efektywność operacyjna",

    "economic_stability":
        "Stabilność ekonomiczna",

    "business_acceptance":
        "Akceptacja biznesowa",

    "continuous_transport":
        "Transport ciągły",

    "workforce_efficiency":
        "Efektywność pracy",

    "data_management":
        "Zarządzanie danymi",
    "technological_maturity":
        "Dojrzałość technologiczna",

    "research_support":
        "Wsparcie badań i rozwoju"
}

POLISH_RELATIONS = {

    "amplifies": "wzmacnia",
    "suppresses": "osłabia",
    "enables": "umożliwia",
    "reduces": "redukuje",
    "conditions": "warunkuje",
    "depends_on": "zależy od",
    "stabilizes": "stabilizuje",
    "conflicts_with": "konflikt z",
    "constrains": "ogranicza"
}

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "dataverse_files"
TRANSCRIPTS_DIR = DATA_DIR / "transkrypcje"

OUTPUT_DIR = BASE_DIR / "output"

AI_OUTPUT_DIR = BASE_DIR / "output_ai"

AI_OUTPUT_DIR.mkdir(exist_ok=True)

EDGES_FILE = OUTPUT_DIR / "relational_edges.csv"

# =====================================================
# LOAD TRANSCRIPTS
# =====================================================

transcript_files = sorted(
    list(TRANSCRIPTS_DIR.glob("*.txt"))
)


selected_file = st.selectbox(
    "Select transcript",
    transcript_files,
    format_func=lambda x: x.name,
)

# =========================================
# RESET SESSION STATE AFTER FILE CHANGE
# =========================================

if "last_selected_file" not in st.session_state:
    st.session_state.last_selected_file = None

if (
    st.session_state.last_selected_file
    != selected_file.name
):

    st.session_state.scan_results = []

    st.session_state.last_selected_file = (
        selected_file.name
    )

AI_RELATIONS_FILE = (
    AI_OUTPUT_DIR /
    f"relational_edges_temp_"
    f"{selected_file.stem}.csv"
)

# =====================================================
# LOAD TRANSCRIPT
# =====================================================


with open(selected_file, "rb") as f:

    raw_data = f.read()

encoding = chardet.detect(raw_data)["encoding"]

if encoding is None:
    encoding = "utf-8"

transcript_text = raw_data.decode(
    encoding,
    errors="ignore"
)

# =====================================================
# SEGMENTATION
# =====================================================

def segment_text(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    segments = []

    for s in sentences:

        s = s.strip()

        if s.endswith("?"):
            continue

        if 20 < len(s) < 300:    
            segments.append(s)

    return segments

segments = segment_text(
    transcript_text
)

st.write(
    f"Loaded segments: {len(segments)}"
)

# =====================================================
# SEMANTIC TRANSCRIPT SCANNER
# =====================================================

st.markdown("---")
st.subheader("Semantic Transcript Scanner")

# =====================================================
# LOAD TRAINING DATA
# =====================================================

OUTPUT_DIR = Path("output")

csv_files = list(
    OUTPUT_DIR.glob("*.csv")
)

if len(csv_files) == 0:

    st.error(
        "No CSV relation files found in /output"
    )

    st.stop()

all_dfs = []

for file in csv_files:

    try:

        temp_df = None

        loading_configs = [

            {"encoding": "utf-8-sig"},
            {"encoding": "utf-16"},
            {
                "encoding": "cp1250",
                "engine": "python"
            }
        ]

        for cfg in loading_configs:

            try:

                temp_df = pd.read_csv(
                    file,
                    **cfg
                )

                st.write(
                    f"Loaded: {file.name}"
                )

                break

            except:
                pass

        if temp_df is None:
            continue

        all_dfs.append(temp_df)

    except Exception as e:

        st.warning(
            f"Skipping {file.name}: {e}"
        )

training_df = pd.concat(
    all_dfs,
    ignore_index=True
)

training_df = training_df.dropna(
    subset=["segment_text"]
)

st.success(
    f"Loaded relations: "
    f"{len(training_df)}"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

model = load_model()

# =====================================================
# BUILD EMBEDDINGS
# =====================================================

@st.cache_data
def build_embeddings(texts):

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return emb

training_texts = (
    training_df["segment_text"]
    .astype(str)
    .tolist()
)

training_embeddings = build_embeddings(
    training_texts
)

# =====================================================
# PARAMETERS
# =====================================================

SIMILARITY_THRESHOLD = st.slider(
    "Similarity threshold",
    min_value=0.40,
    max_value=0.95,
    value=0.75,
    step=0.01
)

TOP_K = st.slider(
    "Top K suggestions",
    min_value=1,
    max_value=10,
    value=3
)

# =====================================================
# SCAN TRANSCRIPT
# =====================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if st.button("Scan Current Transcript"):

    scan_results = []

    for seg_id, segment in enumerate(segments):

        if len(segment.strip()) < 30:
            continue

        query_embedding = model.encode(
            [segment],
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            query_embedding,
            training_embeddings
        )[0]

        top_indices = np.argsort(similarities)[::-1]

        segment_suggestions = []

        seen_relations = set()

        for idx in top_indices[:TOP_K]:

            score = similarities[idx]

            if score < SIMILARITY_THRESHOLD:
                continue

            row = training_df.iloc[idx]

            relation_key = (
                row["source_construct"],
                row["relation_type"],
                row["target_construct"]
            )

            if relation_key in seen_relations:
                continue

            seen_relations.add(relation_key)

            segment_suggestions.append({
                "similarity": round(float(score), 3),
                "source_construct": row[
                    "source_construct"
                ],
                "relation_type": row[
                    "relation_type"
                ],
                "target_construct": row[
                    "target_construct"
                ],
                "matched_text": row[
                    "segment_text"
                ]
            })

        if len(segment_suggestions) > 0:

            scan_results.append({
                "segment_id": seg_id,
                "segment_text": segment,
                "suggestions": segment_suggestions
            })

    st.session_state.scan_results = scan_results

# =================================================
# DISPLAY RESULTS
# =================================================

st.markdown("---")
st.subheader("Candidate Semantic Relations")

scan_results = st.session_state.scan_results

if len(scan_results) == 0:

    st.info("No semantic matches found.")

else:

    for result in scan_results:

        with st.expander(
            f"Segment {result['segment_id']}"
        ):

            st.markdown("### Transcript Fragment")

            st.write(
                result["segment_text"]
            )

            st.markdown("### Suggested Relations")

            for i, s in enumerate(result["suggestions"]):

                st.markdown("---")

                score = s["similarity"]

                if score >= 0.80:
                    confidence = "HIGH"

                elif score >= 0.70:
                    confidence = "MEDIUM"

                else:
                    confidence = "LOW"

                st.write(
                    f"Similarity: {score} "
                    f"({confidence})"
                )

                source_pl = POLISH_LABELS.get(
                    s["source_construct"],
                    s["source_construct"]
                )

                target_pl = POLISH_LABELS.get(
                    s["target_construct"],
                    s["target_construct"]
                )

                relation_pl = POLISH_RELATIONS.get(
                    s["relation_type"],
                    s["relation_type"]
                )

                st.code(
                    f"{source_pl} "
                    f"{relation_pl} "
                    f"{target_pl}"
                )

                st.caption(
                    "Matched training fragment:"
                )

                st.write(
                    s["matched_text"]
                )

                # =========================================
                # APPROVE / REJECT
                # =========================================

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "Approve",
                        key=f"approve_"
                        f"{result['segment_id']}_"
                        f"{i}"
                    ):

                        approved_relation = {

                            "relation_id": str(uuid.uuid4()),

                            "respondent_id":
                                int(selected_file.stem),

                            "transcript_file":
                                selected_file.name,

                            "segment_id":
                                result["segment_id"],

                            "segment_text":
                                result["segment_text"],

                            "source_construct":
                                s["source_construct"],

                            "relation_type":
                                s["relation_type"],

                            "target_construct":
                                s["target_construct"],

                            "intensity":
                                score,

                            "certainty":
                                score,

                            "memo":
                                "AI-assisted approval"
                        }

                        approved_df = pd.DataFrame(
                            [approved_relation]
                        )

                        # =========================================
                        # DUPLICATE CHECK
                        # =========================================

                        already_exists = False

                        if AI_RELATIONS_FILE.exists():

                            existing_df = pd.read_csv(
                                AI_RELATIONS_FILE,
                                encoding="utf-8-sig"
                            )

                            duplicate_mask = (

                                (existing_df["segment_text"]
                                == result["segment_text"])

                                &

                                (existing_df["source_construct"]
                                == s["source_construct"])

                                &

                                (existing_df["relation_type"]
                                == s["relation_type"])

                                &

                                (existing_df["target_construct"]
                                == s["target_construct"])
                            )

                            already_exists = duplicate_mask.any()

                        # =========================================
                        # SAVE
                        # =========================================

                        if already_exists:

                            st.warning(
                                "Relation already saved."
                            )

                        else:

                            if AI_RELATIONS_FILE.exists():

                                approved_df.to_csv(
                                    AI_RELATIONS_FILE,
                                    mode="a",
                                    header=False,
                                    index=False,
                                    encoding="utf-8-sig"
                                )

                            else:

                                approved_df.to_csv(
                                    AI_RELATIONS_FILE,
                                    index=False,
                                    encoding="utf-8-sig"
                                )

                            st.success(
                                "Relation approved and saved."
                            )

                with col2:

                    if st.button(
                        "Reject",
                        key=f"reject_"
                        f"{result['segment_id']}_"
                        f"{i}"
                    ):

                        st.warning(
                            "Relation rejected."
                        )