# Step 1
# 
# semantic_coder.py — V2 (Excel integrated)
#++++++++++++++++++++++++++++++++++++++++++
# URUCHOMIENIE
# streamlit run semantic_coder.py
#+++++++++++++++++++++++++++++++++++++++++
import json
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

import os

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "dataverse_files"
TRANSCRIPTS_DIR = DATA_DIR / "transkrypcje"
EXCEL_FILE = DATA_DIR / "baza_IDI.xlsx"

ONTOLOGY_DIR = BASE_DIR / "ontology"
OUTPUT_DIR = BASE_DIR / "output"

ONTOLOGY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

#RELATIONS_FILE = ONTOLOGY_DIR / "relations.json"
ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.csv"
EDGES_FILE = OUTPUT_DIR / "relational_edges.csv"

# =====================================================
# DEFAULT RELATIONS
# =====================================================

DEFAULT_RELATIONS = [
    "amplifies",
    "suppresses",
    "conditions",
    "depends_on",
    "stabilizes",
    "conflicts_with",
    "enables",
    "reduces",    
]


# =====================================================
# CONSTRUCT NAME MAPPING
# =====================================================

DISPLAY_NAMES = {

    # BENEFITS
    "Optymalizacja_kosztow": "operational_efficiency",
    "Zmiana_wydajnosci_pracy": "workforce_efficiency",
    "Zmniejszenie_zuzycia_energii": "energy_reduction",
    "redukcja_zanieczyszczen_halasu": "environmental_impact",
    "poprawa_spolecznego_wizerunku_przedsiębiorstwa": "social_image",
    "skrocenie_czasu": "traffic_efficiency",
    "koszty_wypadków": "accident_reduction",
    "energy_source": "energy_source",

    # USER BENEFITS
    "ochrona_zdrowia": "health_protection",
    "koszty_serwisowania": "maintenance_costs",
    "zużycie_paliwa": "fuel_consumption",
    "parkowanie_pojazdów": "parking_efficiency",
    "poprawa_mobilności": "mobility_improvement",
    "infrastruktura_drogowa": "road_infrastructure",
    "wolny_czas": "free_time",
    "klimat_środowisko": "environmental_sustainability",

    # BARRIERS
    "zaufanie_konsumentów": "technological_trust",
    "niedojrzałość_technologii": "technological_maturity",
    "koszt_wdrożenia": "implementation_cost",
    "zapotrzebowanie_na_energię": "energy_demand",
    "ochrona_wrażliwych_danych": "data_protection",
    "cyberprzestępczość": "cybersecurity",
    "badania_i_rozwój": "research_support",
    "regulacje_odp": "regulatory_governance",
    "procedury_regulacyjne": "regulatory_procedures",
    "zarządzanie_danymi": "data_management",

    # IMPLEMENTATION
    "dodatkowe_środki_na_inwestycje": "investment_requirements",
    "rozwój_infrastruktury": "infrastructure_readiness",
    "ochrona_ubezpieczeniowa": "insurance_protection",
    "szkolenia": "training_requirements",
    "bezpieczeństwo_konsumentow": "consumer_safety",
    "marketing": "market_adaptation",
    "standardy_etyczne": "ethical_governance",

    # INFRASTRUCTURE
    "obszary_testowe_autostrady": "highway_test_zones",
    "obszary_testowe_miasta": "urban_test_zones",
    "czytelność": "signal_visibility",
    "standaryzacja_technologii": "technology_standardization",
    "optymalizacji_infrastruktury": "infrastructure_optimization",
    "infrastruktura_wspomagająca": "supporting_infrastructure",

    # ORGANIZATIONAL
    "Logistyka": "logistics_sector",
    "Spedycja": "freight_forwarding",
    "Przewóz_towarów": "cargo_transport",
    "Przewóz_osób": "passenger_transport",
}



# =====================================================
# LOAD EXCEL
# =====================================================

if not EXCEL_FILE.exists():
    st.error(f"Missing Excel file: {EXCEL_FILE}")
    st.stop()

excel_df = pd.read_excel(EXCEL_FILE)

# respondent_id = numer wiersza + 1
excel_df = excel_df.reset_index(drop=True)
excel_df["respondent_id"] = excel_df.index + 1

# =====================================================
# LOAD TRANSCRIPTS
# =====================================================

transcript_files = sorted(
    list(TRANSCRIPTS_DIR.glob("*.txt")) +
    list(TRANSCRIPTS_DIR.glob("*.text"))
)

if len(transcript_files) == 0:
    st.error("No transcript files found.")
    st.stop()

# =====================================================
# HELPERS
# =====================================================

def segment_text(text):
    """
    Prosta segmentacja:
    dzielenie po pustych liniach.
    """

    raw_segments = text.split("\n\n")

    segments = []

    for seg in raw_segments:
        seg = seg.strip()

        if len(seg) > 30:
            segments.append(seg)

    return segments


def save_annotation(data):

    df = pd.DataFrame([data])

    if ANNOTATIONS_FILE.exists():
        df.to_csv(
            ANNOTATIONS_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )

    else:
        df.to_csv(
            ANNOTATIONS_FILE,
            index=False,
            encoding="utf-8",
        )


def save_edge(data):

    df = pd.DataFrame([data])

    if EDGES_FILE.exists():
        df.to_csv(
            EDGES_FILE,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )

    else:
        df.to_csv(
            EDGES_FILE,
            index=False,
            encoding="utf-8",
        )


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(layout="wide")

st.title("Relational Interaction Coding Framework")
st.subheader("Semantically controlled fuzzy-ready ontology")

# =====================================================
# FILE SELECTION
# =====================================================

selected_file = st.selectbox(
    "Select transcript",
    transcript_files,
    format_func=lambda x: x.name,
)

# respondent id = nazwa pliku
# np. 8.txt -> respondent_id = 8

try:
    respondent_id = int(selected_file.stem)

except:
    st.error(
        "Filename must be numeric, e.g. 8.txt"
    )
    st.stop()

import chardet

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

segments = segment_text(transcript_text)

if len(segments) == 0:
    st.warning("No segments found.")
    st.stop()

# =====================================================
# MATCH RESPONDENT
# =====================================================

respondent_row = None

if respondent_id in excel_df["respondent_id"].values:

    respondent_row = excel_df[
        excel_df["respondent_id"] == respondent_id
    ].iloc[0]

# =====================================================
# METADATA
# =====================================================

st.markdown("---")
st.subheader("Respondent Metadata")

if respondent_row is not None:

    col1, col2, col3 = st.columns(3)

    with col1:

        if "Płeć" in respondent_row.index:
            st.write(f"**Płeć:** {respondent_row['Płeć']}")

        if "Wiek" in respondent_row.index:
            st.write(f"**Wiek:** {respondent_row['Wiek']}")

    with col2:

        if "Stanowisko" in respondent_row.index:
            st.write(f"**Stanowisko:** {respondent_row['Stanowisko']}")

        if "Branża" in respondent_row.index:
            st.write(f"**Branża:** {respondent_row['Branża']}")

    with col3:

        if "Rynek" in respondent_row.index:
            st.write(f"**Rynek:** {respondent_row['Rynek']}")

else:

    st.warning(
        f"No matching respondent in Excel for respondent_id={respondent_id}"
    )

# =====================================================
# ACTIVE CONSTRUCTS
# =====================================================

st.markdown("---")
st.subheader("Active Constructs From Excel")

excluded_cols = [
    "respondent_id",
    "Płeć",
    "Wiek",
    "Wykształcenie",
    "Stanowisko",
    "Organizacja",
    "Stanowisko_aktualne",
    "Rynek",
    "Branża",
    "Inne_tekst",
]

active_constructs = []
MANUAL_CONSTRUCTS = [
    "energy_source",
    "deployment_risk",
    "urban_density",
    "regulatory_instability",
    "cross_border_interoperability",
    "autonomous_operation",
    "continuous_transport",
    "labor_costs",
    "negative_media_exposure",
    "social_acceptance",
    "traffic_efficiency",
    "business_acceptance",
    "economic_stability",
    "data_management",
]

if respondent_row is not None:

    for col in respondent_row.index:

        if col in excluded_cols:
            continue

        try:

            value = float(respondent_row[col])

            if value == 1.0:
                mapped_name = DISPLAY_NAMES.get(col, col)
                active_constructs.append(mapped_name)

        except:
            pass

# add manual ontology extensions

for mc in MANUAL_CONSTRUCTS:

    if mc not in active_constructs:
        active_constructs.append(mc)


if len(active_constructs) == 0:

    st.warning("No active constructs found.")

else:

    st.dataframe(
        pd.DataFrame(
            active_constructs,
            columns=["active_constructs"]
        )
    )

# =====================================================
# SEGMENT NAVIGATION
# =====================================================

st.markdown("---")
st.subheader("Transcript Segments")

segment_index = st.number_input(
    "Segment index",
    min_value=0,
    max_value=len(segments) - 1,
    value=0,
)

current_segment = segments[segment_index]

st.text_area(
    "Current Segment",
    value=current_segment,
    height=300,
)

# =====================================================
# SELECTED TEXT CODING PANEL
# DODAJ TO PRZED "RELATIONAL CODING"
# =====================================================

st.markdown("---")
st.subheader("Selected Text Fragment")

selected_text = st.text_area(
    "Copy fragment from transcript and paste here:",
    height=200,
    placeholder="Paste only the relevant fragment...",
    key="selected_text"
)

# =====================================================
# CONSTRUCT LIST
# =====================================================

construct_list = sorted(list(set(

    list(DISPLAY_NAMES.values()) +

    [
        "autonomous_operation",
        "continuous_transport",
        "labor_costs",
        "social_acceptance",
        "negative_media_exposure",
        "deployment_risk",
        "urban_density",
        "regulatory_instability",
        "cross_border_interoperability",
        "business_acceptance",
        "economic_stability",
        "data_management"
    ]

)))



# =====================================================
# LOAD RELATIONS DATAFRAME
# =====================================================

if os.path.exists(EDGES_FILE):

    relations_df = pd.read_csv(EDGES_FILE)

else:

    relations_df = pd.DataFrame(columns=[
        "relation_id",
        "respondent_id",
        "transcript_file",
        "segment_id",
        "segment_text",
        "source_construct",
        "relation_type",
        "target_construct",
        "intensity",
        "certainty",
        "memo"
    ])

# =====================================================
# RELATIONAL CODING
# =====================================================

st.markdown("---")
st.subheader("Relational Coding")

source_construct = st.selectbox(
    "Source construct",
    construct_list
)

RELATION_TYPES = [
    "amplifies",
    "suppresses",
    "enables",
    "reduces",
    "conditions",
    "depends_on",
    "stabilizes",
    "conflicts_with",
    "constrains"
]

relation_type = st.selectbox(
    "Relation",
    RELATION_TYPES
)

target_construct = st.selectbox(
    "Target construct",
    construct_list,
    index=min(1, len(construct_list)-1)
)

intensity = st.slider(
    "Intensity",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05
)

certainty = st.slider(
    "Certainty",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)

memo = st.text_area(
    "Analytical memo",
    height=100
)

# =====================================================
# SAVE RELATION
# =====================================================

if st.button("Save Relation"):

    if not selected_text.strip():
        st.error("Please paste a text fragment first.")

    elif source_construct == target_construct:
        st.error("Source and target constructs must differ.")

    else:

        relation_record = {
            "relation_id": str(uuid.uuid4()),
            "respondent_id": respondent_id,
            "transcript_file": selected_file.name,
            "segment_id": 0,
            "segment_text": selected_text,
            "source_construct": source_construct,
            "relation_type": relation_type,
            "target_construct": target_construct,
            "intensity": intensity,
            "certainty": certainty,
            "memo": memo
        }

        relations_df = pd.concat(
            [relations_df, pd.DataFrame([relation_record])],
            ignore_index=True
        )

        relations_df.to_csv(
            EDGES_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        st.success("Relation saved successfully.")
        st.session_state["selected_text"] = ""

# =====================================================
# RECENT RELATIONS PREVIEW
# =====================================================

st.markdown("---")
st.subheader("Recent Relations")

if not relations_df.empty:

    required_cols = [
        "source_construct",
        "relation_type",
        "target_construct",
        "intensity",
        "certainty"
    ]

    existing_cols = [
        col for col in required_cols
        if col in relations_df.columns
    ]

    if len(existing_cols) == len(required_cols):

        preview_df = relations_df[required_cols].tail(10)

        st.dataframe(preview_df, use_container_width=True)

    else:

        st.info("Relations table initialized.")

else:

    st.info("No relations coded yet.")

    
