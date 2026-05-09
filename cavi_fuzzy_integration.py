
# cavi_fuzzy_integration.py
# automatic bridge resolver

import pandas as pd

bridge = pd.read_csv("semantic_bridge.csv")

print(bridge.head(20))

loadings = pd.read_csv("pca_loadings.csv")

print(loadings.columns.tolist())

# =====================================================
# LOAD FILES
# =====================================================

import pandas as pd
import numpy as np

# -----------------------------------------------------

bridge = pd.read_csv(
    "semantic_bridge.csv"
)

loadings = pd.read_csv(
    "pca_loadings.csv"
)

fuzzy_prop = pd.read_csv(
    "fuzzy_propagation.csv"
)

fuzzy_inf = pd.read_csv(
    "fuzzy_influence_scores.csv"
)

# =====================================================
# CLEAN PCA LOADINGS
# =====================================================

loadings = loadings.rename(
    columns={
        "Unnamed: 0": "PC"
    }
)

loadings = loadings.set_index(
    "PC"
)

# =====================================================
# INTEGRATION
# =====================================================

integration_results = []

for _, row in bridge.iterrows():

    construct = row[
        "semantic_construct"
    ]

    survey_var = row[
        "survey_variable"
    ]

    # -------------------------------------------------
    # FIND MATCHING PCA COLUMN
    # -------------------------------------------------

    matched_col = None

    survey_words = set(
        survey_var.lower().split()
    )

    best_overlap = 0

    for col in loadings.columns:

        col_words = set(
            col.lower().split()
        )

        overlap = len(
            survey_words.intersection(
                col_words
            )
        )

        if overlap > best_overlap:

            best_overlap = overlap

            matched_col = col

    if matched_col is None:
        continue

    # -------------------------------------------------
    # PCA LOADINGS
    # -------------------------------------------------

    pc1 = loadings.loc[
        "PC1",
        matched_col
    ]

    pc2 = loadings.loc[
        "PC2",
        matched_col
    ]

    pc3 = loadings.loc[
        "PC3",
        matched_col
    ]

    # dominant PC
    abs_loadings = {

        "PC1": abs(pc1),
        "PC2": abs(pc2),
        "PC3": abs(pc3)
    }

    dominant_pc = max(
        abs_loadings,
        key=abs_loadings.get
    )

    dominant_loading = abs_loadings[
        dominant_pc
    ]

    # -------------------------------------------------
    # FUZZY INFLUENCE
    # -------------------------------------------------

    inf_row = fuzzy_inf[
        fuzzy_inf["construct"]
        == construct
    ]

    prop_row = fuzzy_prop[
        fuzzy_prop["construct"]
        == construct
    ]

    if len(inf_row) == 0:
        continue

    fuzzy_influence = float(

        inf_row[
            "fuzzy_influence"
        ].iloc[0]
    )

    fuzzy_role = inf_row[
        "fuzzy_role"
    ].iloc[0]

    fuzzy_propagation = float(

        prop_row[
            "fuzzy_propagation"
        ].iloc[0]
    )

    propagation_role = prop_row[
        "propagation_role"
    ].iloc[0]

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    integration_results.append({

        "construct":
            construct,

        "dominant_pc":
            dominant_pc,

        "loading_strength":
            round(
                dominant_loading,
                4
            ),

        "fuzzy_influence":
            round(
                fuzzy_influence,
                4
            ),

        "fuzzy_role":
            fuzzy_role,

        "fuzzy_propagation":
            round(
                fuzzy_propagation,
                4
            ),

        "propagation_role":
            propagation_role
    })

# =====================================================
# FINAL TABLE
# =====================================================

integration_df = pd.DataFrame(
    integration_results
)

integration_df = integration_df.sort_values(

    by="fuzzy_propagation",

    ascending=False
)

# =====================================================
# SAVE
# =====================================================

integration_df.to_csv(

    "cavi_fuzzy_integration.csv",

    index=False
)

print(
    "\nSaved: cavi_fuzzy_integration.csv"
)

# =====================================================
# PRINT
# =====================================================

print("\n=== INTEGRATED SEMANTIC LATENT SYSTEM ===")

print(
    integration_df
)