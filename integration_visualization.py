# =====================================================
# PCA-FUZZY SEMANTIC INTEGRATION MAP
# =====================================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

# =====================================================
# LOAD DATA
# =====================================================

with open(
    "semantic_graph.gpickle",
    "rb"
) as f:

    G = pickle.load(f)

integration_df = pd.read_csv(
    "cavi_fuzzy_integration.csv"
)

bridge = pd.read_csv(
    "semantic_bridge.csv"
)

print("\nLoaded integration data.")

# =====================================================
# LOAD PCA LOADINGS
# =====================================================

pca_loadings = pd.read_csv(
    "pca_loadings.csv"
)

pca_loadings = pca_loadings.rename(
    columns={
        "Unnamed: 0": "PC"
    }
)

pca_loadings = pca_loadings.set_index(
    "PC"
)

# =====================================================
# NODE ATTRIBUTES
# =====================================================

pc_color_map = {

    "PC1": "#4C72B0",   # blue
    "PC2": "#C44E52",   # red
    "PC3": "#55A868"    # green
}

node_colors = []
node_sizes = []

# -----------------------------------------------------

integration_map = {}

for _, row in integration_df.iterrows():

    integration_map[
        row["construct"]
    ] = row

# =====================================================
# BUILD VISUAL ATTRIBUTES
# =====================================================

for node in G.nodes():

    if node in integration_map:

        row = integration_map[node]

        # color
        node_colors.append(

            pc_color_map.get(
                row["dominant_pc"],
                "gray"
            )
        )

        # size
        size = (

            row["fuzzy_propagation"]
            * 6000
        ) + 300

        node_sizes.append(size)

    else:

        node_colors.append("lightgray")
        node_sizes.append(500)

# =====================================================
# EDGE ATTRIBUTES
# =====================================================

edge_widths = []

for _, _, d in G.edges(data=True):

    edge_widths.append(

        d["weight"] * 1.5
    )

# =====================================================
# LAYOUT
# =====================================================

pos = nx.spring_layout(

    G,

    seed=42,

    k=3
)

# =====================================================
# FIGURE
# =====================================================

plt.figure(
    figsize=(20, 18)
)

# -----------------------------------------------------
# EDGES
# -----------------------------------------------------

nx.draw_networkx_edges(

    G,
    pos,

    width=edge_widths,

    alpha=0.12,

    edge_color="gray"
)

# -----------------------------------------------------
# NODES
# -----------------------------------------------------

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    edgecolors="black",
    linewidths=0.8,
    alpha=0.9
)

# =====================================================
# TOP NODES
# =====================================================

top_nodes = integration_df[
    integration_df["fuzzy_propagation"] >= 0.5
]["construct"].tolist()

# =====================================================
# LABELS
# =====================================================

for node, (x, y) in pos.items():

    if node in top_nodes:
        fs = 20
        alpha = 1.0

    else:
        fs = 10
        alpha = 0.75

    plt.text(
        x,
        y,
        node,
        fontsize=fs,
        alpha=alpha,
        ha="center",
        va="center"
    )

# =====================================================
# LEGEND
# =====================================================

from matplotlib.lines import Line2D

legend_elements = [

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='PC1',

        markerfacecolor=pc_color_map["PC1"],

        markersize=15
    ),

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='PC2',

        markerfacecolor=pc_color_map["PC2"],

        markersize=15
    ),

    Line2D(
        [0],
        [0],

        marker='o',

        color='w',

        label='PC3',

        markerfacecolor=pc_color_map["PC3"],

        markersize=15
    )
]

plt.legend(

    handles=legend_elements,

    title="Dominant PCA Component",

    fontsize=12,

    title_fontsize=13,

    loc="upper left"
)

# =====================================================
# TITLE
# =====================================================

plt.title(

    "Integrated PCA–Fuzzy Semantic Propagation System",

    fontsize=24
)

plt.axis("off")

plt.tight_layout()

# =====================================================
# SAVE
# =====================================================

plt.savefig(

    "pca_fuzzy_integration_map.png",

    dpi=600,

    bbox_inches="tight"
)

plt.savefig(

    "pca_fuzzy_integration_map.svg",

    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nSaved: pca_fuzzy_integration_map.png"
)

print(
    "Saved: pca_fuzzy_integration_map.svg"
)

# =====================================================
# PCA–FUZZY DIVERGENCE ANALYSIS
# =====================================================

print("\nCalculating PCA–Fuzzy divergence...")

# =====================================================
# REAL PCA IMPORTANCE
# =====================================================

real_pca_importance = []

for _, row in integration_df.iterrows():

    construct = row["construct"]

    # znajdź odpowiadającą zmienną
    bridge_row = bridge[
        bridge["semantic_construct"]
        == construct
    ]

    if len(bridge_row) == 0:

        real_pca_importance.append(
            np.nan
        )

        continue

    survey_var = bridge_row[
        "survey_variable"
    ].iloc[0]

    matched_col = None

    survey_words = set(
        survey_var.lower().split()
    )

    best_overlap = 0

    for col in pca_loadings.columns:

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

        real_pca_importance.append(
            np.nan
        )

        continue

    # ---------------------------------------------
    # REAL LOADINGS
    # ---------------------------------------------

    pc1 = abs(
        pca_loadings.loc[
            "PC1",
            matched_col
        ]
    )

    pc2 = abs(
        pca_loadings.loc[
            "PC2",
            matched_col
        ]
    )

    pc3 = abs(
        pca_loadings.loc[
            "PC3",
            matched_col
        ]
    )

    latent_strength = max(
        pc1,
        pc2,
        pc3
    )

    real_pca_importance.append(
        latent_strength
    )

# =====================================================
# SAVE
# =====================================================

integration_df[
    "pca_importance"
] = real_pca_importance

# =====================================================
# NORMALIZATION
# =====================================================

integration_df["pca_importance_norm"] = (
    integration_df["pca_importance"]
    /
    integration_df["pca_importance"].max()
)

integration_df["fuzzy_propagation_norm"] = (
    integration_df["fuzzy_propagation"]
    /
    integration_df["fuzzy_propagation"].max()
)

# =====================================================
# DIVERGENCE
# =====================================================

integration_df["divergence"] = (
    integration_df["fuzzy_propagation_norm"]
    -
    integration_df["pca_importance_norm"]
)

# =====================================================
# INTERPRETIVE ROLE
# =====================================================

def divergence_role(x):

    if x >= 0.25:
        return "systemic_amplifier"

    elif x <= -0.25:
        return "respondent_sensitive"

    else:
        return "balanced"

integration_df["divergence_role"] = (
    integration_df["divergence"]
    .apply(divergence_role)
)

# =====================================================
# RESULTS
# =====================================================

divergence_df = integration_df[[
    "construct",
    "pca_importance_norm",
    "fuzzy_propagation_norm",
    "divergence",
    "divergence_role"
]]

divergence_df = divergence_df.sort_values(
    "divergence",
    ascending=False
)

print("\n=== PCA–FUZZY DIVERGENCE ===")
print(divergence_df.round(4))

# =====================================================
# SAVE
# =====================================================

divergence_df.to_csv(
    "pca_fuzzy_divergence.csv",
    index=False
)

print("\nSaved: pca_fuzzy_divergence.csv")