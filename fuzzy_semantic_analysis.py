# python fuzzy_semantic_analysis.py

# =====================================================
# FUZZY SEMANTIC ANALYSIS
# =====================================================

import networkx as nx
import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pickle

# =====================================================
# LOAD GRAPH
# =====================================================


GRAPH_FILE = "semantic_graph.gpickle"

with open(
    GRAPH_FILE,
    "rb"
) as f:

    G = pickle.load(f)

print("\nLoaded graph.")

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# =====================================================
# FUZZY WEIGHT CATEGORIES
# =====================================================

def fuzzy_category(weight):

    if weight < 0.30:
        return "weak"

    elif weight < 0.60:
        return "moderate"

    elif weight < 0.85:
        return "strong"

    else:
        return "dominant"
    
# =====================================================
# FUZZY EDGE TABLE
# =====================================================

fuzzy_edges = []

for u, v, d in G.edges(data=True):

    raw_weight = d.get(
        "weight",
        0
    )

    max_weight = max(

        data["weight"]

        for _, _, data
        in G.edges(data=True)
    )

    weight = raw_weight / max_weight

    relation = d.get(
        "relation",
        "other"
    )

    fuzzy_edges.append({

        "source": u,
        "target": v,

        "weight": round(weight, 4),

        "relation": relation,

        "fuzzy_category":
            fuzzy_category(weight)
    })

fuzzy_df = pd.DataFrame(
    fuzzy_edges
)

fuzzy_df.to_csv(

    "fuzzy_edges.csv",

    index=False
)

print(
    "\nSaved: fuzzy_edges.csv"
)

# =====================================================
# FUZZY DISTRIBUTION
# =====================================================

print("\n=== FUZZY RELATION DISTRIBUTION ===")

print(

    fuzzy_df[
        "fuzzy_category"
    ].value_counts()
)

# =====================================================
# FUZZY INFLUENCE SCORE
# =====================================================

print("\nCalculating fuzzy influence scores...")

# -----------------------------------------------------
# CENTRALITIES
# -----------------------------------------------------

weighted_degree = {}

for node in G.nodes():

    in_strength = sum(

        G[u][v]["weight"]

        for u, v in G.in_edges(node)
    )

    out_strength = sum(

        G[u][v]["weight"]

        for u, v in G.out_edges(node)
    )

    weighted_degree[node] = (

        in_strength + out_strength
    )

pagerank = nx.pagerank(

    G,

    weight="weight"
)

eigenvector = nx.eigenvector_centrality(

    G,

    max_iter=500,

    weight="weight"
)

betweenness = nx.betweenness_centrality(

    G,

    weight="weight"
)

# =====================================================
# NORMALIZATION
# =====================================================

def normalize_dict(d):

    values = np.array(
        list(d.values())
    )

    min_val = values.min()

    max_val = values.max()

    norm = {}

    for k, v in d.items():

        if max_val == min_val:

            norm[k] = 0

        else:

            norm[k] = (

                (v - min_val)

                /

                (max_val - min_val)
            )

    return norm

weighted_degree_n = normalize_dict(
    weighted_degree
)

pagerank_n = normalize_dict(
    pagerank
)

eigenvector_n = normalize_dict(
    eigenvector
)

betweenness_n = normalize_dict(
    betweenness
)

# =====================================================
# FUZZY INFLUENCE
# =====================================================

fuzzy_scores = []

for node in G.nodes():

    influence = (

        weighted_degree_n[node]

        +

        pagerank_n[node]

        +

        eigenvector_n[node]

        +

        betweenness_n[node]

    ) / 4

    fuzzy_scores.append({

        "construct": node,

        "fuzzy_influence":
            round(influence, 4)
    })

fuzzy_influence_df = pd.DataFrame(
    fuzzy_scores
)

fuzzy_influence_df = fuzzy_influence_df.sort_values(

    by="fuzzy_influence",

    ascending=False
)

# =====================================================
# FUZZY CATEGORY
# =====================================================

def fuzzy_node_category(score):

    if score < 0.25:
        return "peripheral"

    elif score < 0.50:
        return "supporting"

    elif score < 0.75:
        return "influential"

    else:
        return "dominant"

fuzzy_influence_df[
    "fuzzy_role"
] = fuzzy_influence_df[
    "fuzzy_influence"
].apply(
    fuzzy_node_category
)

# =====================================================
# SAVE
# =====================================================

fuzzy_influence_df.to_csv(

    "fuzzy_influence_scores.csv",

    index=False
)

print(
    "\nSaved: fuzzy_influence_scores.csv"
)

# =====================================================
# TOP NODES
# =====================================================

print("\n=== TOP FUZZY INFLUENCE NODES ===")

print(
    fuzzy_influence_df.head(15)
)

# =====================================================
# FUZZY PROPAGATION
# =====================================================

print("\nCalculating fuzzy propagation...")

# -----------------------------------------------------
# FUZZY SCORE MAP
# -----------------------------------------------------

fuzzy_map = dict(

    zip(

        fuzzy_influence_df["construct"],

        fuzzy_influence_df["fuzzy_influence"]
    )
)

# =====================================================
# PROPAGATION SCORE
# =====================================================

propagation_results = []

for node in G.nodes():

    total_propagation = 0

    outgoing = G.out_edges(
        node,
        data=True
    )

    for _, target, d in outgoing:

        edge_weight = d["weight"]

        target_influence = fuzzy_map[
            target
        ]

        propagated = (

            edge_weight

            *

            target_influence
        )

        total_propagation += propagated

    propagation_results.append({

        "construct": node,

        "fuzzy_propagation":
            round(total_propagation, 4)
    })

# =====================================================
# DATAFRAME
# =====================================================

propagation_df = pd.DataFrame(
    propagation_results
)

# -----------------------------------------------------
# NORMALIZE
# -----------------------------------------------------

max_prop = propagation_df[
    "fuzzy_propagation"
].max()

propagation_df[
    "fuzzy_propagation"
] = (

    propagation_df[
        "fuzzy_propagation"
    ]

    /

    max_prop
)

propagation_df[
    "fuzzy_propagation"
] = propagation_df[
    "fuzzy_propagation"
].round(4)

# =====================================================
# PROPAGATION CATEGORY
# =====================================================

def propagation_category(score):

    if score < 0.25:
        return "localized"

    elif score < 0.50:
        return "distributed"

    elif score < 0.75:
        return "systemic"

    else:
        return "dominant"

propagation_df[
    "propagation_role"
] = propagation_df[
    "fuzzy_propagation"
].apply(
    propagation_category
)

# =====================================================
# SORT
# =====================================================

propagation_df = propagation_df.sort_values(

    by="fuzzy_propagation",

    ascending=False
)

# =====================================================
# SAVE
# =====================================================

propagation_df.to_csv(

    "fuzzy_propagation.csv",

    index=False
)

print(
    "\nSaved: fuzzy_propagation.csv"
)

# =====================================================
# PRINT
# =====================================================

print("\n=== TOP FUZZY PROPAGATION ===")

print(
    propagation_df.head(15)
)