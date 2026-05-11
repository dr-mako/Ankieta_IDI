#
# weighted_network_analysis.py
#
# RUN
# python step_4_weighted_network_analysis.py
#
# semantic relational systems analysis
#
# Etapy analizy
# 1. Weighted Degree, czyli: node strength
#    suma wag relacji.
#
# 2. Weighted Betweenness, czyli weighted paths.
# 
# 3. Eigenvector Centrality
#    Pokaże: wpływowe konstrukty połączone z wpływowymi konstruktami.
#
# 4. PageRank
#    system influence,
#    propagation potential.
# 
# 5. Closeness Centrality, czyli Które konstrukty są najbliżej całego systemu.
# 
# 6. Density, czyli Jak gęsty jest system.
# 
# 7. Reciprocity, czyli czy relacje są:
#    jednostronne,
#    wzajemne.
#
# 8. Community Detection
#    Louvain,
#    modularity.
#
# 9. Edge importance, czyli Które relacje spinają system.
#
# 10. Fuzzy Interpretation Layer, czyli edges mają uncertainty,
#     weights, 
#     frequencies.
#
# ================================================================
# 
# OUTPUT
#
# ===============================================================
#
# CSV:
#      weighted_centrality.csv
#      pagerank.csv
#      communities.csv
#
# FIGURES:
#      centrality ranking,
#      subsystem map,
#      weighted influence graph,
#      heatmap,
#      adjacency matrix.
#
# ============================================================
#
# Nastęny krok:
#
# fuzzy socio-technical semantic network.


# =====================================================
# weighted_network_analysis.py
# =====================================================

# INSTALL
# pip install pandas networkx matplotlib

# =====================================================
# IMPORTS
# =====================================================

from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

OUTPUT_DIR = Path("output")

MIN_FREQUENCY = 3
MIN_INTENSITY = 0.75

# =====================================================
# LOAD CSV FILES
# =====================================================

csv_files = list(
    OUTPUT_DIR.glob("*.csv")
)

if len(csv_files) == 0:

    raise Exception(
        "No CSV files found in /output"
    )

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

                print(
                    f"Loaded: {file.name}"
                )

                break

            except:
                pass

        if temp_df is None:
            continue

        all_dfs.append(temp_df)

    except Exception as e:

        print(
            f"Skipping {file.name}: {e}"
        )

# =====================================================
# CONCAT
# =====================================================

df = pd.concat(
    all_dfs,
    ignore_index=True
)

print(
    f"\nLoaded relations: {len(df)}"
)

# =====================================================
# CLEAN
# =====================================================

df = df.dropna(
    subset=[
        "source_construct",
        "relation_type",
        "target_construct",
        "intensity"
    ]
)

# =====================================================
# BUILD EDGE TABLE
# =====================================================

edge_df = (

    df.groupby(
        [
            "source_construct",
            "relation_type",
            "target_construct"
        ]
    )

    .agg({

        "intensity": "mean",

        "relation_id": "count"

    })

    .reset_index()

)

edge_df.columns = [

    "source",
    "relation",
    "target",
    "mean_intensity",
    "frequency"
]

# =====================================================
# FILTER
# =====================================================

edge_df = edge_df[
    edge_df["frequency"]
    >= MIN_FREQUENCY
]

edge_df = edge_df[
    edge_df["mean_intensity"]
    >= MIN_INTENSITY
]

# =====================================================
# HYBRID EDGE WEIGHT
# =====================================================

edge_df["edge_weight"] = (

    edge_df["mean_intensity"]

    *

    edge_df["frequency"]
)

print(
    f"\nFiltered edges: {len(edge_df)}"
)

# =====================================================
# BUILD GRAPH
# =====================================================

G = nx.DiGraph()

for _, row in edge_df.iterrows():

    G.add_edge(

        row["source"],
        row["target"],

        relation=row["relation"],

        weight=row["edge_weight"],

        intensity=row["mean_intensity"],

        frequency=row["frequency"]
    )

# =====================================================
# NETWORK STATS
# =====================================================

print("\n=== NETWORK STATS ===")

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# =====================================================
# WEIGHTED DEGREE
# =====================================================

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

weighted_degree_df = pd.DataFrame({

    "construct":
        list(weighted_degree.keys()),

    "weighted_degree":
        list(weighted_degree.values())
})

weighted_degree_df = weighted_degree_df.sort_values(

    by="weighted_degree",

    ascending=False
)

weighted_degree_df.to_csv(

    "weighted_degree.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: weighted_degree.csv"
)

print("\n=== TOP WEIGHTED NODES ===")

print(
    weighted_degree_df.head(10)
)

# =====================================================
# WEIGHTED BETWEENNESS
# =====================================================

betweenness = nx.betweenness_centrality(

    G,

    weight="weight"
)

betweenness_df = pd.DataFrame({

    "construct":
        list(betweenness.keys()),

    "weighted_betweenness":
        list(betweenness.values())
})

betweenness_df = betweenness_df.sort_values(

    by="weighted_betweenness",

    ascending=False
)

betweenness_df.to_csv(

    "weighted_betweenness.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: weighted_betweenness.csv"
)

# =====================================================
# EIGENVECTOR CENTRALITY
# =====================================================

components = list(
    nx.weakly_connected_components(G)
)

print(
    f"Weakly connected components: {len(components)}"
)

eigenvector = nx.eigenvector_centrality(

    G,

    max_iter=500,

    weight="weight"
)

eigenvector_df = pd.DataFrame({

    "construct":
        list(eigenvector.keys()),

    "eigenvector_centrality":
        list(eigenvector.values())
})

eigenvector_df = eigenvector_df.sort_values(

    by="eigenvector_centrality",

    ascending=False
)

eigenvector_df.to_csv(

    "eigenvector_centrality.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: eigenvector_centrality.csv"
)

# =====================================================
# PAGERANK
# =====================================================

pagerank = nx.pagerank(

    G,

    weight="weight"
)

pagerank_df = pd.DataFrame({

    "construct":
        list(pagerank.keys()),

    "pagerank":
        list(pagerank.values())
})

pagerank_df = pagerank_df.sort_values(

    by="pagerank",

    ascending=False
)

pagerank_df.to_csv(

    "pagerank.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: pagerank.csv"
)

# =====================================================
# CLOSENESS CENTRALITY
# =====================================================

closeness = nx.closeness_centrality(G)

closeness_df = pd.DataFrame({

    "construct":
        list(closeness.keys()),

    "closeness":
        list(closeness.values())
})

closeness_df = closeness_df.sort_values(

    by="closeness",

    ascending=False
)

closeness_df.to_csv(

    "closeness.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: closeness.csv"
)

# =====================================================
# NETWORK DENSITY
# =====================================================

density = nx.density(G)

print("\n=== NETWORK DENSITY ===")

print(round(density, 4))

# =====================================================
# RECIPROCITY
# =====================================================

reciprocity = nx.reciprocity(G)

print("\n=== NETWORK RECIPROCITY ===")

print(round(reciprocity, 4))

# =====================================================
# CENTRALITY FIGURE
# =====================================================

top10 = weighted_degree_df.head(10)

plt.figure(figsize=(12, 7))

plt.barh(

    top10["construct"],

    top10["weighted_degree"]
)

plt.gca().invert_yaxis()

plt.title(
    "Top Weighted Constructs",
    fontsize=18
)

plt.xlabel(
    "Weighted Degree",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "weighted_degree_ranking.svg",
    bbox_inches="tight"
)

plt.savefig(
    "weighted_degree_ranking.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nAnalysis complete."
)

# =====================================================
# WEIGHTED INFLUENCE GRAPH
# =====================================================

plt.figure(
    figsize=(24, 18)
)

pos = nx.spring_layout(

    G,

    k=10,

    iterations=500,

    seed=42
)

# =====================================================
# NODE SIZES
# =====================================================

node_sizes = []

for node in G.nodes():

    size = (

        1200

        +

        weighted_degree[node] * 18
    )

    node_sizes.append(size)

# =====================================================
# EDGE WIDTHS
# =====================================================

edge_widths = []

for u, v in G.edges():

    w = G[u][v]["weight"]

    edge_widths.append(

        0.5 + w * 0.18
    )

# =====================================================
# EDGE COLORS
# =====================================================

edge_colors = []

for u, v in G.edges():

    relation = G[u][v]["relation"]

    if relation == "amplifies":
        edge_colors.append("green")

    elif relation == "suppresses":
        edge_colors.append("red")

    elif relation == "conflicts_with":
        edge_colors.append("black")

    elif relation == "depends_on":
        edge_colors.append("blue")

    elif relation == "enables":
        edge_colors.append("orange")

    else:
        edge_colors.append("gray")

# =====================================================
# DRAW NODES
# =====================================================

nx.draw_networkx_nodes(

    G,
    pos,

    node_size=node_sizes,

    alpha=0.9
)

# =====================================================
# DRAW EDGES
# =====================================================

nx.draw_networkx_edges(

    G,
    pos,

    edge_color=edge_colors,

    width=edge_widths,

    arrows=True,

    arrowsize=18,

    alpha=0.75
)

# =====================================================
# LABELS
# =====================================================

nx.draw_networkx_labels(

    G,
    pos,

    font_size=18
)

# =====================================================
# TITLE
# =====================================================

plt.title(

    "Weighted Semantic Influence Network",

    fontsize=24
)

# =====================================================
# LEGEND
# =====================================================

from matplotlib.lines import Line2D

legend_elements = [

    Line2D(
        [0],
        [0],
        color='green',
        lw=3,
        label='amplifies'
    ),

    Line2D(
        [0],
        [0],
        color='red',
        lw=3,
        label='suppresses'
    ),

    Line2D(
        [0],
        [0],
        color='black',
        lw=3,
        label='conflicts_with'
    ),

    Line2D(
        [0],
        [0],
        color='blue',
        lw=3,
        label='depends_on'
    ),

    Line2D(
        [0],
        [0],
        color='orange',
        lw=3,
        label='enables'
    ),

    Line2D(
        [0],
        [0],
        color='gray',
        lw=3,
        label='other'
    )
]

plt.legend(

    handles=legend_elements,

    loc='upper left',

    fontsize=12
)

# =====================================================
# SAVE
# =====================================================

plt.axis("off")

plt.tight_layout()

plt.savefig(

    "weighted_influence_graph.svg",

    bbox_inches="tight"
)

plt.savefig(

    "weighted_influence_graph.png",

    dpi=600,

    bbox_inches="tight"
)

plt.show()

plt.close()

# =====================================================
# TOP INFLUENCE SUBGRAPH
# =====================================================

TOP_NODES = 12

top_constructs = list(

    weighted_degree_df
    .head(TOP_NODES)["construct"]
)

# -----------------------------------------------------
# SUBGRAPH
# -----------------------------------------------------

H = G.subgraph(
    top_constructs
).copy()

print("\n=== TOP INFLUENCE SUBGRAPH ===")

print(
    f"Nodes: {H.number_of_nodes()}"
)

print(
    f"Edges: {H.number_of_edges()}"
)

# -----------------------------------------------------
# FIGURE
# -----------------------------------------------------

plt.figure(
    figsize=(18, 12)
)

pos = nx.kamada_kawai_layout(H)

#pos = nx.shell_layout(H)

pos = nx.circular_layout(H)

# -----------------------------------------------------
# NODE SIZES
# -----------------------------------------------------

sub_sizes = []

for node in H.nodes():

    score = weighted_degree_df[
        weighted_degree_df["construct"] == node
    ]["weighted_degree"].values[0]

    sub_sizes.append(
        1200 + score * 15
    )

# -----------------------------------------------------
# EDGE COLORS
# -----------------------------------------------------

edge_colors = []

for u, v in H.edges():

    relation = H[u][v]["relation"]

    if relation == "amplifies":
        edge_colors.append("green")

    elif relation == "suppresses":
        edge_colors.append("red")

    elif relation == "conflicts_with":
        edge_colors.append("black")

    elif relation == "depends_on":
        edge_colors.append("blue")

    elif relation == "enables":
        edge_colors.append("orange")

    else:
        edge_colors.append("gray")

# -----------------------------------------------------
# EDGE WIDTHS
# -----------------------------------------------------

edge_widths = []

for u, v in H.edges():

    freq = H[u][v]["frequency"]

    width = 0.8 + (freq / 10)

    edge_widths.append(width)

# -----------------------------------------------------
# DRAW
# -----------------------------------------------------

nx.draw_networkx_nodes(

    H,
    pos,

    node_size=sub_sizes,

    node_color="#8fb3e0",

    edgecolors="#4a6fa5",

    linewidths=2
)

nx.draw_networkx_edges(

    H,
    pos,

    edge_color=edge_colors,

    width=edge_widths,

    arrows=True,

    arrowsize=16,

    alpha=0.45,

    connectionstyle='arc3,rad=0.08'

)

nx.draw_networkx_labels(

    H,
    pos,

    font_size=12,

    font_weight="bold"
)

# -----------------------------------------------------
# LEGEND
# -----------------------------------------------------

from matplotlib.lines import Line2D

legend_elements = [

    Line2D(
        [0],
        [0],
        color='green',
        lw=3,
        label='amplifies'
    ),

    Line2D(
        [0],
        [0],
        color='red',
        lw=3,
        label='suppresses'
    ),

    Line2D(
        [0],
        [0],
        color='black',
        lw=3,
        label='conflicts_with'
    ),

    Line2D(
        [0],
        [0],
        color='blue',
        lw=3,
        label='depends_on'
    ),

    Line2D(
        [0],
        [0],
        color='orange',
        lw=3,
        label='enables'
    ),

    Line2D(
        [0],
        [0],
        color='gray',
        lw=3,
        label='other'
    )
]

plt.legend(

    handles=legend_elements,

    loc='upper left',

    fontsize=11
)

# -----------------------------------------------------
# FINAL
# -----------------------------------------------------

plt.title(

    "Core Semantic Influence Structure",

    fontsize=24
)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    "top_influence_subgraph.png",

    dpi=600,

    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nSaved: top_influence_subgraph.png"
)

# =====================================================
# WEIGHTED ADJACENCY HEATMAP
# =====================================================

print("\nCreating adjacency heatmap...")

# -----------------------------------------------------
# ADJACENCY MATRIX
# -----------------------------------------------------

adj_matrix = nx.to_pandas_adjacency(

    G,

    weight="weight"
)


LABEL_MAP = {

    "autonomous_operation": "AUTO_OP",
    "business_acceptance": "BUS_ACC",
    "operational_efficiency": "OPER_EFF",
    "traffic_efficiency": "TRAF_EFF",
}

short_columns = [

    LABEL_MAP.get(c, c)

    for c in adj_matrix.columns
]

# -----------------------------------------------------
# OPTIONAL:
# sort by weighted degree
# -----------------------------------------------------

sorted_nodes = list(

    weighted_degree_df["construct"]
)

adj_matrix = adj_matrix.loc[
    sorted_nodes,
    sorted_nodes
]

# -----------------------------------------------------
# FIGURE
# -----------------------------------------------------

plt.figure(
    figsize=(24, 20)
)

plt.imshow(

    adj_matrix,

    interpolation="nearest",

    aspect="auto"
)

# -----------------------------------------------------
# AXES
# -----------------------------------------------------


plt.xticks(

    range(len(adj_matrix.columns)),

    adj_matrix.columns,

    rotation=90,

    fontsize=8
)

plt.yticks(

    range(len(adj_matrix.index)),

    adj_matrix.index,

    fontsize=8
)

# -----------------------------------------------------
# COLORBAR
# -----------------------------------------------------

cbar = plt.colorbar()

cbar.set_label(

    "Semantic Relation Weight",

    fontsize=12
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

plt.title(

    "Weighted Semantic Adjacency Matrix",

    fontsize=22
)

plt.tight_layout()

# -----------------------------------------------------
# SAVE
# -----------------------------------------------------

plt.savefig(

    "semantic_heatmap.png",

    dpi=600,

    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nSaved: semantic_heatmap.png"
)

#=====================================================
#          COMMUNITY DETECTION
#=====================================================
#
print("\nRunning community detection...")

import community as community_louvain

# -----------------------------------------------------
# LOUVAIN NEEDS UNDIRECTED GRAPH
# -----------------------------------------------------

G_undirected = G.to_undirected()

# -----------------------------------------------------
# COMMUNITY DETECTION
# -----------------------------------------------------

partition = community_louvain.best_partition(

    G_undirected,

    weight="weight",

    random_state=42
)

# -----------------------------------------------------
# SAVE COMMUNITIES
# -----------------------------------------------------

community_df = pd.DataFrame({

    "construct": list(partition.keys()),

    "community": list(partition.values())
})

community_df = community_df.sort_values(

    by="community"
)

community_df.to_csv(

    "communities.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: communities.csv"
)

# -----------------------------------------------------
# PRINT COMMUNITIES
# -----------------------------------------------------

print("\n=== COMMUNITIES ===")

for comm_id in sorted(

    community_df["community"].unique()
):

    members = community_df[
        community_df["community"] == comm_id
    ]["construct"].tolist()

    print(f"\nCommunity {comm_id}:")

    for m in members:

        print(f"  - {m}")

# =====================================================
# COMMUNITY GRAPH
# =====================================================

print("\nCreating community graph...")

# -----------------------------------------------------
# COLORS
# -----------------------------------------------------

community_colors = []

for node in G.nodes():

    comm = partition[node]

    community_colors.append(comm)

# -----------------------------------------------------
# FIGURE
# -----------------------------------------------------

plt.figure(
    figsize=(18, 14)
)

pos = nx.spring_layout(

    G,

    k=7,

    iterations=300,

    seed=42
)

# -----------------------------------------------------
# NODE SIZES
# -----------------------------------------------------

node_sizes = []

for node in G.nodes():

    size = weighted_degree_df[
        weighted_degree_df["construct"] == node
    ]["weighted_degree"].values[0]

    node_sizes.append(size * 12)

# -----------------------------------------------------
# EDGES
# -----------------------------------------------------

strong_edges = [

    (u, v)

    for u, v, d in G.edges(data=True)

    if d["weight"] >= 0.95
]

edge_widths = [

    G[u][v]["weight"] * 0.9

    for u, v in strong_edges
]

nx.draw_networkx_edges(

    G,

    pos,

    edgelist=strong_edges,

    width=edge_widths,

    alpha=0.18,

    edge_color="gray",

    arrows=False
)

# -----------------------------------------------------
# NODES
# -----------------------------------------------------

nx.draw_networkx_nodes(

    G,

    pos,

    node_size=node_sizes,

    node_color=community_colors,

    cmap=plt.cm.Set3,

    alpha=0.9
)

# -----------------------------------------------------
# LABELS
# -----------------------------------------------------

nx.draw_networkx_labels(

    G,

    pos,

    font_size=14,

    font_weight="bold"
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

plt.title(

    "Semantic Communities (Louvain Detection)",

    fontsize=24
)

plt.axis("off")

plt.tight_layout()

# -----------------------------------------------------
# SAVE
# -----------------------------------------------------

plt.savefig(

    "semantic_communities.svg",

    bbox_inches="tight"
)

plt.savefig(

    "semantic_communities.png",

    dpi=600,

    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nSaved: semantic_communities.png"
)

# =====================================================
# REORDERED HEATMAP
# =====================================================

print("\nCreating reordered heatmap...")

# -----------------------------------------------------
# SORT BY COMMUNITY
# -----------------------------------------------------

ordered_nodes = community_df[
    "construct"
].tolist()

community_adj = adj_matrix.loc[
    ordered_nodes,
    ordered_nodes
]

# -----------------------------------------------------
# FIGURE
# -----------------------------------------------------

plt.figure(
    figsize=(22, 18)
)

plt.imshow(

    community_adj,

    interpolation="nearest",

    aspect="auto",

    cmap="magma_r"
)

# -----------------------------------------------------
# AXES
# -----------------------------------------------------

plt.xticks(

    range(len(community_adj.columns)),

    community_adj.columns,

    rotation=90,

    fontsize=7
)

plt.yticks(

    range(len(community_adj.index)),

    community_adj.index,

    fontsize=7
)

# -----------------------------------------------------
# COLORBAR
# -----------------------------------------------------

cbar = plt.colorbar()

cbar.set_label(

    "Semantic Relation Weight",

    fontsize=12
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

plt.title(

    "Community-Structured Semantic Heatmap",

    fontsize=22
)

plt.tight_layout()

# -----------------------------------------------------
# SAVE
# -----------------------------------------------------


plt.savefig(

    "community_heatmap.png",

    dpi=600,

    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "\nSaved: community_heatmap.png"
)

# =====================================================
# SAVE GRAPH
# =====================================================

import pickle

with open(
    "semantic_graph.gpickle",
    "wb"
) as f:

    pickle.dump(
        G,
        f
    )

print(
    "\nSaved: semantic_graph.gpickle"
)