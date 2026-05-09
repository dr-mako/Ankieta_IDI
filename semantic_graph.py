# =====================================================
# SEMANTIC RELATION GRAPH BUILDER
# semantic_graph.py
# =====================================================

# INSTALL
# pip install pandas networkx matplotlib pyvis

# URUCHOM
# python semantic_graph.py

# =====================================================
# IMPORTS
# =====================================================

from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pyvis.network import Network

# =====================================================
# CONFIG
# =====================================================

OUTPUT_DIR = Path("output")

GRAPH_HTML = "semantic_network.html"

# usuwa słabe semantycznie relacje
MIN_EDGE_WEIGHT = 0.85
# usuwa żadkie relacje
MIN_FREQUENCY = 5

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
        "target_construct"
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
# usuwa żadkie relacje
edge_df = edge_df[
    edge_df["frequency"]
    >= MIN_FREQUENCY
] 
# usuwa słabe semantycznie relacje
edge_df = edge_df[
    edge_df["mean_intensity"]
    >= MIN_EDGE_WEIGHT
]

print(
    f"Filtered edges: {len(edge_df)}"
)

# =====================================================
# CREATE GRAPH
# =====================================================

G = nx.DiGraph()

# =====================================================
# ADD EDGES
# =====================================================

for _, row in edge_df.iterrows():

    source = row["source"]

    target = row["target"]

    relation = row["relation"]

    weight = row["mean_intensity"]

    frequency = row["frequency"]

    G.add_edge(

        source,
        target,

        relation=relation,

        weight=weight,

        frequency=frequency
    )

# =====================================================
# BASIC NETWORK STATS
# =====================================================

print("\n=== NETWORK STATS ===")

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

# =====================================================
# CENTRALITY
# =====================================================

degree_centrality = nx.degree_centrality(G)

print("\n=== TOP CENTRAL NODES ===")

top_nodes = sorted(

    degree_centrality.items(),

    key=lambda x: x[1],

    reverse=True

)[:10]

for node, score in top_nodes:

    print(
        f"{node}: {round(score, 3)}"
    )

# =====================================================
# EXPORT CENTRALITY TABLE
# =====================================================

centrality_df = pd.DataFrame({

    "construct":
        list(degree_centrality.keys()),

    "degree_centrality":
        list(degree_centrality.values())
})

centrality_df = centrality_df.sort_values(

    by="degree_centrality",

    ascending=False
)

centrality_df.to_csv(

    "centrality_results.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: centrality_results.csv"
)

# =====================================================
# BETWEENNESS CENTRALITY
# =====================================================

betweenness = nx.betweenness_centrality(

    G,

    weight="weight"
)

print("\n=== TOP BETWEENNESS NODES ===")

top_between = sorted(

    betweenness.items(),

    key=lambda x: x[1],

    reverse=True

)[:10]

for node, score in top_between:

    print(
        f"{node}: {round(score,3)}"
    )

# =====================================================
# EXPORT BETWEENNESS
# =====================================================

between_df = pd.DataFrame({

    "construct":
        list(betweenness.keys()),

    "betweenness":
        list(betweenness.values())
})

between_df = between_df.sort_values(

    by="betweenness",

    ascending=False
)

between_df.to_csv(

    "betweenness_results.csv",

    index=False,

    encoding="utf-8-sig"
)

print(
    "\nSaved: betweenness_results.csv"
)

# =====================================================
# STATIC MATPLOTLIB GRAPH
# =====================================================

# canvas
plt.figure(
    figsize=(26, 18)
)

pos = nx.spring_layout(
    G,
    k=6.0, # Większe rozepchnięcie grafu
    iterations=1000, # Więcej iteracji stabilizacji
    seed=42 
)

edge_widths = [

    0.5 + G[u][v]["weight"] * 1.2

    for u, v in G.edges()
]

node_sizes = []

degree_centrality = nx.degree_centrality(G)

for node in G.nodes():

    size = min(
        800 + degree_centrality[node] * 9000,
        5000
    )

    node_sizes.append(size)

for node, (x, y) in pos.items():

    plt.text(
        x,
        y + 0.05, # Etykiety wyjdą poza nody
        node,
        fontsize=10, # Mniejsza czcionka
        ha='center'
    )

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color="#8db3e2",
    edgecolors="#5b8cc9",
    linewidths=1.5
)

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

nx.draw_networkx_edges(
    G,
    pos,
    width=edge_widths,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=20
)

edge_labels = {

    (u, v):

    (
        f"{d['relation']}\n"
        f"w={round(d['weight'],2)}\n"
        f"n={d['frequency']}"
    )

    for u, v, d
    in G.edges(data=True)
}


#nx.draw_networkx_edge_labels(
#    G,
#    pos,
#    edge_labels=edge_labels,
#    font_size=7
#)


plt.title(
    "Semantic Relation Network"
)

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

    fontsize=10
)

plt.axis("off")

plt.tight_layout()


plt.savefig(
    "semantic_network.svg",
    bbox_inches="tight"
)

plt.savefig(
    "semantic_network.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

plt.close()

# =====================================================
# INTERACTIVE HTML GRAPH
# =====================================================

net = Network(

    height="1400px",
    width="100%",

    directed=True,

    bgcolor="#ffffff",

    font_color="black"
)

net.barnes_hut()

net.set_options("""

var options = {

  "nodes": {

    "font": {
      "size": 40,
      "face": "arial"
    },

    "scaling": {
      "label": {
        "enabled": false
      }
    }

  },

  "physics": {
    "barnesHut": {
        "gravitationalConstant": -12000,
        "centralGravity": 0.15,
        "springLength": 350,
        "springConstant": 0.02,
        "damping": 0.95
    },
    "minVelocity": 0.75
  }

}

""")

# =====================================================
# ADD NODES
# =====================================================

for node in G.nodes():

    degree = G.degree(node)

    net.add_node(

        node,

        label=node,

        title=node,

        size=15 + degree * 2,

        font={
            "size": 60 # font html
        },

        borderWidth=2,
    )

# =====================================================
# ADD EDGES
# =====================================================

for u, v, d in G.edges(data=True):

    #if d["weight"] < 0.75:
    #    continue

    if d["weight"] < MIN_EDGE_WEIGHT:
        continue

    relation = d["relation"]

    if relation == "amplifies":
        color = "green"

    elif relation == "suppresses":
        color = "red"

    elif relation == "conflicts_with":
        color = "black"

    elif relation == "depends_on":
        color = "blue"

    elif relation == "enables":
        color = "orange"

    else:
        color = "gray"

    title = (

        f"Relation: {relation}<br>"
        f"Weight: {round(d['weight'], 2)}<br>"
        f"Frequency: {d['frequency']}"
    )

    net.add_edge(

        u,
        v,

        title=title,

        value=d["weight"] * 5,

        color=color
    )

# =====================================================
# SAVE HTML
# =====================================================

GRAPH_HTML = "semantic_network.html"

net.write_html(
    GRAPH_HTML,
    open_browser=False
)

with open(
    GRAPH_HTML,
    "r",
    encoding="utf-8"
) as f:

    html = f.read()

legend_html = """

<div style="
position:absolute;
top:10px;
left:10px;
background:white;
padding:10px;
border:1px solid black;
z-index:9999;
font-size:14px;
">

<b>Relation Types</b><br>

<span style="color:green;">■</span> amplifies<br>
<span style="color:red;">■</span> suppresses<br>
<span style="color:black;">■</span> conflicts_with<br>
<span style="color:blue;">■</span> depends_on<br>
<span style="color:orange;">■</span> enables<br>
<span style="color:gray;">■</span> other

</div>

"""

html = html.replace(
    "<body>",
    f"<body>{legend_html}"
)

with open(
    GRAPH_HTML,
    "w",
    encoding="utf-8"
) as f:

    f.write(html)

print(
    "\nInteractive graph saved:"
)

print(GRAPH_HTML)

# =====================================================
# GROUP DEFINITIONS
# =====================================================

GROUPS = {

    "Technological": [

        "autonomous_operation",
        "technological_maturity",
        "technological_trust",
        "technology_standardization",
        "data_management",
        "system_reliability"
    ],

    "Infrastructure": [

        "supporting_infrastructure",
        "infrastructure_readiness",
        "infrastructure_optimization",
        "signal_visibility",
        "highway_test_zones",
        "urban_test_zones",
        "cross_border_interoperability"
    ],

    "Operational": [

        "operational_efficiency",
        "traffic_efficiency",
        "continuous_transport",
        "workforce_efficiency",
        "mobility_improvement"
    ],

    "Economic": [

        "implementation_cost",
        "investment_requirements",
        "labor_costs",
        "economic_stability",
        "business_profitability"
    ],

    "Social": [

        "social_acceptance",
        "business_acceptance",
        "social_image",
        "consumer_trust"
    ],

    "Safety": [

        "consumer_safety",
        "accident_reduction",
        "health_protection",
        "deployment_risk"
    ],

    "Environmental": [

        "environmental_impact",
        "environmental_sustainability",
        "fuel_consumption",
        "energy_demand",
        "energy_source",
        "energy_reduction"
    ],

    "Regulatory": [

        "regulatory_governance",
        "regulatory_instability",
        "regulatory_procedures",
        "training_requirements",
        "research_support"
    ]
}

# =====================================================
# CONSTRUCT -> GROUP MAP
# =====================================================

construct_to_group = {}

for group, constructs in GROUPS.items():

    for c in constructs:

        construct_to_group[c] = group

# =====================================================
# GROUP-LEVEL EDGE TABLE
# =====================================================

group_edges = []

for _, row in edge_df.iterrows():

    source_construct = row["source"]

    target_construct = row["target"]

    source_group = construct_to_group.get(
        source_construct
    )

    target_group = construct_to_group.get(
        target_construct
    )

    # skip unknown constructs
    if source_group is None:
        continue

    if target_group is None:
        continue

    # skip self-loops
    if source_group == target_group:
        continue

    group_edges.append({

        "source_group":
            source_group,

        "target_group":
            target_group,

        "relation":
            row["relation"],

        "weight":
            row["mean_intensity"],

        "frequency":
            row["frequency"]
    })

# =====================================================
# GROUP EDGE DATAFRAME
# =====================================================

group_df = pd.DataFrame(group_edges)

# =====================================================
# AGGREGATE GROUP RELATIONS
# =====================================================

group_summary = (

    group_df.groupby(
        [
            "source_group",
            "target_group"
        ]
    )

    .agg({

        "weight": "mean",

        "frequency": "sum"
    })

    .reset_index()

)

print("\n=== GROUP RELATIONS ===")

print(group_summary)

# =====================================================
# GROUP-LEVEL GRAPH
# =====================================================

G_group = nx.DiGraph()

# =====================================================
# ADD GROUP EDGES
# =====================================================

for _, row in group_summary.iterrows():

    G_group.add_edge(

        row["source_group"],
        row["target_group"],

        weight=row["weight"],

        frequency=row["frequency"]
    )

# =====================================================
# GROUP-LEVEL GRAPH
# =====================================================

G_group = nx.DiGraph()

# =====================================================
# ADD GROUP EDGES
# =====================================================

for _, row in group_summary.iterrows():

    G_group.add_edge(

        row["source_group"],
        row["target_group"],

        weight=row["weight"],

        frequency=row["frequency"]
    )

# =====================================================
# DRAW GROUP GRAPH
# =====================================================

plt.figure(figsize=(14, 10))
# ustawia pozycje nodów na rys 
pos = nx.spring_layout(
    G_group,
    k=2.5,
    seed=42
)

# node sizes
node_sizes = []

for node in G_group.nodes():

    degree = G_group.degree(node)

    node_sizes.append(
        2500 + degree * 600
    )

# edge widths
edge_widths = []

for u, v in G_group.edges():

    freq = G_group[u][v]["frequency"]

    edge_widths.append(
       0.5 + freq * 0.08
    )

# nodes
nx.draw_networkx_nodes(

    G_group,
    pos,

    node_size=node_sizes,

    node_color="#8db3e2",

    edgecolors="#4a6fa5",

    linewidths=2
)

# edges
nx.draw_networkx_edges(

    G_group,
    pos,

    width=edge_widths,

    arrows=True,

    arrowsize=25,

    alpha=0.8
)

# labels
nx.draw_networkx_labels(

    G_group,
    pos,

    font_size=14,

    font_weight="bold"
)

plt.title(
    "Group-Level Semantic System",
    fontsize=18
)

plt.axis("off")

plt.tight_layout()

plt.title(
    "Group-Level Semantic System",
    fontsize=26,
    pad=25
)

plt.savefig(
    "group_level_system.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "group_level_system.svg",
    bbox_inches="tight"
)

plt.show()

plt.close()

