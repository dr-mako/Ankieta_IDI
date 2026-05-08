# =====================================================
# SEMANTIC RELATION SUGGESTION ENGINE
# semantic_suggester.py
# =====================================================

# INSTALL:
# pip install sentence-transformers scikit-learn pandas

# URUCHOM
# python semantic_suggester.py

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================

RELATIONS_FILE = "output/relational_edges.csv"

SIMILARITY_THRESHOLD = 0.60
TOP_K = 5

# =====================================================
# LOAD DATA
# =====================================================

from pathlib import Path

OUTPUT_DIR = Path("output")

csv_files = list(
    OUTPUT_DIR.glob("*.csv")
)

if len(csv_files) == 0:
    raise Exception(
        "No CSV relation files found in /output"
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
                    f"Loaded: {file.name} "
                    f"with {cfg}"
                )

                break

            except:
                pass

        if temp_df is None:
            raise Exception("Could not decode file")

        print(temp_df.columns)

        all_dfs.append(temp_df)

    except Exception as e:

        print(f"Skipping {file.name}: {e}")

df = pd.concat(
    all_dfs,
    ignore_index=True
)

print(f"\nTotal loaded relations: {len(df)}")


# remove empty rows
df = df.dropna(subset=["segment_text"])
print("After dropna:", len(df))

""""
# remove duplicates
df = df.drop_duplicates(
    subset=[
        "segment_text",
        "source_construct",
        "relation_type",
        "target_construct"
    ]
)
"""



# =====================================================
# LOAD MODEL
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# =====================================================
# BUILD EMBEDDINGS
# =====================================================

texts = df["segment_text"].astype(str).tolist()

print("Building embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Embeddings ready.")
print(f"Loaded relations: {len(df)}")

# =====================================================
# SUGGEST FUNCTION
# =====================================================

def suggest_relations(query_text):

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1]

    suggestions = []

    for idx in top_indices[:TOP_K]:

        score = similarities[idx]

        if score < SIMILARITY_THRESHOLD:
            continue

        row = df.iloc[idx]

        suggestions.append({
            "similarity": round(float(score), 3),
            "matched_text": row["segment_text"],
            "source_construct": row["source_construct"],
            "relation_type": row["relation_type"],
            "target_construct": row["target_construct"],
        })

    return suggestions

# =====================================================
# INTERACTIVE LOOP
# =====================================================

print("\nSemantic Relation Suggestion Engine")
print("Type 'exit' to quit.\n")

while True:

    query = input("Enter text fragment:\n> ")

    if query.lower() == "exit":
        break

    results = suggest_relations(query)

    if len(results) == 0:

        print("\nNo strong semantic matches found.\n")
        continue

    print("\nSuggested relations:\n")

    for i, r in enumerate(results, start=1):

        print("=" * 60)

        print(f"[{i}] Similarity: {r['similarity']}")

        print("\nMatched text:")
        print(r["matched_text"])

        print("\nSuggested relation:")

        print(
            f"{r['source_construct']} "
            f"{r['relation_type']} "
            f"{r['target_construct']}"
        )

        print()

    print("=" * 60)
    print()