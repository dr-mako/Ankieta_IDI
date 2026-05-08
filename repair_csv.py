# repair_csv.py

from pathlib import Path

INPUT_DIR = Path("output")

for file in INPUT_DIR.glob("relational_edges_*.csv"):

    print("Fixing:", file.name)

    with open(file, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.readlines()

    fixed_lines = []

    fixed_lines.append(lines[0])

    for line in lines[1:]:

        line = line.strip()

        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        line = line.replace('""', '"')

        fixed_lines.append(line + "\n")

    with open(file, "w", encoding="utf-8-sig") as f:
        f.writelines(fixed_lines)

print("DONE")