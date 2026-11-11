"""
Synthetic works TSV in the OpenAlex-pre column layout, plus embeddings split
over two files, to smoke-test the whole pipeline (see README).

    python make_toy_data.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else "toy")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

CONCEPTS = {"CS": 41008148, "Math": 33923547, "Physics": 121332964, "Biology": 86803240,
            "Sociology": 144024400, "Economics": 162324750, "Linguistics": 204787440}
NAMES = list(CONCEPTS)
OTHER = 185592680          # a concept outside the seven (Chemistry)
N_AUTH = 25000
home = rng.choice(len(NAMES), N_AUTH, p=[0.35, 0.15, 0.15, 0.15, 0.08, 0.07, 0.05])
start = rng.integers(1988, 2015, N_AUTH)
in_field = [{h} for h in home]
header = ["id", "title", "type", "publication_year", "publication_date", "doi",
          "primary_location:source:id", "primary_location:source:type", "is_retracted",
          "is_paratext", "authorships:author_position", "authorships:author:id",
          "authorships:institutions:id", "corresponding_author_ids", "corresponding_institution_ids",
          "ids:doi", "ids:mag", "ids:pmcid", "ids:pmid", "grants:funder", "grants:award_id",
          "concepts:id", "concepts:score", "referenced_works", "mesh:is_major_topic",
          "mesh:descriptor_ui", "mesh:descriptor_name", "mesh:qualifier_ui", "mesh:qualifier_name"]
pids, pfields = [], []
with open(out / "works.tsv", "w") as f:
    f.write("\t".join(header) + "\n")
    pid = 4_000_000_000
    for year in range(1988, 2024):
        alive = np.where(start <= year)[0]
        for _ in range(2500):
            pid += 1
            lead = rng.choice(alive)
            team = {int(lead)}
            same = alive[home[alive] == home[lead]]
            while len(team) < rng.integers(2, 7):
                team.add(int(rng.choice(same if rng.random() < 0.92 else alive)))
            fields = {home[lead]}
            if rng.random() < 0.08:
                fields.add(int(rng.choice(sorted(set().union(*[in_field[a] for a in team])))))
            if rng.random() < 0.02:
                fields.add(int(rng.integers(len(NAMES))))
            for a in team:
                in_field[a] |= fields
            cids = [CONCEPTS[NAMES[x]] for x in fields] + [OTHER]
            scores = [round(float(rng.uniform(0.35, 0.9)), 4) for _ in fields] + [0.5]
            if rng.random() < 0.1:                     # below the score threshold: dropped
                scores = [0.1] * len(cids)
            refs = rng.choice(pids, size=min(len(pids), 8), replace=False) if pids else []
            typ = "article" if rng.random() > 0.05 else "book-chapter"
            row = [""] * len(header)
            row[0], row[2], row[3] = str(pid), typ, str(year)
            row[11] = "[" + ",".join(str(5_000_000_000 + a) for a in team) + "]"
            row[21] = "[" + ",".join(map(str, cids)) + "]"
            row[22] = "[" + ",".join(map(str, scores)) + "]"
            row[23] = "[" + ",".join(map(str, refs)) + "]"
            f.write("\t".join(row) + "\n")
            pids.append(pid)
            pfields.append(fields)

# embeddings: field-dependent, two overlapping files, ids in "W123" form
fmap = {x: rng.normal(size=32) * 3 for x in range(len(NAMES))}
E = rng.normal(size=(len(pids), 32)).astype("float32")
for i, fs in enumerate(pfields):
    E[i] += sum(fmap[x] for x in fs)
half = len(pids) // 2
for k, (lo, hi) in enumerate([(0, half + 1000), (half, len(pids))]):
    np.save(out / f"emb_{k}.npy", E[lo:hi])
    with open(out / f"emb_{k}_ids.txt", "w") as fh:
        fh.writelines(f"W{p}\n" for p in pids[lo:hi])
print(f"{len(pids):,} works written to {out}")
