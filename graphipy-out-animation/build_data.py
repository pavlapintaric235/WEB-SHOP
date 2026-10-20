"""Build graph-data.js for the Web&Shop / InventoryLens graph web app.

Reads ../graphify-out/graph.json (+ labels and report), computes a deterministic
clustered 3D layout (communities placed as blocks, members laid out inside each
block) and writes a compact JS payload the page loads with a plain <script> tag,
so the app works from file:// without a web server.

Run:  python build_data.py        (needs networkx + numpy)
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import re

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "graphify-out"))
OUT = os.path.join(HERE, "graph-data.js")

SEED = 42
COMMUNITY_SPREAD = 620.0


def load_inputs():
    with open(os.path.join(SRC, "graph.json"), encoding="utf-8") as fh:
        graph = json.load(fh)
    with open(os.path.join(SRC, ".graphify_labels.json"), encoding="utf-8") as fh:
        labels = {int(k): v for k, v in json.load(fh).items()}
    cohesion: dict[int, float] = {}
    report_path = os.path.join(SRC, "GRAPH_REPORT.md")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            for m in re.finditer(r'### Community (\d+) - "[^"]*"\s*\nCohesion: ([\d.]+)', fh.read()):
                cohesion[int(m.group(1))] = float(m.group(2))
    return graph, labels, cohesion


def node_kind(n: dict) -> str:
    ft = n.get("file_type")
    if ft in ("concept", "document", "image", "rationale"):
        return ft
    src = n.get("source_file") or ""
    label = n.get("label") or ""
    meta = n.get("metadata") or {}
    if src and label in (os.path.basename(src), src):
        return "file"
    if meta.get("kind") == "file":
        return "file"
    if n.get("_callable_class"):
        return "class"
    if n.get("_callable") or meta.get("kind") in ("bash_function", "bash_entrypoint", "function"):
        return "function"
    if not src:
        return "external"
    return "symbol"


def community_layout(G: nx.Graph, comm_of: dict, sizes: dict) -> dict[int, np.ndarray]:
    """Place each community as a block; blocks that share many links sit close."""
    C = nx.Graph()
    for c in sizes:
        C.add_node(c)
    for u, v in G.edges():
        cu, cv = comm_of[u], comm_of[v]
        if cu == cv:
            continue
        if C.has_edge(cu, cv):
            C[cu][cv]["n"] += 1
        else:
            C.add_edge(cu, cv, n=1)
    for _, _, d in C.edges(data=True):
        d["weight"] = math.log1p(d["n"])
    pos = nx.spring_layout(C, dim=3, seed=SEED, weight="weight", iterations=300, k=0.9)
    centers = {c: np.array(pos[c], dtype=float) * COMMUNITY_SPREAD for c in sizes}
    radius = {c: cluster_radius(sizes[c]) for c in sizes}

    ids = list(sizes)
    for _ in range(400):
        moved = False
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                delta = centers[b] - centers[a]
                dist = float(np.linalg.norm(delta))
                need = radius[a] + radius[b] + 26.0
                if dist < need:
                    if dist < 1e-6:
                        delta = np.random.default_rng(SEED + i).normal(size=3)
                        dist = float(np.linalg.norm(delta))
                    push = (need - dist) / 2.0
                    unit = delta / dist
                    centers[a] = centers[a] - unit * push
                    centers[b] = centers[b] + unit * push
                    moved = True
        for c in ids:
            centers[c] = centers[c] * 0.995
        if not moved:
            break
    return centers


def cluster_radius(size: int) -> float:
    return 8.0 * math.sqrt(size) + 8.0


def member_layout(G: nx.Graph, members: list[str], center: np.ndarray, size: int) -> dict[str, np.ndarray]:
    if len(members) == 1:
        return {members[0]: center.copy()}
    sub = G.subgraph(members)
    scale = cluster_radius(size) * 0.72
    pos = nx.spring_layout(sub, dim=3, seed=SEED, iterations=120, scale=scale)
    return {n: center + np.array(pos[n], dtype=float) for n in members}


def main() -> None:
    graph, labels, cohesion = load_inputs()
    nodes = graph["nodes"]
    links = graph["links"]
    print(f"input: {len(nodes)} nodes, {len(links)} links")

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for lk in links:
        if lk["source"] in G and lk["target"] in G:
            G.add_edge(lk["source"], lk["target"])

    comm_of = {n["id"]: int(n.get("community", -1)) for n in nodes}
    sizes = collections.Counter(comm_of.values())
    degree = dict(G.degree())

    centers = community_layout(G, comm_of, sizes)
    members = collections.defaultdict(list)
    for n in nodes:
        members[comm_of[n["id"]]].append(n["id"])
    pos: dict[str, np.ndarray] = {}
    for c, ids in members.items():
        pos.update(member_layout(G, ids, centers[c], sizes[c]))

    files: list[str] = []
    file_index: dict[str, int] = {}

    def fidx(path: str) -> int:
        if not path:
            return -1
        if path not in file_index:
            file_index[path] = len(files)
            files.append(path)
        return file_index[path]

    index_of = {n["id"]: i for i, n in enumerate(nodes)}
    out_nodes = []
    file_node_by_path: dict[str, int] = {}
    for i, n in enumerate(nodes):
        kind = node_kind(n)
        p = pos[n["id"]]
        rec = {
            "id": n["id"],
            "l": n.get("label") or n["id"],
            "c": comm_of[n["id"]],
            "k": kind,
            "f": fidx(n.get("source_file") or ""),
            "loc": n.get("source_location") or "",
            "x": round(float(p[0]), 1),
            "y": round(float(p[1]), 1),
            "z": round(float(p[2]), 1),
            "d": degree.get(n["id"], 0),
        }
        if n.get("rationale"):
            rec["r"] = n["rationale"]
        if n.get("source_url"):
            rec["u"] = n["source_url"]
        if n.get("_origin"):
            rec["o"] = n["_origin"]
        out_nodes.append(rec)
        if kind == "file" and n.get("source_file"):
            file_node_by_path.setdefault(n["source_file"], i)

    relations: list[str] = []
    rel_index: dict[str, int] = {}
    out_links = []
    for lk in links:
        s, t = index_of.get(lk["source"]), index_of.get(lk["target"])
        if s is None or t is None:
            continue
        rel = lk.get("relation") or "related"
        if rel not in rel_index:
            rel_index[rel] = len(relations)
            relations.append(rel)
        out_links.append([
            s, t, rel_index[rel],
            {"EXTRACTED": 0, "INFERRED": 1, "AMBIGUOUS": 2}.get(lk.get("confidence"), 0),
            round(float(lk.get("confidence_score") or 0), 2),
            fidx(lk.get("source_file") or ""),
            lk.get("source_location") or "",
            lk.get("context") or "",
        ])

    # File-level aggregation for the "overview" detail level.
    agg: dict[tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for lk in links:
        s, t = index_of.get(lk["source"]), index_of.get(lk["target"])
        if s is None or t is None:
            continue
        fs = file_node_by_path.get(nodes[s].get("source_file") or "")
        ft = file_node_by_path.get(nodes[t].get("source_file") or "")
        if fs is None or ft is None or fs == ft:
            continue
        key = (min(fs, ft), max(fs, ft))
        agg[key][lk.get("relation") or "related"] += 1
    out_agg = [[a, b, sum(cnt.values()), dict(cnt.most_common(4))] for (a, b), cnt in sorted(agg.items())]

    communities = []
    for c in sorted(sizes):
        ids = members[c]
        top = sorted(ids, key=lambda i: -degree.get(i, 0))[:8]
        fcount = collections.Counter(nodes[index_of[i]].get("source_file") or "" for i in ids)
        fcount.pop("", None)
        kinds = collections.Counter(out_nodes[index_of[i]]["k"] for i in ids)
        center = centers[c]
        communities.append({
            "id": c,
            "name": labels.get(c, f"Community {c}"),
            "size": sizes[c],
            "cohesion": cohesion.get(c),
            "top": [index_of[i] for i in top],
            "files": [f for f, _ in fcount.most_common(6)],
            "kinds": dict(kinds),
            "cx": round(float(center[0]), 1),
            "cy": round(float(center[1]), 1),
            "cz": round(float(center[2]), 1),
            "r": round(cluster_radius(sizes[c]), 1),
        })

    hyper = []
    for h in graph.get("hyperedges", []):
        idx = [index_of[n] for n in h.get("nodes", []) if n in index_of]
        if idx:
            hyper.append({"id": h["id"], "label": h.get("label", h["id"]), "rel": h.get("relation", ""),
                          "nodes": idx, "file": h.get("source_file", "")})

    god = [index_of[i] for i, _ in sorted(degree.items(), key=lambda kv: -kv[1])[:14]]

    payload = {
        "meta": {
            "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "commit": graph.get("built_at_commit", ""),
            "nodes": len(out_nodes),
            "links": len(out_links),
            "communities": len(communities),
        },
        "files": files,
        "relations": relations,
        "confidence": ["EXTRACTED", "INFERRED", "AMBIGUOUS"],
        "nodes": out_nodes,
        "links": out_links,
        "aggLinks": out_agg,
        "communities": communities,
        "hyperedges": hyper,
        "godNodes": god,
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("// Generated by build_data.py - do not edit by hand.\n")
        fh.write("window.GRAPH_DATA = ")
        fh.write(body)
        fh.write(";\n")
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB): "
          f"{len(out_nodes)} nodes, {len(out_links)} links, {len(out_agg)} file links, "
          f"{len(communities)} communities, {len(files)} files")


if __name__ == "__main__":
    main()
