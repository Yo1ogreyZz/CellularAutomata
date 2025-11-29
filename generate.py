#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate figures for "Rules as Graphs" section:
  - fig/rule30_dBG.pdf
  - fig/rule110_dBG.pdf

Requirements:
  - networkx
  - matplotlib
"""

import os
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc


# ----------------------------------------------------------------------
# 1. Utilities: ECA rule → de Bruijn graph (k=2, r=1)
# ----------------------------------------------------------------------

def build_eca_debruijn(rule_number: int) -> nx.DiGraph:
    """
    Build the de Bruijn graph for an elementary cellular automaton
    (k=2, r=1) with given rule_number in [0, 255].

    Nodes: "00", "01", "10", "11"
    Edges: each length-3 neighbourhood abc, from ab -> bc
           with edge label f(abc) ∈ {0,1}.
    """
    if not (0 <= rule_number <= 255):
        raise ValueError("rule_number must be in [0, 255] for ECA.")

    G = nx.DiGraph()

    # All length-2 binary strings (nodes)
    nodes = [f"{i:02b}" for i in range(4)]  # 00, 01, 10, 11
    G.add_nodes_from(nodes)

    # ECA standard: bitstring from 111,110,...,000 (MSB→LSB)
    patterns = ["111", "110", "101", "100", "011", "010", "001", "000"]

    # rule_number in binary, 8 bits, MSB→LSB (e.g. 30 → "00011110")
    bitstring = f"{rule_number:08b}"

    # Map pattern → output bit
    rule_map = {pat: int(bitstring[i]) for i, pat in enumerate(patterns)}

    # Build edges
    for pat in patterns:
        a, b, c = pat
        u = a + b
        v = b + c
        out = rule_map[pat]
        G.add_edge(u, v, label=str(out))  # store label as string "0"/"1"

    return G


# ----------------------------------------------------------------------
# 2. Drawing: consistent layout for all ECA dBGs
# ----------------------------------------------------------------------

def get_fixed_positions_eca_dbg():
    """
    Fixed positions for the four nodes of an ECA de Bruijn graph.
    更大间距的 2x2 格子，便于看清边和自环。
    """
    pos = {
        "00": (0.0, 0.0),
        "01": (2.5, 0.0),
        "10": (0.0, -2.5),
        "11": (2.5, -2.5),
    }
    return pos


def draw_eca_debruijn(G: nx.DiGraph, rule_number: int, filename: str):
    """
    Draw an ECA de Bruijn graph and save as a PDF.

    使用手工绘制节点 / 边 / 自环，避免文字被挡、自环不清的问题。

    Parameters
    ----------
    G : nx.DiGraph
        Graph returned by build_eca_debruijn.
    rule_number : int
        ECA rule number (for title, if needed).
    filename : str
        Path to save the PDF, e.g. "fig/rule30_dBG.pdf".
    """
    pos = get_fixed_positions_eca_dbg()

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    # -------- 画节点（圆 + 节点标签） --------
    node_radius = 0.35
    for node, (x, y) in pos.items():
        circle = plt.Circle(
            (x, y),
            node_radius,
            edgecolor="black",
            facecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )

    # -------- 画边（含自环）和边标签 --------
    for u, v, data in G.edges(data=True):
        label = data.get("label", "")

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if u == v:
            # ---------- 自环：画在节点上方 ----------
            loop_radius = 0.8  # 自环半径
            # 弧线主体
            arc = Arc(
                (x1, y1 + node_radius),         # 中心略偏上
                width=loop_radius,
                height=loop_radius,
                angle=0,
                theta1=40,                      # 起止角度决定形状
                theta2=320,
                lw=1.2,
                zorder=1,
            )
            ax.add_patch(arc)

            # 箭头：沿弧线方向大致放一个箭头
            ax.annotate(
                "",
                xy=(x1 + 0.35 * 0.8, y1 + node_radius + 0.35 * 0.4),
                xytext=(x1 + 0.35 * 0.2, y1 + node_radius + 0.35 * 0.9),
                arrowprops=dict(arrowstyle="-|>", lw=1.2),
                zorder=2,
            )

            # 自环标签：放在节点上方一点
            label_x, label_y = x1 + 0.6, y1 + 1.0

        else:
            # ---------- 普通有向边 ----------
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=15,   # 箭头尺寸
                lw=1.2,
                zorder=1,
                connectionstyle="arc3,rad=0.0",  # 直线；如想弯曲可调 rad
            )
            ax.add_patch(arrow)

            # 边标签：取中点 + 法向偏移，避免压在边上
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length != 0:
                offset = 0.20
                ox, oy = -dy / length * offset, dx / length * offset
            else:
                ox = oy = 0.0
            label_x, label_y = mx + ox, my + oy

        # 画边标签，加一个白底小圆角框，防止被线挡住
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
            zorder=5,
        )

    # -------- 其他外观设置 --------
    ax.set_xlim(-1.0, 3.5)
    ax.set_ylim(-3.5, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # 可留一个小 title 用于 debug；论文里也可以删掉
    ax.set_title(f"ECA Rule {rule_number} de Bruijn Graph", fontsize=11)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


# ----------------------------------------------------------------------
# 3. Main: generate all figures used in the LaTeX section
# ----------------------------------------------------------------------

def main():
    # Rule 30
    G30 = build_eca_debruijn(30)
    draw_eca_debruijn(G30, 30, "fig/rule30_dBG.pdf")

    # Rule 110
    G110 = build_eca_debruijn(110)
    draw_eca_debruijn(G110, 110, "fig/rule110_dBG.pdf")


# ----------------------------------------------------------------------
# 4. (Optional) skeletons for pair/subset graph construction
# ----------------------------------------------------------------------

def build_pair_graph_from_dbg(G: nx.DiGraph) -> nx.DiGraph:
    """
    OPTIONAL: build pair graph from a de Bruijn graph.
    Nodes: (u, v) pairs of dBG nodes.
    Edge: (u, v) -> (u', v') if there are edges u->u', v->v' in G
          with the same edge label.
    """
    PG = nx.DiGraph()

    # Pre-collect outgoing edges grouped by label
    outgoing_by_label = {}
    for u, v, data in G.edges(data=True):
        lab = data.get("label")
        outgoing_by_label.setdefault((u, lab), []).append(v)

    # Build nodes & edges
    nodes = list(G.nodes())
    labels = {d["label"] for _, _, d in G.edges(data=True)}

    for u in nodes:
        for v in nodes:
            PG.add_node((u, v))

    for u in nodes:
        for v in nodes:
            for lab in labels:
                outs_u = outgoing_by_label.get((u, lab), [])
                outs_v = outgoing_by_label.get((v, lab), [])
                for u_next in outs_u:
                    for v_next in outs_v:
                        PG.add_edge((u, v), (u_next, v_next), label=lab)

    return PG


def build_subset_graph_from_dbg(G: nx.DiGraph, max_subset_size: int = 2) -> nx.DiGraph:
    """
    OPTIONAL: *small* subset graph for illustrative figures.

    Full power-set graph is huge; for figures, you usually restrict to
    subsets up to some small size (e.g. 1 or 2) or to a chosen collection
    of subsets.

    Here we only build subsets of size <= max_subset_size.
    """
    import itertools

    SG = nx.DiGraph()
    nodes = list(G.nodes())
    labels = {d["label"] for _, _, d in G.edges(data=True)}

    # Generate small subsets
    subsets = []
    for size in range(1, max_subset_size + 1):
        subsets.extend(itertools.combinations(nodes, size))

    # Represent subsets as sorted tuples
    subsets = [tuple(sorted(s)) for s in subsets]
    SG.add_nodes_from(subsets)

    # Define transitions under each label like an NFA→DFA subset construction
    for S in subsets:
        for lab in labels:
            # Collect successors of all nodes in S under edges with label lab
            succ = set()
            for u in S:
                for _, v, data in G.out_edges(u, data=True):
                    if data.get("label") == lab:
                        succ.add(v)
            if not succ:
                continue
            T = tuple(sorted(succ))
            # Only keep if T is also in our restricted node set
            if T in SG.nodes:
                SG.add_edge(S, T, label=lab)

    return SG


if __name__ == "__main__":
    main()