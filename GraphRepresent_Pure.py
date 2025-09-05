import json
from itertools import product
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
from graphviz import Digraph, Source


# Core: Wolfram Elementary Cellular Automata (ECA) to graph JSON

class CAToGraph:
    """
    Convert a Wolfram Elementary Cellular Automaton (ECA) rule (0..255) into a
    small multi-digraph structure, then summarize it into a JSON-friendly dict.

    Notes
    -----
    - Defaults are tailored for Wolfram ECA: num_states=2, radius=1.
    - Nodes are the center cell states (0 and 1).
    - Parallel edges represent different neighbor configurations that map
      the center state to the next state according to the rule.
    """

    def __init__(self, num_states: int = 2, radius: int = 1):
        """
        Parameters
        ----------
        num_states : int
            Number of states in the CA (for ECA this is 2).
        radius : int
            Neighborhood radius (for ECA this is 1).
        """
        self.num_states = num_states
        self.radius = radius
        self.neighborhood_size = 2 * radius + 1
        if not (self.num_states == 2 and self.radius == 1):
            print("[Warn] Current rule table only supports Wolfram ECA ordering (2 states, radius=1).")

    # ---------- Rule table (ECA) ----------
    def rule_number_to_table(self, rule_number: int) -> Dict[Tuple[int, ...], int]:
        """
        Convert a Wolfram rule number into a lookup table mapping a 3-bit
        neighborhood (left, center, right) to the next center state.

        Parameters
        ----------
        rule_number : int
            ECA rule number in [0, 255].

        Returns
        -------
        Dict[Tuple[int, ...], int]
            Mapping from neighborhood tuple (l, c, r) to next center state (0/1).
        """
        neighborhoods = [
            (1, 1, 1),
            (1, 1, 0),
            (1, 0, 1),
            (1, 0, 0),
            (0, 1, 1),
            (0, 1, 0),
            (0, 0, 1),
            (0, 0, 0),
        ]
        if not (0 <= rule_number <= 255):
            raise ValueError("Wolfram ECA rule number must be in [0, 255].")

        # MSB->LSB, e.g. 30 -> '00011110'
        rule_binary = format(rule_number, '08b')
        rule_outputs = [int(bit) for bit in rule_binary]

        rule_table: Dict[Tuple[int, ...], int] = {}
        for neighborhood, output in zip(neighborhoods, rule_outputs):
            rule_table[neighborhood] = output
        return rule_table

    # ---------- Neighbor enumerations ----------
    def generate_neighbor_configs(self) -> List[Tuple[int, ...]]:
        """
        Enumerate all possible (left_neighbors + right_neighbors) tuples
        excluding the center. For ECA radius=1, that is [(l, r)] with 4 combos.

        Returns
        -------
        List[Tuple[int, ...]]
            All neighbor side-pairs for the given number of states and radius.
        """
        neighbor_positions = 2 * self.radius
        return list(product(range(self.num_states), repeat=neighbor_positions))

    def neighbor_config_to_matrix(self, config: Tuple[int, ...]) -> np.ndarray:
        """
        Convert a side-neighbor tuple into a 2 x R matrix [left; right].
        For radius=1, it's [[l], [r]].

        Parameters
        ----------
        config : Tuple[int, ...]
            Concatenated (left_neighbors + right_neighbors).

        Returns
        -------
        np.ndarray
            Shape (2, R) matrix: first row is left side, second row is right side.
        """
        left_neighbors = config[:self.radius]
        right_neighbors = config[self.radius:]
        if self.radius == 1:
            return np.array([[left_neighbors[0]], [right_neighbors[0]]])
        return np.array([list(left_neighbors), list(right_neighbors)])

    # ---------- Graph build ----------
    def build_graph_from_rule_table(self, rule_table: Dict[Tuple[int, ...], int], rule_id: int):
        """
        Build a conceptual MultiDiGraph-like structure (as edges list) for ECA.
        Nodes are the two center states {0,1}, edges are transitions from the
        current center state to the next state induced by each neighbor side-pair.

        Parameters
        ----------
        rule_table : Dict[Tuple[int, ...], int]
            Mapping (l, c, r) -> next center state.
        rule_id : int
            Rule number.

        Returns
        -------
        Dict[str, Any]
            A dict with:
            - 'nodes': [0,1]
            - 'edges': list of dicts with fields:
                from_state, to_state, neighbor_config, neighbor_matrix,
                neighbor_string, full_neighborhood, rule_id
        """
        nodes = [0, 1]
        edges: List[Dict[str, Any]] = []
        neighbor_configs = self.generate_neighbor_configs()

        for center_state in nodes:
            for neighbor_config in neighbor_configs:
                left_neighbors = neighbor_config[:self.radius]
                right_neighbors = neighbor_config[self.radius:]
                full_neighborhood = left_neighbors + (center_state,) + right_neighbors
                if full_neighborhood in rule_table:
                    next_state = rule_table[full_neighborhood]
                    neighbor_matrix = self.neighbor_config_to_matrix(neighbor_config)
                    edges.append({
                        'from_state': center_state,
                        'to_state': next_state,
                        'neighbor_config': neighbor_config,
                        'neighbor_matrix': neighbor_matrix,
                        'neighbor_string': self.format_neighbor_string(neighbor_config),
                        'full_neighborhood': full_neighborhood,
                        'rule_id': rule_id
                    })
        return {"nodes": nodes, "edges": edges}

    def format_neighbor_string(self, config: Tuple[int, ...]) -> str:
        """
        Format side-neighbor tuple into 'L|R' string.

        Parameters
        ----------
        config : Tuple[int, ...]
            Concatenated (left_neighbors + right_neighbors).

        Returns
        -------
        str
            'L|R' representation, e.g., '1|0', '01|11', etc.
        """
        left = ''.join(map(str, config[:self.radius]))
        right = ''.join(map(str, config[self.radius:]))
        return f'{left}|{right}'

    def build_graph_from_rule_number(self, rule_number: int) -> Dict[str, Any]:
        """
        Convenience wrapper: build the edges structure directly from rule number.

        Parameters
        ----------
        rule_number : int
            ECA rule number in [0, 255].

        Returns
        -------
        Dict[str, Any]
            See `build_graph_from_rule_table` for structure.
        """
        rule_table = self.rule_number_to_table(rule_number)
        return self.build_graph_from_rule_table(rule_table, rule_number)

    # ---------- Feature extraction / summary ----------
    def extract_gnn_features(self, graph_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic edge index and attributes for downstream ML/graph usage.

        Parameters
        ----------
        graph_obj : Dict[str, Any]
            Output from `build_graph_from_rule_table`:
            {'nodes': [...], 'edges': [...]}

        Returns
        -------
        Dict[str, Any]
            Keys:
            - 'edge_index': np.ndarray of shape (2, E)
            - 'edge_attr': np.ndarray of flattened neighbor matrices (dtype=object)
            - 'rule_ids' : np.ndarray of rule ids per edge
            - 'num_nodes': int
            - 'num_edges': int
        """
        nodes = graph_obj["nodes"]
        edges = graph_obj["edges"]

        edge_index = [[], []]
        edge_attr, rule_ids = [], []
        for e in edges:
            u = e["from_state"]; v = e["to_state"]
            edge_index[0].append(u); edge_index[1].append(v)
            neighbor_matrix = np.array(e['neighbor_matrix'])
            edge_attr.append(neighbor_matrix.flatten())
            rule_ids.append(e['rule_id'])

        return {
            'edge_index': np.array(edge_index),
            'edge_attr': np.array(edge_attr, dtype=object),
            'rule_ids': np.array(rule_ids),
            'num_nodes': len(nodes),
            'num_edges': len(edges),
        }

    def get_graph_representation(self, rule_number: int) -> Dict[str, Any]:
        """
        Build a JSON-friendly summary for a single rule: basic graph statistics,
        the rule's 8-bit binary form, per-edge details and compact GNN features.

        Parameters
        ----------
        rule_number : int
            ECA rule number in [0, 255].

        Returns
        -------
        Dict[str, Any]
            Keys:
            - 'graph_stats': dict with rule_number, num_nodes, num_edges, density,
              and simple in/out degree stats (computed from edges).
            - 'rule_table': dict mapping neighborhood tuple string to next state.
            - 'rule_binary': 8-bit binary string of the rule number.
            - 'edge_details': list of per-edge dictionaries.
            - 'gnn_features': dict of edge_index/edge_attr/rule_ids (lists).
        """
        # Build graph-like structure
        graph_obj = self.build_graph_from_rule_number(rule_number)
        nodes = graph_obj["nodes"]
        edges = graph_obj["edges"]

        # Basic stats (for 2 nodes, density is simple E / (N*(N-1)) ignoring loops)
        num_nodes = len(nodes)
        num_edges = len(edges)
        # Directed density including self loops isn't directly meaningful here,
        # keep a simple ratio E / (N*N) to stay bounded in [0,1]
        density = num_edges / float(num_nodes * num_nodes)

        # Degree stats
        in_deg = {n: 0 for n in nodes}
        out_deg = {n: 0 for n in nodes}
        for e in edges:
            out_deg[e["from_state"]] += 1
            in_deg[e["to_state"]] += 1

        graph_stats = {
            'rule_number': rule_number,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'is_strongly_connected': None,  # Not using NX; left as None
            'max_in_degree': max(in_deg.values()) if in_deg else 0,
            'max_out_degree': max(out_deg.values()) if out_deg else 0,
            'avg_in_degree': float(np.mean(list(in_deg.values()))) if in_deg else 0.0,
            'avg_out_degree': float(np.mean(list(out_deg.values()))) if out_deg else 0.0,
        }

        # Rule table + edges as details
        rule_table = self.rule_number_to_table(rule_number)
        rule_binary = format(rule_number, '08b')
        edge_details = []
        for e in edges:
            edge_details.append({
                'from_state': e['from_state'],
                'to_state': e['to_state'],
                'key': 0,  # placeholder; MultiDiGraph key concept (not used here)
                'full_neighborhood': e['full_neighborhood'],
                'neighbor_config': e['neighbor_config'],
                'neighbor_string': e['neighbor_string']
            })

        # GNN features
        feats = self.extract_gnn_features(graph_obj)

        return {
            'graph_stats': graph_stats,
            'rule_table': {str(k): v for k, v in rule_table.items()},
            'rule_binary': rule_binary,
            'edge_details': edge_details,
            'gnn_features': {
                'edge_index': feats['edge_index'].tolist(),
                'edge_attr': [attr.tolist() for attr in feats['edge_attr']],
                'rule_ids': feats['rule_ids'].tolist()
            }
        }


# Analysis: compute and persist all 256 rules, plus summary stats

def analyze_all_rules() -> List[Dict[str, Any]]:
    """
    Build the JSON representation for all 256 ECA rules and save it.

    Returns
    -------
    List[Dict[str, Any]]
        The list of rule data dicts (length 256). Also saves
        `all_eca_rules_graph_representation.json` in the current folder.
    """
    ca_converter = CAToGraph(num_states=2, radius=1)
    all_rules_data: List[Dict[str, Any]] = []

    print("Analyzing graph representations for all ECA rules...")
    print("=" * 60)

    for rule_number in range(256):
        try:
            rule_data = ca_converter.get_graph_representation(rule_number)
            all_rules_data.append(rule_data)
            stats = rule_data['graph_stats']
            print(f"Rule {rule_number:3d}: "
                  f"Nodes={stats['num_nodes']}, "
                  f"Edges={stats['num_edges']}, "
                  f"Density={stats['density']:.3f}")
        except Exception as e:
            print(f"Error processing Rule {rule_number}: {e}")
            continue

    print("=" * 60)
    print(f"Successfully analyzed {len(all_rules_data)} rules.")

    out = "all_eca_rules_graph_representation.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_rules_data, f, indent=2, ensure_ascii=False)
    print(f"Graph representations for all rules have been saved to: {out}")

    generate_summary_statistics(all_rules_data)
    return all_rules_data


def generate_summary_statistics(all_rules_data: List[Dict[str, Any]]) -> None:
    """
    Compute simple per-rule statistics and save them as a CSV.

    Parameters
    ----------
    all_rules_data : List[Dict[str, Any]]
        Output from `analyze_all_rules()`.
    -----
    eca_rules_summary.csv : CSV
        One row per rule: rule_number, rule_binary, num_nodes, num_edges, density,
        degree stats.
    """
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)

    num_edges_list = [d['graph_stats']['num_edges'] for d in all_rules_data]
    density_list = [d['graph_stats']['density'] for d in all_rules_data]

    print(f"Edge Count Statistics:")
    print(f"  Min Edges: {min(num_edges_list)}")
    print(f"  Max Edges: {max(num_edges_list)}")
    print(f"  Avg Edges: {np.mean(num_edges_list):.2f}")

    print(f"\nDensity Statistics:")
    print(f"  Min Density: {min(density_list):.3f}")
    print(f"  Max Density: {max(density_list):.3f}")
    print(f"  Avg Density: {np.mean(density_list):.3f}")

    # Build CSV rows
    rows = []
    for d in all_rules_data:
        s = d['graph_stats']
        rows.append({
            'rule_number': s['rule_number'],
            'rule_binary': d['rule_binary'],
            'num_nodes': s['num_nodes'],
            'num_edges': s['num_edges'],
            'density': s['density'],
            'max_in_degree': s['max_in_degree'],
            'max_out_degree': s['max_out_degree'],
            'avg_in_degree': s['avg_in_degree'],
            'avg_out_degree': s['avg_out_degree'],
        })
    df = pd.DataFrame(rows)
    csv_file = "eca_rules_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSummary data has been saved to: {csv_file}")


# Graphviz helpers: colors and legend (HTML-like labels)

def _distinct_colors(k: int) -> List[str]:
    """
    Generate k visually distinct colors in HEX.
    Prefer `tab10`/`tab20`, then fall back to HSV sampling.

    Parameters
    ----------
    k : int
        Number of colors.

    Returns
    -------
    List[str]
        HEX color strings (e.g. '#1f77b4').
    """
    cols: List[str] = []
    if k <= 10:
        cmap = mpl.colormaps.get_cmap("tab10")
        cols = [mpl.colors.to_hex(cmap(i/9.0)) for i in range(10)][:k]
    elif k <= 20:
        cmap = mpl.colormaps.get_cmap("tab20")
        cols = [mpl.colors.to_hex(cmap(i/19.0)) for i in range(20)][:k]
    else:
        for i in range(k):
            rgb = mpl.colors.hsv_to_rgb([i / k, 0.65, 0.95])
            cols.append(mpl.colors.to_hex(rgb))
    return cols


def _legend_html(pairs: List[Tuple[str, str]], title: str = "Legend") -> str:
    """
    Build a Graphviz HTML-like label for a legend: one row per (label, color).

    Parameters
    ----------
    pairs : List[Tuple[str, str]]
        Each tuple is (label_text, color_hex).
    title : str
        Legend title.

    Returns
    -------
    str
        Graphviz HTML-like label string (must be set on a node with shape=plaintext).
    """
    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []
    for lab, col in pairs:
        lab = esc(lab if lab else "∅")
        rows.append(
            f'<TR>'
            f'  <TD WIDTH="10" BGCOLOR="{col}"></TD>'
            f'  <TD ALIGN="LEFT"><FONT POINT-SIZE="12">{lab}</FONT></TD>'
            f'</TR>'
        )
    html = (
        '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
        f'<TR><TD COLSPAN="2" ALIGN="CENTER"><B><FONT POINT-SIZE="13">{esc(title)}</FONT></B></TD></TR>'
        + "".join(rows) +
        '</TABLE>>'
    )
    return html


def _val_or_none(x) -> Optional[float]:
    """
    Safely convert to float and return None if not finite.

    Parameters
    ----------
    x : Any

    Returns
    -------
    Optional[float]
        Finite float or None.
    """
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _wmap(val: Optional[float], vmin: float, vmax: float,
          wmin: float = 2.2, wmax: float = 7.0) -> float:
    """
    Map a numeric value into a pen width range.

    Parameters
    ----------
    val : Optional[float]
        Value to normalize. If None, the midpoint is used.
    vmin, vmax : float
        Value range for normalization.
    wmin, wmax : float
        Output width range.

    Returns
    -------
    float
        Pen width for Graphviz edge.
    """
    if val is None or vmax <= vmin:
        return (wmin + wmax) / 2.0
    x = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))
    return wmin + (wmax - wmin) * x


# Graphviz rendering: one image per rule, legend on the right

def draw_rule_graph_graphviz_from_item(
    rule_data: Dict[str, Any],
    out_path: str,
    width_attr: str = "weight",
    title: Optional[str] = None,
    color_by_neighbor: bool = True,
    show_legend: bool = True,
    extra_formats: Optional[List[str]] = None,
    dpi: int = 300,
    bgcolor: str = "white"
) -> List[str]:
    # -------- collect nodes and edges --------
    nodes = sorted({int(ed["from_state"]) for ed in rule_data["edge_details"]}
                   | {int(ed["to_state"]) for ed in rule_data["edge_details"]})
    edges = []
    for ed in rule_data["edge_details"]:
        e = dict(u=int(ed["from_state"]),
                 v=int(ed["to_state"]),
                 nbr=ed.get("neighbor_string") or "")
        if width_attr in ed:
            e["w"] = ed[width_attr]
        edges.append(e)

    # -------- neighborhood -> color mapping --------
    nbr_keys = sorted({e["nbr"] for e in edges})
    palette = _distinct_colors(len(nbr_keys))
    nbr2color = {k: c for k, c in zip(nbr_keys, palette)}

    # line width range
    def _v(x):
        try:
            f = float(x)
            return f if np.isfinite(f) else None
        except Exception:
            return None
    wvals = [_v(e.get("w")) for e in edges if _v(e.get("w")) is not None]
    wmin, wmax = (min(wvals), max(wvals)) if wvals else (0.0, 1.0)
    def _wmap(val, vmin, vmax, wmin_=2.2, wmax_=7.0):
        if val is None or vmax <= vmin: return (wmin_ + wmax_) / 2.0
        x = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))
        return wmin_ + (wmax_ - wmin_) * x

    # -------- output formats --------
    out_path = Path(out_path)
    main_fmt = out_path.suffix.lstrip(".").lower() or "svg"
    if main_fmt not in {"svg", "png", "jpg", "jpeg", "pdf"}:
        main_fmt = "svg"; out_path = out_path.with_suffix(".svg")
    extras = []
    if extra_formats:
        for f in extra_formats:
            f = f.lower()
            if f == "jpeg": f = "jpg"
            if f not in extras:
                extras.append(f)

    # -------- build DOT graph --------
    dot = Digraph(name=title or "CA Rule", format=main_fmt, engine="dot")
    dot.attr(charset="UTF-8", forcelabels="true")
    dot.attr(rankdir="LR", splines="true", overlap="false",
             nodesep="1.3", ranksep="1.1", pad="0.28", margin="0.18",
             bgcolor=bgcolor, dpi=str(dpi))

    # larger nodes
    dot.attr("node", shape="circle", fixedsize="true",
             width="1.40", height="1.40", margin="0.06,0.05",
             style="filled", fillcolor="#eef3ff", color="#5b6ea6", penwidth="1.8",
             fontsize="18")

    # edges (no text labels, only color/width)
    dot.attr("edge", arrowsize="1.0", arrowhead="normal",
             color="#333333", fontsize="14", labelfontsize="13")

    # create nodes
    for n in nodes:
        dot.node(str(n), label=str(n))

    # Key ①: align all state nodes horizontally (same rank)
    with dot.subgraph(name="rank_same_states") as s:
        s.attr(rank="same")
        for n in nodes:
            s.node(str(n))

    # distribute self-loops across four quadrants
    loop_ports = ["ne", "se", "sw", "nw"]
    loop_counters: Dict[int, int] = {n: 0 for n in nodes}

    # edges
    for e in edges:
        u, v = str(e["u"]), str(e["v"])
        col = nbr2color[e["nbr"]]
        pw  = _wmap(_v(e.get("w")), wmin, wmax)

        if u == v:
            k = loop_counters[int(u)] % len(loop_ports)
            port = loop_ports[k]
            loop_counters[int(u)] += 1
            dot.edge(u, v,
                     color=col, penwidth=f"{pw:.2f}",
                     minlen="2",
                     tailport=port, headport=port,
                     constraint="false")
        else:
            dot.edge(u, v,
                     color=col, penwidth=f"{pw:.2f}",
                     minlen="2")

    # title
    if title:
        dot.attr(label=title, labelloc="t", fontsize="18")

    # legend (right side)
    if show_legend and len(nbr2color) > 0:
        legend_pairs = [(k, v) for k, v in nbr2color.items()]
        legend_label = _legend_html(sorted(legend_pairs), title="Neighbor → Color")

        # legend node: auto-sized
        dot.node("LEGEND", label=legend_label,
                 shape="plaintext", width="0", height="0",
                 margin="0", fixedsize="false")

        # Key ②: position legend at far right (sink rank)
        with dot.subgraph(name="rank_sink_legend") as s:
            s.attr(rank="sink")
            s.node("LEGEND")

        # use an invisible high-weight edge to pull legend to the right and align with nodes
        node_ids = [str(n) for n in nodes]
        anchor = "1" if "1" in node_ids else node_ids[-1]
        dot.edge(anchor, "LEGEND", style="invis", weight="20")

    # render
    out_dir = out_path.parent; out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_path.with_suffix(""))
    outputs = []
    dot.format = main_fmt
    outputs.append(str(Path(dot.render(base, cleanup=True))))
    for fmt in extras:
        s = Source(dot.source, engine="dot", format=fmt)
        outputs.append(str(Path(s.render(base, cleanup=True))))
    return outputs

def visualize_all_rules_graphviz_split(
    json_path: str,
    outdir_svg: str = "plots_svg",
    outdir_jpg: str = "plots_jpg",
    width_attr: str = "weight",
    dpi: int = 300
) -> None:
    """
    Batch-render all rules from a saved JSON file:
    - SVG files go to `outdir_svg`
    - JPG files go to `outdir_jpg`
    Both use the same layout: nodes left-right aligned, legend fixed on the right.

    Parameters
    ----------
    json_path : str
        Path to the JSON file produced by `analyze_all_rules()`.
    outdir_svg : str
        Output directory for SVG files.
    outdir_jpg : str
        Output directory for JPG files.
    width_attr : str
        Edge attribute name used for pen width normalization (if present).
    dpi : int
        DPI hint for Graphviz renderers (SVG unaffected; raster formats use it).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    Path(outdir_svg).mkdir(parents=True, exist_ok=True)
    Path(outdir_jpg).mkdir(parents=True, exist_ok=True)

    for item in arr:
        r = item["graph_stats"]["rule_number"]
        title = f"Rule {r}"

        # SVG
        svg_path = Path(outdir_svg) / f"rule_{r:03d}.svg"
        _ = draw_rule_graph_graphviz_from_item(
            item, str(svg_path),
            width_attr=width_attr,
            title=title,
            show_legend=True,
            extra_formats=None,
            dpi=dpi, bgcolor="white"
        )

        # JPG
        jpg_path = Path(outdir_jpg) / f"rule_{r:03d}.jpg"
        _ = draw_rule_graph_graphviz_from_item(
            item, str(jpg_path),
            width_attr=width_attr,
            title=title,
            show_legend=True,
            extra_formats=None,
            dpi=dpi, bgcolor="white"
        )

    print(f"[OK] SVG → {outdir_svg} | JPG → {outdir_jpg}")


# Script entry

if __name__ == "__main__":
    # 1) Build and persist all rules to JSON
    data = analyze_all_rules()

    # 2) Render each rule (SVG and JPG in separate folders)
    visualize_all_rules_graphviz_split(
        "all_eca_rules_graph_representation.json",
        outdir_svg="plots_svg",
        outdir_jpg="plots_jpg",
        width_attr="weight",
        dpi=300
    )