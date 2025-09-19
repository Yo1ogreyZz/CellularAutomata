import json
from itertools import product
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
from graphviz import Digraph, Source


# Core: Cellular Automata (CA) to graph JSON
class CAToGraph:
    """
    Convert a Cellular Automaton (CA) rule into a small multi-digraph structure,
    then summarize it into a JSON-friendly dict.

    This is generalized for any radius 'r' with a fixed number of states (2).
    """

    def __init__(self, num_states: int = 2, radius: int = 1):
        """
        Parameters
        ----------
        num_states : int
            Number of states in the CA (default is 2).
        radius : int
            Neighborhood radius.
        """
        if num_states != 2:
            raise NotImplementedError("This implementation currently only supports num_states=2.")
        self.num_states = num_states
        self.radius = radius
        self.neighborhood_size = 2 * radius + 1
        self.num_neighborhoods = self.num_states ** self.neighborhood_size

    # ---------- Rule table ----------
    def rule_number_to_table(self, rule_number: int) -> Dict[Tuple[int, ...], int]:
        """
        Convert a rule number into a lookup table mapping a neighborhood
        to the next center state.

        Parameters
        ----------
        rule_number : int
            CA rule number. The valid range depends on the radius.

        Returns
        -------
        Dict[Tuple[int, ...], int]
            Mapping from neighborhood tuple to the next center state.
        """
        # Generate all possible neighborhoods in reverse lexicographical order (like Wolfram's ECA)
        neighborhoods = list(product(range(self.num_states), repeat=self.neighborhood_size))[::-1]

        max_rule = (2 ** self.num_neighborhoods) - 1
        if not (0 <= rule_number <= max_rule):
            raise ValueError(f"For r={self.radius}, rule number must be in [0, {max_rule}].")

        # Format rule number to binary string, padded with zeros to match the number of neighborhoods
        rule_binary = format(rule_number, f'0{self.num_neighborhoods}b')
        rule_outputs = [int(bit) for bit in rule_binary]

        rule_table: Dict[Tuple[int, ...], int] = {}
        for neighborhood, output in zip(neighborhoods, rule_outputs):
            rule_table[neighborhood] = output
        return rule_table

    # ---------- Neighbor enumerations ----------
    def generate_neighbor_configs(self) -> List[Tuple[int, ...]]:
        """
        Enumerate all possible (left_neighbors + right_neighbors) tuples excluding the center.

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
        return np.array([list(left_neighbors), list(right_neighbors)])

    # ---------- Graph build ----------
    def build_graph_from_rule_table(self, rule_table: Dict[Tuple[int, ...], int], rule_id: int) -> Dict[str, Any]:
        """
        Build a conceptual MultiDiGraph-like structure (as an edges list).
        Nodes are the center states {0, 1}, edges are transitions.

        Parameters
        ----------
        rule_table : Dict[Tuple[int, ...], int]
            Mapping (l..., c, r...) -> next center state.
        rule_id : int
            Rule number.

        Returns
        -------
        Dict[str, Any]
            A dict with 'nodes' and 'edges' list.
        """
        nodes = list(range(self.num_states))
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
        Format side-neighbor tuple into 'L...|R...' string.
        """
        left = ''.join(map(str, config[:self.radius]))
        right = ''.join(map(str, config[self.radius:]))
        return f'{left}|{right}'

    def build_graph_from_rule_number(self, rule_number: int) -> Dict[str, Any]:
        """
        Convenience wrapper: build the graph structure directly from a rule number.
        """
        rule_table = self.rule_number_to_table(rule_number)
        return self.build_graph_from_rule_table(rule_table, rule_number)

    # ---------- Feature extraction / summary ----------
    def extract_gnn_features(self, graph_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract basic edge index and attributes for downstream ML/graph usage.
        """
        nodes = graph_obj["nodes"]
        edges = graph_obj["edges"]
        edge_index = [[], []]
        edge_attr, rule_ids = [], []

        for e in edges:
            u, v = e["from_state"], e["to_state"]
            edge_index[0].append(u)
            edge_index[1].append(v)
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
        Build a JSON-friendly summary for a single rule.
        """
        graph_obj = self.build_graph_from_rule_number(rule_number)
        nodes = graph_obj["nodes"]
        edges = graph_obj["edges"]

        num_nodes = len(nodes)
        num_edges = len(edges)
        density = num_edges / float(num_nodes * num_nodes) if num_nodes > 0 else 0.0

        in_deg = {n: 0 for n in nodes}
        out_deg = {n: 0 for n in nodes}
        for e in edges:
            out_deg[e["from_state"]] += 1
            in_deg[e["to_state"]] += 1

        graph_stats = {
            'rule_number': rule_number,
            'radius': self.radius,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'max_in_degree': max(in_deg.values()) if in_deg else 0,
            'max_out_degree': max(out_deg.values()) if out_deg else 0,
            'avg_in_degree': np.mean(list(in_deg.values())) if in_deg else 0.0,
            'avg_out_degree': np.mean(list(out_deg.values())) if out_deg else 0.0,
        }

        rule_table = self.rule_number_to_table(rule_number)
        rule_binary = format(rule_number, f'0{self.num_neighborhoods}b')

        edge_details = []
        for e in edges:
            edge_details.append({
                'from_state': e['from_state'],
                'to_state': e['to_state'],
                'key': 0,
                'full_neighborhood': e['full_neighborhood'],
                'neighbor_config': e['neighbor_config'],
                'neighbor_string': e['neighbor_string']
            })

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


# Analysis: compute and persist all rules for a given radius
def analyze_all_rules(radius: int) -> List[Dict[str, Any]]:
    """
    Build and save the JSON representation for all rules of a given radius.
    """
    ca_converter = CAToGraph(num_states=2, radius=radius)
    num_rules = 2 ** (2 ** (2 * radius + 1))

    all_rules_data: List[Dict[str, Any]] = []
    print(f"Analyzing graph representations for r={radius} (0..{num_rules - 1})...")
    print("=" * 60)

    for rule_number in range(num_rules):
        try:
            rule_data = ca_converter.get_graph_representation(rule_number)
            all_rules_data.append(rule_data)
            if rule_number % 10 == 0 or rule_number == num_rules - 1:  # Print progress
                stats = rule_data['graph_stats']
                print(f"Rule {rule_number:3d}: "
                      f"Nodes={stats['num_nodes']}, "
                      f"Edges={stats['num_edges']}, "
                      f"Density={stats['density']:.3f}")
        except Exception as e:
            print(f"Error processing Rule {rule_number}: {e}")
            continue

    print("=" * 60)
    print(f"Successfully analyzed {len(all_rules_data)} rules for r={radius}.")

    out_filename = f"r{radius}_all_rules_graph_representation.json"
    with open(out_filename, 'w', encoding='utf-8') as f:
        json.dump(all_rules_data, f, indent=2, ensure_ascii=False)
    print(f"Graph representations saved to: {out_filename}")

    generate_summary_statistics(all_rules_data, radius)
    return all_rules_data


def generate_summary_statistics(all_rules_data: List[Dict[str, Any]], radius: int) -> None:
    """
    Compute per-rule statistics and save them to a CSV file named with the radius.
    """
    if not all_rules_data:
        print("No data to generate statistics from.")
        return

    print("\n" + "=" * 60)
    print(f"Summary Statistics for r={radius}:")
    print("=" * 60)

    num_edges_list = [d['graph_stats']['num_edges'] for d in all_rules_data]
    density_list = [d['graph_stats']['density'] for d in all_rules_data]

    print(f"Edge Count Stats: Min={min(num_edges_list)}, Max={max(num_edges_list)}, Avg={np.mean(num_edges_list):.2f}")
    print(
        f"Density Stats:    Min={min(density_list):.3f}, Max={max(density_list):.3f}, Avg={np.mean(density_list):.3f}")

    rows = []
    for d in all_rules_data:
        s = d['graph_stats']
        rows.append({
            'rule_number': s['rule_number'],
            'rule_binary': d['rule_binary'],
            'radius': s['radius'],
            'num_nodes': s['num_nodes'],
            'num_edges': s['num_edges'],
            'density': s['density'],
            'max_in_degree': s['max_in_degree'],
            'max_out_degree': s['max_out_degree'],
            'avg_in_degree': s['avg_in_degree'],
            'avg_out_degree': s['avg_out_degree'],
        })

    df = pd.DataFrame(rows)
    csv_file = f"r{radius}_rules_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSummary data has been saved to: {csv_file}")


# Graphviz helpers (largely unchanged, but with minor tweaks for safety)
def _distinct_colors(k: int) -> List[str]:
    cols: List[str] = []
    if k <= 10:
        cmap = mpl.colormaps.get_cmap("tab10"); cols = [mpl.colors.to_hex(cmap(i / 9.0)) for i in range(10)][:k]
    elif k <= 20:
        cmap = mpl.colormaps.get_cmap("tab20"); cols = [mpl.colors.to_hex(cmap(i / 19.0)) for i in range(20)][:k]
    else:
        for i in range(k):
            rgb = mpl.colors.hsv_to_rgb([i / k, 0.65, 0.95])
            cols.append(mpl.colors.to_hex(rgb))
    return cols


def _legend_html(pairs: List[Tuple[str, str]], title: str = "Legend") -> str:
    def esc(s: str) -> str: return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = [
        f'<TR><TD WIDTH="10" BGCOLOR="{col}"></TD><TD ALIGN="LEFT"><FONT POINT-SIZE="12">{esc(lab if lab else "∅")}</FONT></TD></TR>'
        for lab, col in pairs]
    return '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">' + f'<TR><TD COLSPAN="2" ALIGN="CENTER"><B><FONT POINT-SIZE="13">{esc(title)}</FONT></B></TD></TR>' + "".join(
        rows) + '</TABLE>>'


def _val_or_none(x) -> Optional[float]:
    try:
        f = float(x); return f if np.isfinite(f) else None
    except Exception:
        return None


def _wmap(val: Optional[float], vmin: float, vmax: float, wmin: float = 2.2, wmax: float = 7.0) -> float:
    if val is None or vmax <= vmin: return (wmin + wmax) / 2.0
    x = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))
    return wmin + (wmax - wmin) * x


# Graphviz rendering: one image per rule, named with radius
def draw_rule_graph_graphviz_from_item(rule_data: Dict[str, Any], out_path: str, width_attr: str = "weight",
                                       title: Optional[str] = None, show_legend: bool = True,
                                       extra_formats: Optional[List[str]] = None, dpi: int = 300,
                                       bgcolor: str = "white") -> List[str]:
    nodes = sorted({int(ed["from_state"]) for ed in rule_data["edge_details"]} | {int(ed["to_state"]) for ed in
                                                                                  rule_data["edge_details"]})
    edges = [{'u': int(ed["from_state"]), 'v': int(ed["to_state"]), 'nbr': ed.get("neighbor_string", ""),
              'w': ed.get(width_attr)} for ed in rule_data["edge_details"]]

    nbr_keys = sorted({e["nbr"] for e in edges})
    palette = _distinct_colors(len(nbr_keys))
    nbr2color = {k: c for k, c in zip(nbr_keys, palette)}

    wvals = [_val_or_none(e.get("w")) for e in edges if _val_or_none(e.get("w")) is not None]
    wmin, wmax = (min(wvals), max(wvals)) if wvals else (0.0, 1.0)

    out_path = Path(out_path)
    main_fmt = out_path.suffix.lstrip(".").lower() or "svg"
    extras = [f.lower().replace("jpeg", "jpg") for f in extra_formats or []]

    dot = Digraph(name=title or "CA Rule", format=main_fmt, engine="dot")
    dot.attr(charset="UTF-8", rankdir="LR", splines="true", overlap="false", nodesep="1.3", ranksep="1.1",
             bgcolor=bgcolor, dpi=str(dpi), label=title or "", labelloc="t", fontsize="18")
    dot.attr("node", shape="circle", fixedsize="true", width="1.4", style="filled", fillcolor="#eef3ff",
             color="#5b6ea6", penwidth="1.8", fontsize="18")
    dot.attr("edge", arrowsize="1.0")

    for n in nodes: dot.node(str(n), label=str(n))

    with dot.subgraph(name="rank_same_states") as s:
        s.attr(rank="same")
        for n in nodes: s.node(str(n))

    loop_ports = ["ne", "se", "sw", "nw"];
    loop_counters: Dict[int, int] = {n: 0 for n in nodes}
    for e in edges:
        u, v = str(e["u"]), str(e["v"])
        col = nbr2color[e["nbr"]]
        pw = _wmap(_val_or_none(e.get("w")), wmin, wmax)
        if u == v:
            port = loop_ports[loop_counters[int(u)] % len(loop_ports)];
            loop_counters[int(u)] += 1
            dot.edge(u, v, color=col, penwidth=f"{pw:.2f}", minlen="2", tailport=port, headport=port,
                     constraint="false")
        else:
            dot.edge(u, v, color=col, penwidth=f"{pw:.2f}", minlen="2")

    if show_legend and nbr_keys:
        legend_label = _legend_html(sorted(nbr2color.items()), title="Neighbor → Color")
        dot.node("LEGEND", label=legend_label, shape="plaintext")
        with dot.subgraph(name="rank_sink_legend") as s:
            s.attr(rank="sink");
            s.node("LEGEND")
        anchor = "1" if "1" in [str(n) for n in nodes] else str(nodes[-1])
        dot.edge(anchor, "LEGEND", style="invis", weight="20")

    out_dir = out_path.parent;
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_path.with_suffix(""))
    outputs = [str(Path(dot.render(base, cleanup=True)))]
    for fmt in extras:
        s = Source(dot.source, engine="dot", format=fmt)
        outputs.append(str(Path(s.render(base, cleanup=True))))
    return outputs


def visualize_all_rules_graphviz_split(json_path: str, radius: int, dpi: int = 300) -> None:
    """
    Batch-render all rules from a JSON file, creating radius-specific folders.
    """
    outdir_svg = f"r{radius}_plots_svg"
    outdir_jpg = f"r{radius}_plots_jpg"

    with open(json_path, "r", encoding="utf-8") as f:
        all_rules_data = json.load(f)

    Path(outdir_svg).mkdir(parents=True, exist_ok=True)
    Path(outdir_jpg).mkdir(parents=True, exist_ok=True)

    print(f"\nRendering graphs for r={radius}...")
    for item in all_rules_data:
        r_num = item["graph_stats"]["rule_number"]
        title = f"Rule {r_num} (r={radius})"

        # SVG
        svg_path = Path(outdir_svg) / f"rule_{r_num:03d}.svg"
        draw_rule_graph_graphviz_from_item(item, str(svg_path), title=title, dpi=dpi)

        # JPG
        jpg_path = Path(outdir_jpg) / f"rule_{r_num:03d}.jpg"
        draw_rule_graph_graphviz_from_item(item, str(jpg_path), title=title, dpi=dpi)

    print(f"[OK] SVG images saved to → {outdir_svg}")
    print(f"[OK] JPG images saved to → {outdir_jpg}")


# Script entry
if __name__ == "__main__":
    # Define the radius 'r' here.
    # r=1 corresponds to the original Elementary Cellular Automata (256 rules).
    # Be cautious with r > 1, as the number of rules grows very fast (2^(2^5)=2^32 for r=2).
    R_VALUE = 2

    # 1) Build and persist all rules for the given radius to a JSON file.
    json_filename = f"r{R_VALUE}_all_rules_graph_representation.json"
    analyze_all_rules(radius=R_VALUE)

    # 2) Render each rule from the generated JSON file.
    visualize_all_rules_graphviz_split(json_filename, radius=R_VALUE, dpi=300)