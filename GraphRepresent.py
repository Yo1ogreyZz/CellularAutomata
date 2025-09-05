import json
import math
import os
from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Any, Tuple
from graphviz import Digraph, Source


class CAToGraph:
    """
    Convert Wolfram Elementary Cellular Automata (ECA) rules to a multi-digraph.

    Defaults assume ECA: num_states=2, radius=1.
    """
    def __init__(self, num_states: int = 2, radius: int = 1):
        self.num_states = num_states
        self.radius = radius
        self.neighborhood_size = 2 * radius + 1
        if not (self.num_states == 2 and self.radius == 1):
            print("[Warn] Current rule table only supports Wolfram ECA ordering (2 states, radius=1).")

    # ---------- Rule table (ECA) ----------
    def rule_number_to_table(self, rule_number: int) -> Dict[Tuple[int, ...], int]:
        """
        Wolfram ECA convention:
        Neighborhood order: 111,110,101,100,011,010,001,000
        Map to bits of rule_number from MSB to LSB.
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

        rule_binary = format(rule_number, '08b')  # MSB -> LSB, e.g., 30 -> '00011110'
        rule_outputs = [int(bit) for bit in rule_binary]

        rule_table: Dict[Tuple[int, ...], int] = {}
        for neighborhood, output in zip(neighborhoods, rule_outputs):
            rule_table[neighborhood] = output

        return rule_table

    # ---------- Neighbor enumerations ----------
    def generate_neighbor_configs(self) -> List[Tuple[int, ...]]:
        """
        All possible (left_neighbors + right_neighbors) tuples, length = 2 * radius.
        For radius=1, it's [(l,r)] with 4 combinations for binary states.
        """
        neighbor_positions = 2 * self.radius
        return list(product(range(self.num_states), repeat=neighbor_positions))

    def neighbor_config_to_matrix(self, config: Tuple[int, ...]) -> np.ndarray:
        """
        For radius=1, return [[l], [r]] to keep backward compatibility with your flattening code.
        For radius>1, returns a 2 x R matrix: [left; right].
        """
        left_neighbors = config[:self.radius]
        right_neighbors = config[self.radius:]

        if self.radius == 1:
            return np.array([[left_neighbors[0]], [right_neighbors[0]]])
        else:
            return np.array([list(left_neighbors), list(right_neighbors)])

    # ---------- Graph build ----------
    def build_graph_from_rule_table(self, rule_table: Dict[Tuple[int, ...], int], rule_id: int) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()

        # Nodes are center states
        for state in range(self.num_states):
            G.add_node(state, label=f"center={state}")

        neighbor_configs = self.generate_neighbor_configs()

        for center_state in range(self.num_states):
            for neighbor_config in neighbor_configs:
                left_neighbors = neighbor_config[:self.radius]
                right_neighbors = neighbor_config[self.radius:]
                full_neighborhood = left_neighbors + (center_state,) + right_neighbors

                if full_neighborhood in rule_table:
                    next_state = rule_table[full_neighborhood]
                    neighbor_matrix = self.neighbor_config_to_matrix(neighbor_config)

                    G.add_edge(
                        center_state,
                        next_state,
                        neighbor_config=neighbor_config,
                        neighbor_matrix=neighbor_matrix,
                        neighbor_string=self.format_neighbor_string(neighbor_config),
                        full_neighborhood=full_neighborhood,
                        rule_id=rule_id
                    )
        return G

    def format_neighbor_string(self, config: Tuple[int, ...]) -> str:
        left = ''.join(map(str, config[:self.radius]))
        right = ''.join(map(str, config[self.radius:]))
        return f'{left}|{right}'

    def build_graph_from_rule_number(self, rule_number: int) -> nx.MultiDiGraph:
        rule_table = self.rule_number_to_table(rule_number)
        return self.build_graph_from_rule_table(rule_table, rule_number)

    # ---------- GNN features ----------
    def extract_gnn_features(self, G: nx.MultiDiGraph) -> Dict[str, Any]:
        edges = list(G.edges(keys=True))
        edge_index = [[e[0] for e in edges], [e[1] for e in edges]]

        edge_attr = []
        rule_ids = []
        for u, v, key in edges:
            edge_data = G[u][v][key]
            neighbor_matrix = np.array(edge_data['neighbor_matrix'])
            edge_attr.append(neighbor_matrix.flatten())
            rule_ids.append(edge_data['rule_id'])

        return {
            'edge_index': np.array(edge_index),
            'edge_attr': np.array(edge_attr, dtype=object),
            'rule_ids': np.array(rule_ids),
            'num_nodes': G.number_of_nodes(),
            'num_edges': len(edges),
        }

    def get_graph_representation(self, rule_number: int) -> Dict[str, Any]:
        """
        Get the graph representation of a rule, including basic statistics and structural information.
        """
        graph = self.build_graph_from_rule_number(rule_number)
        features = self.extract_gnn_features(graph)

        # Basic graph statistics
        graph_stats = {
            'rule_number': rule_number,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_strongly_connected': nx.is_strongly_connected(graph),
        }

        # Node degree statistics
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        graph_stats.update({
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
            'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0,
        })

        # Edge details
        edge_details = []
        for u, v, k in graph.edges(keys=True):
            edge_data = graph[u][v][k]
            edge_details.append({
                'from_state': u,
                'to_state': v,
                'key': k,
                'full_neighborhood': edge_data['full_neighborhood'],
                'neighbor_config': edge_data['neighbor_config'],
                'neighbor_string': edge_data['neighbor_string']
            })

        # Rule table
        rule_table = self.rule_number_to_table(rule_number)
        rule_binary = format(rule_number, '08b')

        return {
            'graph_stats': graph_stats,
            'rule_table': {str(k): v for k, v in rule_table.items()},
            'rule_binary': rule_binary,
            'edge_details': edge_details,
            'gnn_features': {
                'edge_index': features['edge_index'].tolist(),
                'edge_attr': [attr.tolist() for attr in features['edge_attr']],
                'rule_ids': features['rule_ids'].tolist()
            }
        }


def analyze_all_rules():
    """
    Analyze the graph representation of all 256 ECA rules (0-255).
    """
    ca_converter = CAToGraph(num_states=2, radius=1)
    all_rules_data = []

    print("Analyzing graph representations for all ECA rules...")
    print("=" * 60)

    for rule_number in range(256):
        try:
            rule_data = ca_converter.get_graph_representation(rule_number)
            all_rules_data.append(rule_data)

            # Print basic information
            stats = rule_data['graph_stats']
            print(f"Rule {rule_number:3d}: "
                  f"Nodes={stats['num_nodes']}, "
                  f"Edges={stats['num_edges']}, "
                  f"Density={stats['density']:.3f}, "
                  f"StronglyConnected={stats['is_strongly_connected']}")

        except Exception as e:
            print(f"Error processing Rule {rule_number}: {e}")
            continue

    print("=" * 60)
    print(f"Successfully analyzed {len(all_rules_data)} rules.")

    # Save all data to a JSON file
    output_file = "all_eca_rules_graph_representation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_rules_data, f, indent=2, ensure_ascii=False)

    print(f"Graph representations for all rules have been saved to: {output_file}")

    # Generate summary statistics
    generate_summary_statistics(all_rules_data)

    return all_rules_data


def generate_summary_statistics(all_rules_data):
    """
    Generate summary statistics.
    """
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)

    # Extract statistical data
    num_edges_list = [data['graph_stats']['num_edges'] for data in all_rules_data]
    density_list = [data['graph_stats']['density'] for data in all_rules_data]
    strongly_connected_count = sum(1 for data in all_rules_data if data['graph_stats']['is_strongly_connected'])

    # Basic statistics
    print(f"Edge Count Statistics:")
    print(f"  Min Edges: {min(num_edges_list)}")
    print(f"  Max Edges: {max(num_edges_list)}")
    print(f"  Avg Edges: {sum(num_edges_list)/len(num_edges_list):.2f}")

    print(f"\nDensity Statistics:")
    print(f"  Min Density: {min(density_list):.3f}")
    print(f"  Max Density: {max(density_list):.3f}")
    print(f"  Avg Density: {sum(density_list)/len(density_list):.3f}")

    print(f"\nConnectivity:")
    print(f"  Strongly Connected Rules: {strongly_connected_count}/256 ({strongly_connected_count/256*100:.1f}%)")

    # Group by edge count
    edge_count_groups = {}
    for data in all_rules_data:
        edge_count = data['graph_stats']['num_edges']
        if edge_count not in edge_count_groups:
            edge_count_groups[edge_count] = []
        edge_count_groups[edge_count].append(data['graph_stats']['rule_number'])

    print(f"\nGrouped by Edge Count:")
    for edge_count in sorted(edge_count_groups.keys()):
        rules = edge_count_groups[edge_count]
        print(f"  {edge_count} edges: {len(rules)} rules - {rules[:10]}{'...' if len(rules) > 10 else ''}")

    # Save summary statistics to CSV
    summary_data = []
    for data in all_rules_data:
        stats = data['graph_stats']
        summary_data.append({
            'rule_number': stats['rule_number'],
            'rule_binary': data['rule_binary'],
            'num_nodes': stats['num_nodes'],
            'num_edges': stats['num_edges'],
            'density': stats['density'],
            'is_strongly_connected': stats['is_strongly_connected'],
            'max_in_degree': stats['max_in_degree'],
            'max_out_degree': stats['max_out_degree'],
            'avg_in_degree': stats['avg_in_degree'],
            'avg_out_degree': stats['avg_out_degree']
        })

    df = pd.DataFrame(summary_data)
    csv_file = "eca_rules_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSummary data has been saved to: {csv_file}")

def graph_from_rule_data(rule_data: Dict[str, Any]) -> nx.MultiDiGraph:
    """
    用 JSON 里的 edge_details 还原 nx.MultiDiGraph。
    若 edge_details 里包含 weight / p_spread / p_recover / delta_diff_mean 等字段，会一并写入边属性。
    """
    G = nx.MultiDiGraph()
    # 收集节点
    states = set()
    for ed in rule_data.get("edge_details", []):
        states.add(int(ed["from_state"]))
        states.add(int(ed["to_state"]))
    for s in sorted(states):
        G.add_node(int(s), label=f"center={int(s)}")

    # 加边 + 把动态属性塞回去
    for ed in rule_data.get("edge_details", []):
        u = int(ed["from_state"]); v = int(ed["to_state"])
        attrs = dict(
            neighbor_config=tuple(ed["neighbor_config"]),
            neighbor_string=ed.get("neighbor_string")
        )
        # 可选动态字段（若不存在就不写）
        for k in ("count","weight","p_spread","p_recover","delta_diff_mean"):
            if k in ed:
                attrs[k] = ed[k]
        G.add_edge(u, v, **attrs)
    return G

def _radii_for_multiples(n: int, max_abs: float = 0.38) -> List[float]:
    """为同一对(u,v)的 n 条边分配不同曲率；n=1返回[0.0]，n>1在[-max_abs, max_abs]均匀分布。"""
    if n <= 1:
        return [0.0]
    return list(np.linspace(-max_abs, max_abs, n))

def _map_width(val: float, vmin: float, vmax: float, wmin=2.0, wmax=6.5) -> float:
    if val is None or np.isnan(val) or vmax <= vmin:
        return (wmin + wmax) / 2.0
    x = (val - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    return wmin + (wmax - wmin) * x

def _draw_self_loop(ax, pos_u, radius=0.5, angle=110, color="#333", width=2.2, arrowstyle="-|>", arrowsize=18):
    """
    画自环：围绕节点画一段小圆弧并加箭头。
    angle 决定自环出现在哪个象限；radius 决定弧大小。
    """
    from matplotlib.patches import FancyArrowPatch
    x, y = pos_u
    # 自环控制点（近似）：从节点偏移一个向量作为弧线端点
    theta1 = math.radians(angle)
    theta2 = math.radians(angle + 140)  # 140度跨度，够明显
    p1 = (x + radius * math.cos(theta1), y + radius * math.sin(theta1))
    p2 = (x + radius * math.cos(theta2), y + radius * math.sin(theta2))
    con_style = "arc3,rad=0.6"
    patch = FancyArrowPatch(
        p1, p2, connectionstyle=con_style,
        arrowstyle=arrowstyle, mutation_scale=arrowsize,
        linewidth=width, color=color
    )
    ax.add_patch(patch)


def draw_rule_graph_networkx(
    G: nx.MultiDiGraph,
    outfile: str = None,
    title: str = None,
    color_attr: str = "p_spread",   # 边颜色：扰动扩散率（没有则统一颜色）
    width_attr: str = "weight",     # 线宽：触发频率（没有则统一宽度）
    show_labels: bool = True,
    layout: str = "LR"
) -> None:
    # 1) 节点布局：ECA常见只有0/1两个节点，固定左右；否则spring布局
    nodes = list(G.nodes())
    if layout == "LR" and set(nodes) == {0, 1}:
        pos = {0: (-1.1, 0.0), 1: (1.1, 0.0)}
    else:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=160)

    # 2) 预取边属性与分组（处理平行边）
    edge_groups: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    color_vals, width_vals = [], []
    for u, v, k in G.edges(keys=True):
        edge_groups.setdefault((u, v), []).append((u, v, k))
        d = G[u][v][k]
        color_vals.append(float(d.get(color_attr, np.nan)))
        width_vals.append(float(d.get(width_attr, np.nan)))

    c_has = np.isfinite(color_vals).any()
    w_has = np.isfinite(width_vals).any()
    cmin, cmax = (np.nanmin(color_vals), np.nanmax(color_vals)) if c_has else (0.0, 1.0)
    wmin, wmax = (np.nanmin(width_vals), np.nanmax(width_vals)) if w_has else (0.0, 1.0)

    # 3) 节点
    nodes_artist = nx.draw_networkx_nodes(
        G, pos,
        node_color="#e8f0fe", node_size=1250,
        edgecolors="#5b6ea6", linewidths=1.4, ax=ax
    )
    try:
        nodes_artist.set_zorder(3)
    except Exception:
        pass  # 个别后端不需要 / 不支持

    label_artists = nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold", ax=ax)
    for txt in label_artists.values():
        try:
            txt.set_zorder(4)
        except Exception:
            pass

    # 4) 画非自环边（每组不同曲率），加强箭头与边距
    cmap = mpl.colormaps.get_cmap("viridis")
    for (u, v), ek in edge_groups.items():
        if u == v:
            continue  # 自环稍后单独处理
        rads = _radii_for_multiples(len(ek), max_abs=0.38)
        for idx, (rad, (u1, v1, k1)) in enumerate(zip(rads, ek)):
            d = G[u1][v1][k1]
            # 颜色
            if c_has:
                cval = float(d.get(color_attr, (cmin + cmax) / 2))
                cnorm = 0.5 if cmax <= cmin else (cval - cmin) / (cmax - cmin)
                ecolor = cmap(np.clip(cnorm, 0, 1))
            else:
                ecolor = "#3b3b3b"
            # 线宽
            if w_has:
                wval = float(d.get(width_attr, (wmin + wmax) / 2))
                width = _map_width(wval, wmin, wmax, wmin=2.0, wmax=6.5)
            else:
                width = 2.6

            common = dict(
                edgelist=[(u1, v1)],
                connectionstyle=f"arc3,rad={rad}",
                arrows=True, arrowstyle="-|>", arrowsize=18,
                width=width, edge_color=[ecolor], ax=ax
            )
            # 新版NetworkX支持的“箭头离节点留白”（方向更清晰）
            try:
                nx.draw_networkx_edges(
                    G, pos,
                    min_source_margin=15, min_target_margin=15,  # 15像素距
                    **common
                )
            except TypeError:
                nx.draw_networkx_edges(G, pos, **common)

            # 边标签（随曲率错位，白底半透明）
            if show_labels:
                lbl = d.get("neighbor_string")
                if lbl is None:
                    cfg = d.get("neighbor_config")
                    if cfg is not None:
                        cfg = list(cfg)
                        r = len(cfg)//2
                        lbl = f"{''.join(map(str,cfg[:r]))}|{''.join(map(str,cfg[r:]))}"
                if lbl is None:
                    lbl = ""
                x0, y0 = pos[u1]; x1, y1 = pos[v1]
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                dx, dy = (x1 - x0), (y1 - y0)
                nxp, nyp = -dy, dx
                norm = math.hypot(nxp, nyp) or 1.0
                # 让标签随曲率远离重叠，idx也稍微错层
                off = 0.30 * rad + 0.06 * (idx - (len(ek)-1)/2)
                lx, ly = mx + off * (nxp / norm), my + off * (nyp / norm)
                ax.text(
                    lx, ly, lbl, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
                    ha="center", va="center", zorder=5, clip_on=False
                )

    # 5) 自环单独绘制，避免被节点覆盖
    for (u, v), ek in edge_groups.items():
        if u != v:
            continue
        # 多个自环就改变角度与半径区分
        for j, (_, _, k1) in enumerate(ek):
            d = G[u][v][k1]
            # 颜色&宽度（与上面一致）
            if c_has:
                cval = float(d.get(color_attr, (cmin + cmax) / 2))
                cnorm = 0.5 if cmax <= cmin else (cval - cmin) / (cmax - cmin)
                ecolor = cmap(np.clip(cnorm, 0, 1))
            else:
                ecolor = "#3b3b3b"
            if w_has:
                wval = float(d.get(width_attr, (wmin + wmax) / 2))
                width = _map_width(wval, wmin, wmax, wmin=2.0, wmax=6.5)
            else:
                width = 2.6
            angle = 110 + 40 * j   # 多个自环错开角度
            _draw_self_loop(ax, pos[u], radius=0.55, angle=angle, color=ecolor, width=width)

            if show_labels:
                lbl = d.get("neighbor_string", "")
                x, y = pos[u]
                # 标签放在自环外侧
                theta = math.radians(angle + 70)
                ax.text(
                    x + 0.80 * math.cos(theta), y + 0.80 * math.sin(theta),
                    lbl, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
                    ha="center", va="center", zorder=5, clip_on=False
                )

    # 6) 标题 & 色条 & 边距
    if title:
        ax.set_title(title, fontsize=14, pad=8)
    if c_has:
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax),
                                   cmap=mpl.colormaps.get_cmap("viridis"))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label(color_attr, rotation=270, labelpad=10)

    # 给画布更多余量，防止箭头/文字被裁
    ax.margins(x=0.35, y=0.35)
    ax.axis("off")
    plt.tight_layout()
    if outfile:
        os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    else:
        plt.show()

def plot_all_rules_from_json(json_path: str,
                             outdir: str = "plots",
                             color_attr: str = "p_spread",
                             width_attr: str = "weight",
                             title_template: str = "Rule {r}") -> None:
    """
    从你已有的 JSON（analyze_all_rules 生成的，或后续合并后带动态字段的）批量绘图。
    若 JSON 没有 weight/p_spread，就会用统一宽度/颜色。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    os.makedirs(outdir, exist_ok=True)

    for item in arr:
        G = graph_from_rule_data(item)
        r = item.get("graph_stats", {}).get("rule_number", item.get("rule_number"))
        draw_rule_graph_networkx(
            G,
            outfile=os.path.join(outdir, f"rule_{int(r):03d}.png"),
            title=title_template.format(r=int(r)),
            color_attr=color_attr,
            width_attr=width_attr,
            show_labels=True,
            layout="LR"
        )
    print(f"[OK] saved plots to: {outdir}")

def _html_badge(text: str, color_hex: str, point_size: int = 12) -> str:
    """
    生成一个HTML-like标签：白底小盒子 + 同色圆点，增强边与标签的关联。
    注意：Graphviz HTML-like标签需要用 <...> 包裹整个字符串。
    """
    if text is None:
        text = ""
    # 简单转义
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLPADDING="3" CELLSPACING="0">
  <TR><TD BGCOLOR="white" COLOR="{color_hex}">
    <FONT POINT-SIZE="{point_size}">
      <FONT COLOR="{color_hex}">&#9679;</FONT>&nbsp;{safe}
    </FONT>
  </TD></TR>
</TABLE>
>'''

def _distinct_colors(k: int) -> list[str]:
    """生成 k 个差异较大的离散颜色（hex）。优先用 tab10/tab20，不够再用 hsv 均匀采样。"""
    import matplotlib as mpl
    import numpy as np
    cols = []
    if k <= 10:
        cmap = mpl.colormaps.get_cmap("tab10")
        cols = [mpl.colors.to_hex(cmap(i/9.0)) for i in range(10)][:k]
    elif k <= 20:
        cmap = mpl.colormaps.get_cmap("tab20")
        cols = [mpl.colors.to_hex(cmap(i/19.0)) for i in range(20)][:k]
    else:
        # 退化：在 hsv 上均匀采样
        for i in range(k):
            cols.append(mpl.colors.to_hex(mpl.colors.hsv_to_rgb([i/k, 0.65, 0.95])))
    return cols

def _legend_html(pairs: list[tuple[str, str]], title: str = "Legend") -> str:
    """
    用 Graphviz HTML-like label 生成图例：每行一个彩色小方块 + 文本。
    pairs: [(label_text, color_hex), ...]
    """
    # 简单转义
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

def _val_or_none(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _map_width(val, vmin, vmax, wmin=1.8, wmax=6.0):
    if val is None or vmax <= vmin:
        return (wmin + wmax) / 2.0
    x = (val - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    return wmin + (wmax - wmin) * x

def _color_from_value(val, vmin, vmax, cmap_name="viridis", default="#333333"):
    if val is None or vmax <= vmin:
        return default
    cmap = mpl.colormaps.get_cmap(cmap_name)
    c = cmap((val - vmin) / (vmax - vmin))
    return mpl.colors.to_hex(c, keep_alpha=False)

def graph_from_rule_data(rule_data: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """从你的 JSON item 里取出节点列表与边列表（便于 Graphviz 渲染）。"""
    nodes = sorted({int(ed["from_state"]) for ed in rule_data["edge_details"]}
                   | {int(ed["to_state"]) for ed in rule_data["edge_details"]})
    edges = []
    for ed in rule_data["edge_details"]:
        e = dict(
            u=int(ed["from_state"]),
            v=int(ed["to_state"]),
            label=ed.get("neighbor_string")  # 例如 "0|1"
        )
        # 可能存在的动态字段
        for k in ("weight", "p_spread", "p_recover", "delta_diff_mean"):
            if k in ed:
                e[k] = ed[k]
        edges.append(e)
    return nodes, edges

def _val_or_none(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _map_width(val, vmin, vmax, wmin=1.8, wmax=6.0):
    if val is None or vmax <= vmin:
        return (wmin + wmax) / 2.0
    x = (val - vmin) / (vmax - vmin)
    x = float(np.clip(x, 0.0, 1.0))
    return wmin + (wmax - wmin) * x

def _color_from_value(val, vmin, vmax, cmap_name="viridis", default="#333333"):
    if val is None or vmax <= vmin:
        return default
    cmap = mpl.colormaps.get_cmap(cmap_name)
    c = cmap((val - vmin) / (vmax - vmin))
    return mpl.colors.to_hex(c, keep_alpha=False)

def graph_from_rule_data(rule_data: Dict[str, Any]) -> Tuple[List[int], List[Dict[str, Any]]]:
    nodes = sorted({int(ed["from_state"]) for ed in rule_data["edge_details"]}
                   | {int(ed["to_state"]) for ed in rule_data["edge_details"]})
    edges = []
    for ed in rule_data["edge_details"]:
        e = dict(
            u=int(ed["from_state"]),
            v=int(ed["to_state"]),
            label=ed.get("neighbor_string")  # 例如 "0|1"
        )
        for k in ("weight", "p_spread", "p_recover", "delta_diff_mean"):
            if k in ed:
                e[k] = ed[k]
        edges.append(e)
    return nodes, edges

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
    # -------- 取节点与边 --------
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

    # -------- 邻域 -> 颜色 --------
    nbr_keys = sorted({e["nbr"] for e in edges})
    palette = _distinct_colors(len(nbr_keys))
    nbr2color = {k: c for k, c in zip(nbr_keys, palette)}

    # 线宽范围
    def _v(x):
        try:
            f = float(x);
            return f if np.isfinite(f) else None
        except Exception:
            return None
    wvals = [_v(e.get("w")) for e in edges if _v(e.get("w")) is not None]
    wmin, wmax = (min(wvals), max(wvals)) if wvals else (0.0, 1.0)
    def _wmap(val, vmin, vmax, wmin_=2.2, wmax_=7.0):
        if val is None or vmax <= vmin: return (wmin_ + wmax_) / 2.0
        x = float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))
        return wmin_ + (wmax_ - wmin_) * x

    # -------- 输出格式 --------
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

    # -------- 构建 DOT --------
    dot = Digraph(name=title or "CA Rule", format=main_fmt, engine="dot")
    dot.attr(charset="UTF-8", forcelabels="true")
    dot.attr(rankdir="LR", splines="true", overlap="false",
             nodesep="1.3", ranksep="1.1", pad="0.28", margin="0.18",
             bgcolor=bgcolor, dpi=str(dpi))

    # 更大节点
    dot.attr("node", shape="circle", fixedsize="true",
             width="1.40", height="1.40", margin="0.06,0.05",
             style="filled", fillcolor="#eef3ff", color="#5b6ea6", penwidth="1.8",
             fontsize="18")

    # 边（无标签，只颜色/宽度）
    dot.attr("edge", arrowsize="1.0", arrowhead="normal",
             color="#333333", fontsize="14", labelfontsize="13")

    # 创建节点
    for n in nodes:
        dot.node(str(n), label=str(n))

    # 关键①：所有状态节点水平对齐（同一 rank）
    with dot.subgraph(name="rank_same_states") as s:
        s.attr(rank="same")
        for n in nodes:
            s.node(str(n))

    # 自环四象限分散
    loop_ports = ["ne", "se", "sw", "nw"]
    loop_counters: Dict[int, int] = {n: 0 for n in nodes}

    # 边
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

    # 标题
    if title:
        dot.attr(label=title, labelloc="t", fontsize="18")

    # 图例（右侧）
    if show_legend and len(nbr2color) > 0:
        legend_pairs = [(k, v) for k, v in nbr2color.items()]
        legend_label = _legend_html(sorted(legend_pairs), title="Neighbor → Color")

        # 图例节点：自适应大小
        dot.node("LEGEND", label=legend_label,
                 shape="plaintext", width="0", height="0",
                 margin="0", fixedsize="false")

        # 关键②：把图例放在最右（sink rank）
        with dot.subgraph(name="rank_sink_legend") as s:
            s.attr(rank="sink")
            s.node("LEGEND")

        # 不可见高权重边，把 legend 拉到右边并与节点对齐
        node_ids = [str(n) for n in nodes]
        anchor = "1" if "1" in node_ids else node_ids[-1]
        dot.edge(anchor, "LEGEND", style="invis", weight="20")

    # 渲染
    out_dir = out_path.parent; out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_path.with_suffix(""))
    outputs = []
    dot.format = main_fmt
    outputs.append(str(Path(dot.render(base, cleanup=True))))
    for fmt in extras:
        s = Source(dot.source, engine="dot", format=fmt)
        outputs.append(str(Path(s.render(base, cleanup=True))))
    return outputs

def visualize_all_rules_graphviz(json_path: str,
                                 outdir: str = "plots_gv",
                                 color_attr: str = "p_spread",
                                 width_attr: str = "weight",
                                 also_jpg: bool = True,
                                 dpi: int = 300):
    """
    批量渲染所有规则。默认每张输出 SVG + JPG（白底，高 DPI）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    for item in arr:
        r = item["graph_stats"]["rule_number"]
        title = f"Rule {r}"
        svg_path = Path(outdir) / f"rule_{r:03d}.svg"
        extras = ["jpg"] if also_jpg else None
        _ = draw_rule_graph_graphviz_from_item(
            item, str(svg_path),
            width_attr=width_attr,
            title=title,
            color_by_neighbor=True,  # 邻域上色
            show_legend=True,  # 显示图例
            extra_formats=["jpg"],  # 同时导出 JPG（可选）
            dpi=dpi, bgcolor="white"
        )
    print(f"[OK] saved to {outdir} (SVG{' + JPG' if also_jpg else ''}).")

def visualize_all_rules_graphviz_split(json_path: str,
                                       outdir_svg: str = "plots_svg",
                                       outdir_jpg: str = "plots_jpg",
                                       width_attr: str = "weight",
                                       dpi: int = 300):
    """
    批量渲染：SVG 存到 outdir_svg，JPG 存到 outdir_jpg。
    固定：状态节点水平对齐，图例在右侧。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        arr = json.load(f)

    Path(outdir_svg).mkdir(parents=True, exist_ok=True)
    Path(outdir_jpg).mkdir(parents=True, exist_ok=True)

    for item in arr:
        r = item["graph_stats"]["rule_number"]
        title = f"Rule {r}"

        # 1) SVG
        svg_path = Path(outdir_svg) / f"rule_{r:03d}.svg"
        _ = draw_rule_graph_graphviz_from_item(
            item, str(svg_path),
            width_attr=width_attr,
            title=title,
            color_by_neighbor=True,
            show_legend=True,
            extra_formats=None,   # 这里只出 SVG
            dpi=dpi, bgcolor="white"
        )

        # 2) JPG
        jpg_path = Path(outdir_jpg) / f"rule_{r:03d}.jpg"
        _ = draw_rule_graph_graphviz_from_item(
            item, str(jpg_path),
            width_attr=width_attr,
            title=title,
            color_by_neighbor=True,
            show_legend=True,
            extra_formats=None,   # 这里只出 JPG
            dpi=dpi, bgcolor="white"
        )

    print(f"[OK] SVG → {outdir_svg} | JPG → {outdir_jpg}")

if __name__ == "__main__":
    data = analyze_all_rules()
    visualize_all_rules_graphviz_split(
        "all_eca_rules_graph_representation.json",
        outdir_svg="plots_svg",
        outdir_jpg="plots_jpg",
        width_attr="weight",
        dpi=300
    )
