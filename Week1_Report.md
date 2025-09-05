# 于 ECA 规则的图表示：完整总结（含公式）

> 这份 Markdown 汇总了我们在对 **Wolfram 初等元胞自动机（ECA）** 的**图表示**、**数据结构**、**可视化**、**实验扩展**与**排错**方面做过的所有工作与决策。可直接用于组会汇报与归档。

------

## 1. 背景与目标

- 目标：把每个 **ECA 规则**（二值状态、半径 $r=1$）转成一个**有向多重图**，并导出可供分析/学习/绘图的结构化数据（JSON），同时生成出版质量的图像（SVG/JPG）。
- 图的含义：
  - **顶点（节点）**：中心元胞的状态 $S=\{0,1\}$；
  - **有向边**：给定邻域 $(l,c,r)$ 时，中心由 $c$ 演化到 $v=f(l,c,r)$，记一条 $c\to v$ 的边；
  - **多重性**：不同邻域对同一 $c\to v$ 会形成**平行边**。

------

## 2. 数学定义与图构建

### 2.1 局部更新与规则编码

- 二值 ECA、半径 $r=1$ 时，局部更新为：
  $$
  x_i(t+1)=f\!\big(x_{i-1}(t),\,x_i(t),\,x_{i+1}(t)\big),\quad x_i\in\{0,1\}.
  $$

- Wolfram 规则号 $R\in[0,255]$ 的 8 位二进制：
  $$
  R=(b_7 b_6 b_5 b_4 b_3 b_2 b_1 b_0)_2,\quad b_k\in\{0,1\},
  $$
  与邻域（从 MSB 到 LSB）映射顺序为：
  $$
  \{111,\,110,\,101,\,100,\,011,\,010,\,001,\,000\}.
  $$

### 2.2 规则图（多重图）构造

- 对所有中心 $c\in\{0,1\}$ 和邻居 $(l,r)\in\{0,1\}^2$，令
  $$
  v=f(l,c,r),\qquad \text{添加有向边 }(c\to v),
  $$
  并记录**边标签** $l\mid r$（我们最终**不在图中显示文字**，而是用**颜色图例**代表）。

### 2.3 多重计数与马尔可夫视角

- 记 $m_{uv}$ 为从 $u\to v$ 的**平行边数量**（由不同 $(l,r)$ 触发）。定义：
  $$
  M_R=[m_{uv}],\qquad
  P_R=[p_{uv}],\quad p_{uv}=\frac{m_{uv}}{4}.
  $$

- 出/入度（计多重边）：
  $$
  \deg^+_R(u)=\sum_v m_{uv},\qquad \deg^-_R(v)=\sum_u m_{uv}.
  $$

- 密度（仍按多重边计数）：
  $$
  \rho_R=\frac{|E|}{|V|(|V|-1)}.
  $$
  由于 $|V|=2$、总边数（计多重）为 8，$\rho_R$ 可大于 1，这是**多重图**的自然现象。

### 2.4 GNN 特征打包（边索引与边特征）

- 将图按 `networkx.MultiDiGraph.edges(keys=True)` 的顺序，打包为：
  - `edge_index` $\in \mathbb{N}^{2\times m}$：有向边的 $[u;v]$ 索引；
  - `edge_attr`：半径 1 下，每条边对应扁平化邻域矩阵 $[l,r]$；
  - `rule_ids`：长度为 $m$ 的数组，元素恒为规则号 $R$。

------

## 3. 保存的数据结构（JSON Schema 与含义）

### 3.1 顶层结构

```
{
  "graph_stats": {
    "rule_number": 110,
    "num_nodes": 2,
    "num_edges": 8,
    "density": 4.0,
    "is_strongly_connected": true,
    "max_in_degree": 4,
    "max_out_degree": 4,
    "avg_in_degree": 4.0,
    "avg_out_degree": 4.0
  },
  "rule_table": {
    "(1, 1, 1)": 0,
    "(1, 1, 0)": 1,
    "...": 0
  },
  "rule_binary": "01101110",
  "edge_details": [
    {
      "from_state": 0,
      "to_state": 1,
      "key": 0,
      "full_neighborhood": [1,0,0],
      "neighbor_config": [1,0],
      "neighbor_string": "1|0"

      /* 可选动态字段（来自大规模仿真聚合）：
         "count": 3578,
         "weight": 0.37,
         "p_spread": 0.18,
         "p_recover": 0.62,
         "delta_diff_mean": -0.03 */
    }
  ],
  "gnn_features": {
    "edge_index": [[...],[...]],
    "edge_attr": [[l, r], ...],
    "rule_ids": [110, 110, ...]
  }
}
```

### 3.2 关键字段说明

- `graph_stats`：按多重图统计的**结构量**（节点/边/密度/强连通性/度分布统计）。
- `rule_table`：$(l,c,r)\mapsto f(l,c,r)$ 的映射表（与 Wolfram 编码一致）。
- `rule_binary`：8 位二进制字符串，便于核对。
- `edge_details`：逐边记录（含**邻域文本** `l|r`）；若后续融合动力学统计（如 100 元胞实验），会在此补充 `count` / `weight` / `p_spread` 等字段。
- `gnn_features`：给图学习用的数据打包（索引、边特征、规则 id）。

------

## 4. 可视化设计（Graphviz）

### 4.1 布局与图例（核心约束）

- **节点水平对齐**：固定 `rankdir="LR"`，并用子图 `rank="same"` 强制 `{0,1}` 在同一水平线；

- **图例固定在右侧**：构造 `LEGEND` 为 `shape="plaintext"` 的 HTML 表格节点，放入 `rank="sink"` 子图；再用**不可见高权重**的边把它推到最右；

- **移除边内文字标签**：避免拥挤——**每个邻域 `l|r` 对应一种唯一颜色**，在右侧图例显示；

- **自环分象限**：在 `{ne,se,sw,nw}` 端口轮换，尽量减少重叠；

- **线宽映射（可选）**：若边上有 `weight`，线宽按线性缩放：
  $$
  \mathrm{penwidth}(e)=w_{\min}+(w_{\max}-w_{\min})\cdot
  \mathrm{clip}\!\Big(\frac{w-w_{\min}}{w_{\max}-w_{\min}},\,0,\,1\Big).
  $$

### 4.2 色板与清晰度

- 颜色：优先使用 `tab10/tab20`，若类别>20，改为 HSV 均匀采样；
- 输出：**SVG**（矢量清晰）与 **JPG**（高 DPI）**分目录**保存；
- 画布比例：通过 `nodesep/ranksep/pad/margin/minlen/weight` 调整，让“左右两节点 + 右侧图例”比例**均衡**。

------

## 5. 主要函数（职责要点）

> 这里仅总结职责与参数，不重复大段代码。

### 5.1 `class CAToGraph`

- `__init__(num_states=2, radius=1)`：默认二值、半径 1（本工作假定 Wolfram ECA）。
- `rule_number_to_table(rule_number)`：Wolfram 规则号 $\to$ 邻域表。
- `generate_neighbor_configs()`：列出全部 $(l,r)$ 组合。
- `neighbor_config_to_matrix(config)`：半径 1 时返回 `[[l],[r]]`。
- `build_graph_from_rule_table(rule_table, rule_id)`：生成 `MultiDiGraph`，边属性含 `neighbor_config/neighbor_string/full_neighborhood/rule_id`。
- `build_graph_from_rule_number(rule_number)`：包装器。
- `extract_gnn_features(G)`：打包 `edge_index/edge_attr/rule_ids`。
- `get_graph_representation(rule_number)`：汇总 `graph_stats/rule_table/rule_binary/edge_details/gnn_features`。

### 5.2 分析与导出

- `analyze_all_rules()`：遍历 0–255，打印统计，输出
  - `all_eca_rules_graph_representation.json`
  - `eca_rules_summary.csv`
- `generate_summary_statistics(all_rules_data)`：汇总统计写入 CSV。

### 5.3 重建与绘图

- `graph_from_rule_data(rule_data)`：从 JSON **还原** `MultiDiGraph`（包含可选动态字段）。
- `draw_rule_graph_graphviz_from_item(rule_data, out_path, width_attr="weight", title=None, color_by_neighbor=True, show_legend=True, extra_formats=None, dpi=300, bgcolor="white")`
  - 固定**左右对齐**、**右侧图例**、**邻域配色**、**自环分象限**、**可选线宽映射**；
  - `out_path` 的后缀决定主格式；`extra_formats` 可追加多格式。
- `visualize_all_rules_graphviz_split(json_path, outdir_svg="plots_svg", outdir_jpg="plots_jpg", width_attr="weight", dpi=300)`
  - **批量**渲染：SVG $\to$ `plots_svg/`，JPG $\to$ `plots_jpg/`。

> 另：`draw_rule_graph_networkx` 作为 **预览**方案（非最终展示），Graphviz 方案用于汇报图。

------

## 6. 结果阅读指南（示例与解读）

- **两节点**：左侧 `0`，右侧 `1`；
- **有向彩色弧线**：表示不同邻域触发的跃迁（并行边会自然分离；自环分象限）；
- **右侧图例**：颜色 ↔ `l|r` 映射；
- **线宽**（若启用 `weight`）：代表该跃迁的相对频次/权重。

**极端示例（规则 0）**
 所有邻域输出都为 0：
$$
M_R=
\begin{bmatrix}
4 & 0\\
4 & 0
\end{bmatrix},\quad
P_R=
\begin{bmatrix}
1 & 0\\
1 & 0
\end{bmatrix}.
$$
解读：无论中心是 0 还是 1，都被推向 0；图中 `1→0` 的边会显著占优。

------

## 7. 扩展到 100 元胞实验与分类（与同伴分类结合）

### 7.1 统计增强（仿真 $\to$ 边权）

- 在长度 100 的环或线性边界上，长时间演化、多初态采样，统计每条**局部事件**出现的次数，汇总到边上：
  $$
  \hat{m}_{uv}=\#\{(l,r)\text{ 导致 }u\to v\},\qquad
  \hat{P}_{uv}=\frac{\hat{m}_{uv}}{\sum_{v'} \hat{m}_{uv'}}.
  $$

- 将 `count/weight` 等字段写入 `edge_details`，用于**加粗**频繁跃迁的边、构造经验马尔可夫矩阵 $\hat{P}$。

### 7.2 特征与分类

- **结构特征**：自环数量/强度、是否双向连通、有无两类出邻居、$|E|$、$\deg^\pm$ 分布；
- **谱特征**：$\hat{P}$ 的特征值、谱间隙、稳态分布；
- **动力学特征**：`p_spread`、`p_recover`、`delta_diff_mean` 等；
- **GNN/传统模型**：
  - 简单 **GNN**（2 节点多重图，边特征 = `[l,r]` + 权重）；
  - 或 **树模型**（特征可解释性好、训练快）。
- **与同伴分类结合**：把结构/谱/动力学特征与“有序/混沌/复杂（Class I–IV）”等视觉类别对齐，做交叉验证与稳健性评估。

------

## 8. 已解决的环境与排错问题

- **Graphviz 可执行找不到（Windows）**
   `FileNotFoundError: failed to execute 'dot'`
   → 安装系统版 Graphviz，并把 `dot.exe` 加到 `PATH`；Python 包 `graphviz` 只是封装，**两者都要**。

- **Pango 字体警告（DejaVu Sans）**
   → 方案 A：系统安装 `DejaVuSans.ttf/DejaVuSans-Bold.ttf`；
   → 方案 B（Conda）：

  ```
  conda install -c conda-forge fonts-conda-ecosystem fonts-conda-forge
  ```

  同时我们**不**在 Graphviz 中硬编码 DejaVu；默认字体也能正常渲染。

- **SVG 无法打开/损坏**
   → 使用 `Digraph(...).render(base, cleanup=True)` 输出，不手动改写 `dot.source`（避免 `AttributeError: can't set attribute`）；必要时用 `Source(dot.source, format=...)` 另存。

- **图例尺寸警告（Node size too small）**
   → 将图例设为 `shape=plaintext` 的 HTML 表格，`fixedsize=false`，`width/height=0`，让内容自适应。

- **比例失衡（图太扁/太挤）**
   → 调整 `nodesep/ranksep/pad/margin/minlen/weight/dpi`；固定左右对齐 + 右侧图例，保证观感稳定。

------

## 9. 复现与使用流程

1. **生成全规则 JSON 与汇总 CSV**

   ```
   python GraphRepresent.py     # 或 GraphRepresent_Pure.py
   # 产物：
   #   all_eca_rules_graph_representation.json
   #   eca_rules_summary.csv
   ```

2. **批量渲染（SVG/JPG 分目录）**

   ```
   visualize_all_rules_graphviz_split(
       "all_eca_rules_graph_representation.json",
       outdir_svg="plots_svg",
       outdir_jpg="plots_jpg",
       width_attr="weight",   # 若无该字段，则线宽为默认值
       dpi=300
   )
   ```

   - `plots_svg/`：矢量文件，便于论文/打印；
   - `plots_jpg/`：位图，便于幻灯片与网页。

------

## 10. 后续工作清单（Roadmap）

-  从 100 元胞长时仿真中填充 `count/weight/p_*` 到 `edge_details`；
-  基于 $\hat{P}$ 做谱分析（谱间隙、稳态分布、混合速率）；
-  结构 + 谱 + 动力学特征与同伴分类体系的融合、训练可解释模型；
-  评估初始条件分布变化下的鲁棒性（方差、置信区间）；
-  丰富可视化（批量拼图、对比同一规则在不同统计下的差异）。

------

## 11. 关键参数建议（绘图）

- `rankdir="LR"`、`rank="same"`（节点水平对齐）；
- 图例：`shape="plaintext"`, `fixedsize=false`, `rank="sink"`, 不可见边 `weight` 大、`minlen` 稍大；
- 自环端口：循环 `{ne,se,sw,nw}`；
- 色板：`tab10/tab20`，类别大于 20 时 HSV 采样；
- 位图分辨率：`dpi≥300`；
- 画布间距：适度调大 `nodesep/ranksep/pad/margin`，避免挤压。

------

## 12. 术语表

- **ECA**：初等元胞自动机（binary, radius 1）；
- **多重图**：允许平行边的有向图；
- **邻域标签 `l|r`**：左右邻居（中心 $c$ 由源节点表示）；
- **$M_R$**：多重计数矩阵（平行边计数）；
- **$P_R$**：按行归一的跃迁概率矩阵；
- **谱间隙**：最大与次大特征值模之差，表征混合速度；
- **GNN 特征**：`edge_index/edge_attr/rule_ids`，供图神经网络使用。