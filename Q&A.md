# EN — Spectrum of $P$ (Eigenvalues & Spectral Gap)

**What is $P$?**

- $P$ is the **2×2 Markov transition matrix** built from multiplicities $m_{uv}$ (or weights):
  $$
  P=
  \begin{bmatrix}
  p & 1-p\\
  q & 1-q
  \end{bmatrix},\quad
  p=\frac{m_{00}}{4},\;q=\frac{m_{10}}{4}\quad(\text{ECA: each row sums to }1)
  $$
  If you use simulation weights, replace counts by sums of weights and row-normalize to get $\hat P$.

**Eigenvalues (closed form for 2×2 stochastic matrix).**

- Any row-stochastic $P$ has one eigenvalue **$\lambda_1=1$**.

- The **second eigenvalue** is
  $$
  \lambda_2 = p - q.
  $$
  (It’s real for the 2-state case.)

**Stationary distribution.**

- If the chain is **irreducible & aperiodic** (here: $q>0$ and $1-p>0$), the stationary distribution is
  $$
  \pi=\big(\pi_0,\pi_1\big),\quad
  \pi_0=\frac{q}{\,q+1-p\,},\;\;
  \pi_1=\frac{1-p}{\,q+1-p\,}.
  $$
  Degenerate (reducible) cases still have a stationary distribution (e.g., absorbing), but mixing-time interpretations change.

**Spectral gap (how fast it mixes).**

- Define **SLEM** $=\max_{i\ge2}|\lambda_i|$. For 2-state chains, $\text{SLEM}=|\lambda_2|=|p-q|$.

- The **(absolute) spectral gap** is
  $$
  \gamma \;=\; 1-\text{SLEM}\;=\;1-|p-q|.
  $$
  Larger $\gamma\Rightarrow$ faster convergence to stationarity (heuristic scaling $t_{\text{mix}}\sim 1/\gamma$ when assumptions hold).

**Examples.**

- **Rule 30 (unweighted, equal side-neighborhoods):** we derived
   $P=\begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix} \Rightarrow \lambda_2=0,\; \gamma=1$.
   (Balanced toggling; mixes in one step.)
- **Rule 0:**
   $P=\begin{bmatrix}1&0\\1&0\end{bmatrix} \Rightarrow \lambda_2=0,\; \gamma=1$.
   (Not irreducible; state 0 is absorbing, so you collapse in one step—“fast” for a trivial reason.)

**Notes for weighted $\hat P$.**

- Replace $p$ by $\hat p=\big(\sum \text{weights of }0\!\to\!0\big)/\big(\sum \text{weights out of }0\big)$, similarly for $\hat q$.
- Compute $\hat\lambda_2=\hat p-\hat q$, $\hat\gamma=1-|\hat p-\hat q|$. This reflects **empirical** (simulation-driven) mixing.

------

# ZH — $P$ 的谱（特征值与谱间隙）

**$P$ 是什么？**

- $P$ 就是我们从**多重计数**（或仿真权重）做**按行归一**得到的 **2×2 马尔可夫矩阵**：
  $$
  P=
  \begin{bmatrix}
  p & 1-p\\
  q & 1-q
  \end{bmatrix},\quad
  p=\frac{m_{00}}{4},\;q=\frac{m_{10}}{4}.
  $$
  若用仿真权重，就把计数换成“权重求和”，再按行归一得到 $\hat P$。

**特征值（2×2 行随机矩阵的闭式）。**

- 行随机矩阵必有特征值 **$\lambda_1=1$**；

- 第二个特征值
  $$
  \lambda_2 = p - q.
  $$
  （两状态时一定为实数。）

**平稳分布。**

- 若链 **不可约且非周期**（这里等价于 $q>0$ 且 $1-p>0$），平稳分布为
  $$
  \pi=\big(\pi_0,\pi_1\big),\quad
  \pi_0=\frac{q}{\,q+1-p\,},\;\;
  \pi_1=\frac{1-p}{\,q+1-p\,}.
  $$
  若可约（如有吸收态），仍有平稳分布，但“混合时间”的意义发生变化（通常“一步坍缩”或“多吸收类”）。

**谱间隙（反映收敛快慢）。**

- 定义 **SLEM** $=\max_{i\ge2}|\lambda_i|$。两状态时 $ \text{SLEM}=|\lambda_2|=|p-q|$。

- **（绝对）谱间隙**：
  $$
  \gamma \;=\; 1-|p-q|.
  $$
  $\gamma$ 越大，越“快”接近平稳（在假设成立时 $t_{\text{mix}}\sim 1/\gamma$）。

**例子。**

- **Rule 30（未加权、四种 $(l,r)$ 等可能）**：
   $P=\begin{bmatrix}0.5&0.5\\0.5&0.5\end{bmatrix} \Rightarrow \lambda_2=0,\; \gamma=1$ ——“一步混合”。
- **Rule 0**：
   $P=\begin{bmatrix}1&0\\1&0\end{bmatrix} \Rightarrow \lambda_2=0,\; \gamma=1$ ——**非**不可约；因为“吸收”，所以“一步到位”。

**加权 $\hat P$ 的用法。**

- 用仿真统计替换 $p,q$ 得 $\hat p,\hat q$，再算 $\hat\lambda_2=\hat p-\hat q$、$\hat\gamma=1-|\hat p-\hat q|$，即可反映**经验转移**的“稳定/混合”程度。

# EN — Density Nuances (what exactly are we reporting?)

**Definitions.**
 Let $n=|V|$ (number of nodes), $m=|E|$ (number of directed edges, counting parallel edges). In our NX build we create a **MultiDiGraph** and add one edge per side-neighborhood event; this is why parallel edges exist.   Also, for ECA with radius $r=1$, the side-neighborhood list has **4** combos per center state. 

**NetworkX variant (MultiDiGraph).**
 We compute density via `nx.density(G)`, which uses $m/(n(n-1))$ for directed graphs. The denominator counts only inter-node pairs (loops not in the denominator), but the numerator **does count** all edges (including parallel edges and self-loops), so the value can exceed 1. In code: `'density': nx.density(graph)` in the NX pipeline. 
 → For ECA $k=2,r=1$, $n=2$ and we always add 8 event-edges in total, so $m=8$ and $m/(n(n-1))=8/2=4$ (constant across rules; not discriminative, only reflects “multi-edge intensity”). (Reason: we loop 2 centers × 4 side neighborhoods and add an edge each time.  )

**Pure variant (no NetworkX object).**
 Here we intentionally avoid the directed simple-graph denominator and keep a **simple ratio**
$$
\text{density}=\frac{E}{N\cdot N}
$$
as a bounded proxy “in $[0,1]$” per the inline comment. Implementation: compute `num_edges/float(num_nodes * num_nodes)`.  
 ⚠️ **Nuance:** with multiplicity, $E$ can exceed $N\!\cdot\!N$ (e.g., $8/4=2$ for ECA), so this proxy is **not strictly bounded** in our current multigraph setting; it only stays within $[0,1]$ if edges are de-duplicated.

**Takeaways.**

- NX density = $m/(n(n-1))$ on a MultiDiGraph → can be $>1$ and is **constant (4)** for ECA $r=1$. 
- Pure density = $E/(N^2)$ → intended to be bounded, but **is 2** under multiplicity for ECA; thus also uninformative across rules. 

**Recommendation (what to show on slides).**
 Report density with **labels** and add a genuinely bounded, interpretable pair of diagnostics:

1. **Reachability density (unique pairs):**

$$
d_{\text{reach}}=\frac{|\{(u,v):u\neq v,\ m_{uv}>0\}|}{n(n-1)}\in[0,1]
$$

(ignores multiplicity; asks “which ordered pairs are reachable at all?”)

2) **Non-loop share (with multiplicity):**

$$
\phi_{\text{cross}}=\frac{\sum_{u\neq v}m_{uv}}{k^{2r+1}}\quad\text{(ECA: divide by 8)}
$$

equivalently $1-$ **loop share** $\phi_{\text{loop}}=\frac{\sum_{u}m_{uu}}{8}$. These are bounded and intuitive (“how often do local events change the state?”).

------

# ZH — 密度口径说明（到底在报什么）

**术语。**
 $n=|V|$（节点数），$m=|E|$（边数，**计并行边**）。NX 构图是 **MultiDiGraph**，每个侧邻域事件都“落一条边”，因此天然有平行边。  半径 $r=1$ 下，每个中心的侧邻域组合为 **4** 个。

**NX 口径（MultiDiGraph）。**
 使用 `nx.density(G)`，公式为 $m/(n(n-1))$。分母只按“跨节点对”，**不包含自环**；分子里的 $m$ 会把**平行边与自环**都计入，因此密度可以大于 1。实现：`'density': nx.density(graph)`。
 → 对 ECA $k=2,r=1$：$n=2$，总是添 8 条事件边，$m=8$，故密度恒为 $8/2=4$（随规则不变，只体现“多重边强度”）。 

**Pure 口径（非 NX 对象）。**
 实现中用简化比值
$$
\text{density}=\frac{E}{N^2}
$$
并在注释里说明“保持在 $[0,1]$”。代码：`num_edges/float(num_nodes * num_nodes)`。 
 ⚠️ **注意：\**在存在多重边时，$E$ 可能超过 $N^2$（ECA 为 $8/4=2$），因此这一口径\**并非严格有界**；只有在“去重边”的情况下才会 $\le1$。

**结论。**

- NX 密度 $m/(n(n-1))$（MultiDiGraph）→ 可能 $>1$，且在 ECA $r=1$ 下恒为 **4**。
- Pure 密度 $E/(N^2)$ → 设计初衷是有界，但在当前“计多重”的实现中为 **2**，同样不区分规则。

**建议（报告中怎么放）。**
 在表格里同时列出 **“NX density (MultiDiGraph)”** 与 **“Pure density (edges/N²)”**，并补充两个**有界且可解释**的指标：

1. **可达密度（去重口径）**

$$
d_{\text{reach}}=\frac{|\{(u,v):u\neq v,\ m_{uv}>0\}|}{n(n-1)}\in[0,1]
$$

1. **跨状态份额（计多重）**

$$
\phi_{\text{cross}}=\frac{\sum_{u\neq v}m_{uv}}{8},\quad
\phi_{\text{loop}}=1-\phi_{\text{cross}}=\frac{\sum_{u}m_{uu}}{8}
$$

（ECA 下 8 为总事件数），直观表征“局部事件有多大比例会让中心**改变**状态”

在 ECA（半径 r=1）里，每个位置的局部环境是三元组 (l, u, r)。把中心 $u$ 固定后，左右两侧就只有 **4** 种组合：
$$
(l,r)\in\{(0,0),(0,1),(1,0),(1,1)\}.
$$
“等概率抽样”就是把这四种 $(l,r)$ 视为**各占 1/4** 的局部情形。

下面以 **Rule 30** 为例逐一列举（Wolfram 编码次序：111,110,101,100,011,010,001,000；Rule 30 的输出为 00011110）：

### 当中心 $u=0$

- (l,r) = **(0,0)**：三元组 (0,0,0) = **000** → $v=0$ ⇒ **0→0**，标签 `0|0`
- (l,r) = **(0,1)**：三元组 (0,0,1) = **001** → $v=1$ ⇒ **0→1**，标签 `0|1`
- (l,r) = **(1,0)**：三元组 (1,0,0) = **100** → $v=1$ ⇒ **0→1**，标签 `1|0`
- (l,r) = **(1,1)**：三元组 (1,0,1) = **101** → $v=0$ ⇒ **0→0**，标签 `1|1`

结论：从 $u=0$ 出发，**0→0 有 2 次**（(00),(11)），**0→1 有 2 次**（(01),(10)）。

### 当中心 $u=1$

- (l,r) = **(0,0)**：三元组 (0,1,0) = **010** → $v=1$ ⇒ **1→1**，标签 `0|0`
- (l,r) = **(0,1)**：三元组 (0,1,1) = **011** → $v=1$ ⇒ **1→1**，标签 `0|1`
- (l,r) = **(1,0)**：三元组 (1,1,0) = **110** → $v=0$ ⇒ **1→0**，标签 `1|0`
- (l,r) = **(1,1)**：三元组 (1,1,1) = **111** → $v=0$ ⇒ **1→0**，标签 `1|1`

结论：从 $u=1$ 出发，**1→1 有 2 次**（(00),(01)），**1→0 有 2 次**（(10),(11)）。

### 对应的多重计数矩阵 $M$ 与马尔可夫矩阵 $P$

$$
M=\begin{bmatrix}
m_{0\to0} & m_{0\to1}\\
m_{1\to0} & m_{1\to1}
\end{bmatrix}
=
\begin{bmatrix}
2 & 2\\
2 & 2
\end{bmatrix},
\qquad
P=\frac{1}{4}M=
\begin{bmatrix}
0.5 & 0.5\\
0.5 & 0.5
\end{bmatrix}.
$$

> 小提示：这是在“**四种 (l,r) 等可能**”假设下得到的局部随机近似。在真实长时仿真里，各种 (l,r) 的出现频率未必相等；此时可把每条边的经验频次（如 `weight`）按行归一，得到加权版 $\hat P$

1. **理论（组合）版**：只根据规则表 $f(l,u,r)$ 统计并行边数就能得到 $M$（每行和=4），再行归一得 $P$。完全不依赖仿真，这是规则自身的“静态结构特征”。
2. **经验（加权）版**：做 100-cell 等长时仿真，把每次局部事件的出现频次记为权重，得到 $\hat M$、$\hat P$。它反映规则在**真实演化**下的“动力学特征”（受初态分布、边界条件、时间长度影响）。

这两者都**能用于研究与归类规则**，但关注点不同，互补使用更稳妥：

------

### 怎么理解与使用

- **静态版 $M,P$（无需仿真）**
  - 构造：对每个源状态 $u\in\{0,1\}$ 枚举四个 $(l,r)$，把落到同一 $v$ 的次数累加成 $m_{uv}$，得
     $M=\begin{bmatrix}m_{00}&m_{01}\\ m_{10}&m_{11}\end{bmatrix}$，行和=4；
     $P=M/4$。
  - 作用：快速看出**自环/跨跃迁**的比例、是否“偏向某一状态”、以及
     $\lambda_2=p-q$（其中 $p=m_{00}/4,\ q=m_{10}/4$）等谱量。
  - 局限：有时会把“吸收/可约”与“快速混合”在数值上混淆（例如 Rule 0 的 $\gamma=1$ 源自吸收而非真正混合）。因此用它做**粗分型**可以，但结论要结合可达性/强连通性等一起看。
- **经验版 $\hat M,\hat P$（来自 100-cell 仿真）**
  - 构造：把每条边的频次或概率写到 `weight`，再对每行做归一化：
     $\hat p_{uv}=\dfrac{\sum_{e:u\to v} w_e}{\sum_{v'}\sum_{e:u\to v'} w_e}$。
  - 作用：刻画**实际演化**下的偏好与稳定性，提取更有判别力的特征：
    - **跨状态份额** $\phi_{\text{cross}}=\frac{\sum_{u\neq v}\hat m_{uv}}{\sum_{u,v}\hat m_{uv}}$（=1−自环份额），
    - **行熵** $H(u)=-\sum_v \hat p_{uv}\log \hat p_{uv}$，
    - **谱间隙** $\hat\gamma=1-|\hat p-\hat q|$（两状态时），
    - **平稳分布偏置** $\hat\pi$ 及其对初态的鲁棒性（多次种子取平均并给出方差）。
  - 优点：能区分“看起来结构相似但动力学差异显著”的规则（例如某些 Class II vs Class IV）。

------

### 推荐的归类工作流（简要）

1. **每个规则**：先算静态 $M,P$（无仿真），记录：自环/跨跃迁比例、可达对数、强连通性、$\lambda_2,\gamma$。
2. **多初态、多轮次仿真**：累计出 $\hat M,\hat P$ 与其不确定度（均值±方差），提取上面那些动态特征。
3. **与视觉/汉明距离分类对齐**：用树模型或简单 GNN 做融合特征分类，检验与 Wolfram Class I–IV 的一致性与鲁棒性。

> 小结：**是的**，$M$ 与 $P$（或其加权版本 $\hat M,\hat P$）都可以作为“规则指纹”。静态版提供**规则结构**，经验版提供**演化动力学**；两者结合，最有利于对规则进行可靠归类与解释。

# 它们各自“是什么”？

**NX = NetworkX 版本（`GraphRepresent.py`）**

- 用 **NetworkX 的 `MultiDiGraph`** 真正建图：每个局部邻域事件落一条边（允许平行边、自环）【】【】。
- 直接用 NX 的图算法算统计量：**density**、**strong connectivity** 等【】。
- GNN 特征从 NX 图里抽取（`edges(keys=True)` → `edge_index/edge_attr/rule_ids`）【】。
- 也提供 Graphviz 绘图（从 `edge_details` 复原并渲染）【】【】。

**Pure = 纯 Python 版本（`GraphRepresent_Pure.py`）**

- **不用 NetworkX**，而是自己维护 `{nodes: [...], edges: [...]}` 的轻量“图对象”，每条边是字典，包含 `from_state/to_state/neighbor_*` 等【】【】。
- 统计量**手动**计算：例如把 **density** 记为 `E/(N*N)`，并把 `is_strongly_connected` 置为 `None`（因为没跑 NX 算法）【】【】。
- 也同样导出 `edge_index/edge_attr/rule_ids`（从自建的 edges 列表抽取）【】。
- Graphviz 绘图直接用 `edge_details` 渲染，和 NX 版思路一致（右侧图例、节点同 rank、自环分象限、线宽映射等）【】【】。

------

# “方法”各自怎么做？

- **建图方式**
  - NX：`nx.MultiDiGraph()`，`G.add_edge(...)` 写入边属性（邻域矩阵、`neighbor_string`、`full_neighborhood`、`rule_id`）【】。
  - Pure：自己 append 到 `edges` 列表，字段与 NX 版本保持一致，方便之后统一导出与绘图【】。
- **统计与特征**
  - NX：`nx.density(G)`、`nx.is_strongly_connected(G)`、`in_degree/out_degree` 等由库直接给出【】【】；`edges(keys=True)` 抽取 GNN 特征【】。
  - Pure：`density = E/(N*N)`（代码注释里说“保持在 [0,1] 的简单比值”，但在多重边语境下其实可能>1），其余度统计手算；SCC 留空【】【】。
- **绘图**
  - 两者在 Graphviz 出图上的**布局约定完全一致**：LR 水平对齐、右侧图例、邻域配色、自环四象限、线宽按 `weight` 线性映射等（NX 与 Pure 的 Graphviz 函数实现几乎同构）【】【】【】。

------

# 主要区别（总结表）

- **依赖与体量**
  - NX：依赖 NetworkX，功能全面；适合要跑图算法（SCC、最短路、连通性等）的场景【】。
  - Pure：零依赖（去掉 NX），数据结构更轻，易于嵌入最小环境或批处理脚本【】。
- **图算法能力**
  - NX：直接得到“强连通”“密度（NX 口径）”等标准图论量【】。
  - Pure：不带这些算法，`is_strongly_connected=None`，需要你自己补【】。
- **密度口径**
  - NX：`nx.density`（有向图）= $m/(n(n-1))$，**多重边会让值>1**；在 ECA(2节点、8条边) 下恒为 4，区分度有限【】。
  - Pure：用 `E/(N*N)` 的**简化比率**（注释声称“bounded in [0,1]”，但多重边下并不严格有界）【】。
- **数据导出与 GNN 对接**
  - 两者都输出同一套 JSON 结构与 GNN 特征；Pure 里还显式把 `key` 设 `0` 作为 MultiDiGraph 的占位（因为不使用 NX 的 edge key 概念）【】。
- **可移植性与速度**
  - NX：更“重”，但省时省心（少写代码，少踩坑）。
  - Pure：更“轻”，启动快、依赖少，更适合无图算法需求的大批量流水线（例如只想快速产 JSON/CSV/图）。

------

# 选用建议（什么时候用哪个？）

- **需要图算法或快速验证结构性质**（强连通、度分布、后续要加 PageRank/谱分解等）→ **用 NX 版**，因为一行就能拿到想要的指标【】。
- **只想要规则→JSON→Graphviz 出图→GNN 特征** 的“瘦身流水线”，并尽量减少依赖 → **用 Pure 版**（统计可自定义，可控且可测试）【】【】。
- 实际项目里，你们**两个版本并存**是有意义的：
  - 用 **NX 版**跑基准与校验（有标准答案的图论量）；
  - 用 **Pure 版**做规模化导出与部署（轻量、稳定）。