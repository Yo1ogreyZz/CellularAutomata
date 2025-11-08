# A Taxonomic Review of Graph-Based Representations for Cellular Automata

## I. Introduction: From Local Rules to Global Networks

A central challenge in cellular automata is to explain how **local, discrete, deterministic** update rules give rise to **global, complex, often surprising** spatiotemporal behavior [1–3]. To bridge this gap, researchers have developed families of representations that map different aspects of CA into **graphs/networks**, enabling the use of graph theory and network science to analyze structure and dynamics [1–4].
 This review organizes existing and extended approaches into a **taxonomy based on what the graph’s nodes and edges represent**. We distinguish four primary classes—**rule space**, **state space**, **emergent structures**, and **causal constructions**—and treat graph-cellular automata as a model **generalization** rather than a mere representation. We also introduce a **meta-analysis** layer: applying network-science measures to the graphs themselves.

------

## II. A Foundational Taxonomy: What Does the Graph Represent?

### Class 1 — **Rule-Space Representations (Static Blueprints)**

Graphs in this class encode the **local transition function** itself, independently of any initial condition or trajectory.

#### 1.1 De Bruijn Graphs (dBG): The Standard Encapsulation of Local Syntax

**Construction.** For a 1D CA with radius $r$, nodes are all length-$2r$ neighborhood strings. A directed edge $u\to v$ exists iff the last $2r-1$ symbols of $u$ match the first $2r-1$ of $v$; each edge corresponds to a length-$2r+1$ neighborhood and is **labeled** by the rule output for the center cell [4–7]. The dBG thus realizes the rule as a **finite-state transducer** and presents the set of locally admissible contexts as a regular language [4–7].
 **What it buys you.** De Bruijn graphs support **algorithmic decisions** about global maps—e.g., **surjectivity** (existence of “Garden of Eden” configurations) and **injectivity/reversibility**—especially powerful for linear CA via algebra over characteristic polynomials [4–7].
 **Broader footprint.** The dBG is a common currency across computing and biology (assembly graphs, succinct dynamics, etc.), with deep links to symbolic dynamics and information theory [6,8–12].

#### 1.2 Derived Analysis Graphs: Pair, Subset, and (Internal) Cycle Graphs

Built **on top of** the dBG to prove global properties [4]:

- **Pair Graphs.** Nodes are ordered pairs of dBG vertices $(v,w)$. If $v\to x$ and $w\to y$ are edges with the **same output label**, then $(v,w)\to(x,y)$. This **synchronizes** two walks to reason about reversibility and preimage multiplicity [4].
- **Subset Graphs.** Nodes are elements of the **power set** of dBG nodes (the standard NFA→DFA subset construction). Under a fixed output symbol, subsets transition according to possible preimages, making **surjectivity/Garden-of-Eden** reasoning effective [4].
- **Cycle Graphs (in this context).** Analysis of **cycles within** the dBG/derived graphs (not to be confused with global state-transition cycles), relevant to periodic word structure induced by the local rule [4].
   **Duality with symbolic dynamics.** The dBG presents the **local grammar** of allowable space-time words (sofic shifts), while the state-transition graph (below) presents the **global action** on configurations; together they form complementary “syntax versus dynamics” views of the same system [13–17].

------

### Class 2 — **State-Space Representations (Global Dynamics Maps)**

Nodes are **entire configurations**; edges apply the global map $\Phi$.

#### 2.1 State-Transition Graphs (STG) and Basins of Attraction

**Definition.** For a CA with $N$ cells and $k$ states per cell, each of the $k^N$ configurations is a node. If $\Phi(C_t)=C_{t+1}$, add a directed edge $C_t\to C_{t+1}$ [18–21].
 **Structure.** Each component contains one or more **attractor cycles** and in-trees of **transients**, forming **basins**; their union is the **basin-of-attraction (BoA) field**. STGs are the **ground truth** for long-run behavior and often correlate with Wolfram classes (e.g., simple fixed-point attractors vs. intricate, tangled fields) [18–21].
 **Feasibility and tooling.** Because $k^N$ scales exponentially, direct enumeration is limited. Wuensche’s **DDLab** develops inverse algorithms and visual analytics to systematically probe BoA fields and STG structure at modest scales [22–26].

#### 2.2 Perturbation Networks (“Jump-Graphs”)

A **coarse graining** of the STG at the level of attractors/basins: nodes are **attractors (or basins)**; add a directed edge $A\to B$ if a small perturbation (e.g., single-bit flip) to a state in $A$ leads, after relaxation, to $B$. Edge weights capture the transition probabilities under a perturbation model [24–29].
 This exposes the **meta-dynamics** among long-run behaviors—robustness vs. sensitivity to noise. In DDLab one finds exact f-jumps from full BoA fields and approximate h-jumps from space-time sampling [24–26].
 **Cryptographic lens.** STG topology translates directly into PRNG/stream-cipher quality: **short cycles** are catastrophic; **multiple attractors** imply seed-dependent quality. The ideal is a **single, extremely long cycle** attracting almost all states [30–33]; algebra of linear/hybrid CA and characteristic polynomials is crucial for construction and testing [30–34].

------

### Class 3 — **Representations of Emergent Structures (The Physics of Computation)**

Nodes/edges represent **persistent, interacting space-time structures** rather than cells or configurations.

#### 3.1 Subsystem/Module Graphs: From Particle Catalogs to Reaction Graphs

In systems like Conway’s Life, gliders, guns, and still lifes are treated as **units** (nodes), and their **interactions** (collisions, annihilations, scattering) define edges—yielding a high-level “**particle physics**” of the CA [35–38]. This supports reasoning about **computation and logic** without reverting to pixel-level evolution [36–38].

#### 3.2 Formalization via Computational Mechanics: Domains, Filters, Particles

The computational-mechanics program (Crutchfield, Hanson, Shalizi) provides a rigorous pipeline:

1. Identify **domains**—periodic backgrounds describable by regular languages;
2. Build **domain filters** to remove backgrounds and reveal propagating **defects/particles**;
3. Catalog **particle interactions** and assemble an interaction graph—the system’s emergent **computational syntax** [39–44].
    The Rule-54 case study demonstrates a full domain–particle–interaction characterization and the resulting computational capability [42,43].

------

### Class 4 — **Causal Constructions (Information Flow)**

#### 4.1 Dependency Graphs (Parallel-Computation View)

Nodes are **computational tasks** (e.g., computing cell $i$ at time $t+1$); edges encode **data dependencies** from neighbors at time $t$. This DAG captures **potential information flow** and the available **parallelism**, emphasizing computational structure rather than long-term dynamics [45–48].

#### 4.2 Causal Networks (Event-Level Causality)

In Wolfram’s NKS framework, **nodes are update events**; draw an edge A→B iff A’s output is an input to B. The underlying lattice is abstracted away—only the **actual causal history** of a run remains—especially powerful for **asynchronous** or mobile automata without a global clock [49–52].

#### 4.3 $\epsilon$-Machines (Causal Architecture)

In computational mechanics, nodes are **causal states**—equivalence classes of pasts with identical predictive distributions over futures—and edges are symbol-emitting transitions among them. Reconstructing an $\epsilon$-machine from data yields the **minimal predictive model** and quantitative measures like **statistical complexity**; this has been applied to CA for structure discovery and rule reconstruction [39,53–56].
 **Hierarchy recap.** Dependency graphs capture **computational logic** (potential flows), causal networks capture **physical causality** (realized event chains), and $\epsilon$-machines capture **predictive causality** (information equivalence).

------

## III. Generalizations and Extensions

### 3.1 Graph Cellular Automata (GCA) and Learning-Based Rules

GCA do not **map** a lattice CA to a graph; they **define the CA on an arbitrary graph** $G=(V,E)$: nodes are cells; edges define neighborhoods [57–60]. This liberates CA from lattice regularity (Cayley graphs, random/dynamic graphs). Recent work merges GCA with deep learning—**Graph Neural Cellular Automata (GNCA)**, equivariant GNCA, and **developmental GCA**—where a GNN **parametrizes the local rule** and is trained to produce desired emergent behavior [57–60].

### 3.2 Spatial Pattern Analysis (“Color/Cell-Lattice Graphs”)

Treat the lattice as a graph with **node attributes** (cell states). This connects CA to image analysis, spatial statistics, and percolation; one can quantify spatial clustering, morphology, and geometry in 2D or multi-state CAs [61–63].

### 3.3 Non-Classical Automata

- **Asynchronous CA.** Without global synchrony, evolution becomes **nondeterministic branching** in state space; the STG is no longer a function but a relation, and analysis must account for update schedules and distributions [64–66].
- **Quantum CA (QCA).** Local **unitary** dynamics on graphs/lattices; **index theory** furnishes invariants (“quantum information flow”) to classify dynamics, paralleling topological viewpoints [67–71].

------

## IV. A New Layer: Network-Science Meta-Analysis of CA Graphs

Once you have a CA graph (STG, causal network, dBG, …), you can **quantify** it using network science:

- **Community Structure.** Reveals tightly knit dynamical regions in STGs or functional modules in causal networks.
- **Centralities.** Identify “gateway” configurations or pivotal causal events.
- **Motifs.** Over-represented small subgraphs are building blocks of causal logic or rule grammar [72–74].
   This yields a pipeline: **dynamics → graph → metrics → macroscopic behavior**, enabling comparative complexity and predictability studies across rules and models [72–74].

------

## V. Synthesis and Comparative Analysis

**Is the list complete?**
 The seven classic methods form a solid core, but a complete map adds:

1. **Causality/information-flow** representations (dependency graphs, causal networks, $\epsilon$-machines);
2. **Network-science meta-analysis** of any CA-derived graph.
    Also, **GCA** should be treated as a **model generalization** (CA on graphs), not just another representation.

### Comparative Table

| **Method**                               | **Taxonomic Class** | **Nodes**                                        | **Edges**                                        | **Primary Aim**                                              | **Key Strengths**                       | **Limits / Scale**                                 |
| ---------------------------------------- | ------------------- | ------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------- | -------------------------------------------------- |
| **De Bruijn Graph**                      | Rule space          | Overlapping local contexts (length $2r$)         | Length $2r{+}1$ neighborhoods labeled by outputs | Analyze rule-intrinsic properties (e.g., surjectivity, reversibility) | Formal, automata-theoretic, algorithmic | Does not directly show global dynamics             |
| **State-Transition Graph / Basin Field** | State space         | Global configurations (length $N$)               | Global synchronous update $\Phi$                 | Full map of global dynamics; identify attractors             | “Ground truth” of long-run behavior     | $k^N$ growth; only feasible for small $N$          |
| **Jump-Graph (Perturbation Network)**    | State space         | Attractors or basins                             | Transitions between basins under perturbations   | Robustness/sensitivity to noise; meta-dynamics               | Effective coarse-graining of STG        | Depends on perturbation model                      |
| **Subsystem / Module Graph**             | Emergent structures | Persistent patterns (gliders, guns, still lifes) | Possible interactions (collisions/reactions)     | High-level “particle physics” of CA computing                | Intuitive, captures emergent logic      | Requires pattern discovery/annotation              |
| **Pair / Subset (Derived) Graphs**       | Rule space          | dBG vertex pairs / subsets                       | Synchronized transitions of pairs / subsets      | Proof techniques (reversibility, surjectivity)               | Powerful formal analysis                | Higher-order tools, not standalone representations |
| **Color / Cell-Lattice Graph**           | Generalization      | Cells (as graph nodes)                           | Spatial adjacency; states as attributes          | 2D+ spatial pattern analysis & image-like metrics            | Leverages spatial statistics            | Focuses on space more than time                    |
| **Graph Cellular Automata (GCA)**        | Generalization      | “Cells” on arbitrary graphs                      | Arbitrary neighborhood relations                 | Extend CA beyond lattices                                    | Great modeling flexibility; GNCA        | Changes the classical CA definition                |
| **Dependency Graph**                     | Causal construction | Computation tasks at $t{+}1$                     | Dataflow from neighbors at $t$                   | Expose parallelism & information flow                        | Clear computational structure           | Not a long-run dynamics map                        |
| **Causal Network**                       | Causal construction | Update events                                    | Direct causal links among events                 | Actual causal history (esp. asynchronous)                    | Fundamental, lattice-free causality     | Can be large/complex                               |
| **$\epsilon$-Machine**                   | Causal construction | Causal states (predictive equivalence classes)   | Symbol-emitting transitions                      | Minimal predictive model & complexity                        | Reveals hidden predictive structure     | Reconstruction is nontrivial                       |
| **Network-Science Analysis**             | Meta-analysis       | (Depends on target graph)                        | (Depends on target graph)                        | Quantify structure (community, centrality, motifs)           | Adds comparable metrics of complexity   | Post-hoc analysis, not a representation            |

------

## VI. Conclusions: A Unified “CA → Graph → Metrics → Behavior” Map

Organizing by **what graphs represent** yields a coherent taxonomy: **rule space**, **state space**, **emergent structure**, and **causal construction**, augmented by **GCA** (model generalization) and **network-science meta-analysis**. This framework highlights key bridges:

- **dBG ↔ STG** (syntax ↔ dynamics);
- **STG topology ↔ cryptographic quality**;
- **Dependency → Causal network → $\epsilon$-machine** (increasing causal abstraction);
- **Computational mechanics** for domain–particle–interaction structure.
   Looking forward, **learning-based GNCA** promises data-driven sculpting of local rules [57–60], while the combination of **$\epsilon$-machine/Local Causal States** with **network metrics** offers a reproducible stack for structure discovery and comparative complexity [39,41,55,72–74].

------

## References

[1] Cellular Automata - Stanford Encyclopedia of Philosophy. https://plato.stanford.edu
 [2] Twenty Problems in the Theory of Cellular Automata - Wolfram. https://content.wolfram.com
 [3] A Survey on Cellular Automata (cs.unibo.it). https://cs.unibo.it
 [4] Graphs Related to Reversibility and Complexity in Cellular Automata - UAEH. https://uaeh.edu.mx
 [5] De Bruijn Graphs and Linear Cellular Automata - Wolfram. https://wpmedia.wolfram.com
 [6] A Comprehensive Review of the de Bruijn Graph and Its Interdisciplinary Applications in Computing. https://espublisher.com
 [7] De Bruijn Graphs and Linear Cellular Automata by Klaus Sutner - Complex Systems. https://complex-systems.com
 [8] The De Bruijn Mapping Problem with Changes in the Graph - bioRxiv. https://biorxiv.org
 [9] Succinct dynamic de Bruijn graphs | Bioinformatics - Oxford Academic. https://academic.oup.com
 [10] De Bruijn Graph assembly (cs.jhu.edu). https://cs.jhu.edu
 [11] Velvet: Algorithms for de novo short read assembly using de Bruijn graphs - PMC. https://pmc.ncbi.nlm.nih.gov
 [12] De Bruijn graphs and entropy at finite scales | Climenhaga’s Math Blog. https://vaughnclimenhaga.wordpress.com
 [13] Symbolic dynamics - Scholarpedia. https://scholarpedia.org
 [14] DYNAMICS OF SOFIC SHIFTS - Dialnet. https://dialnet.unirioja.es
 [15] Symbolic dynamics (U. Marne-la-Vallée). https://www-igm.univ-mlv.fr
 [16] Symbolic dynamics - ResearchGate (review). https://researchgate.net
 [17] Limit sets of stable Cellular Automata - arXiv:1301.3790. https://arxiv.org
 [18] The Global Dynamics of Cellular Automata (users.sussex.ac.uk). https://users.sussex.ac.uk
 [19] Cellular Automaton State Transition Diagrams - Wolfram Demonstrations. https://demonstrations.wolfram.com
 [20] Interactive Graph Visualization in DDLab - arXiv. https://arxiv.org
 [21] EXPLORING DISCRETE DYNAMICS (DDLab). https://ddlab.org
 [22] Discrete Dynamics Lab (site A). https://ddlab.com
 [23] Discrete Dynamics Lab (site B). https://ddlab.org
 [24] Basins of attraction in network dynamics (DDLab page). https://ddlab.org
 [25] The DDLab screen showing two layouts of a basin of attraction field - ResearchGate. https://researchgate.net
 [26] The basin of attraction field of a multi-value v=3 n=6, k=3 CA - ResearchGate. https://researchgate.net
 [27] Simple Networks on Complex Cellular Automata: From de … - ResearchGate. https://researchgate.net
 [28] The space-time pattern of a 1D complex cellular automaton … - ResearchGate. https://researchgate.net
 [29] Network View of Binary Cellular Automata - ResearchGate. https://researchgate.net
 [30] Cellular Automaton-Based Pseudorandom Number Generator | Wolfram. https://content.wolfram.com
 [31] Random Sequence Generation by Cellular Automata | Wolfram. https://content.wolfram.com
 [32] Exploring Hybrid Cellular Automata (HCA) for cryptographic applications - UPIT. https://upit.ro
 [33] Cryptographic Algorithm Based on Hybrid One-Dimensional Cellular Automata - MDPI. https://mdpi.com
 [34] Algebraic Theory of Bounded One-dimensional Cellular Automata - Duke CS. https://users.cs.duke.edu
 [35] A “Glider Gun” in the Game of Life - Stanford CS. https://cs.stanford.edu
 [36] Digital Logic Gates on Conway’s Game of Life - Nicholas Carlini. https://nicholas.carlini.com
 [37] Lexicon - John Conway’s Game of Life. https://playgameoflife.com
 [38] Cellular Automata, Emergent Phenomena in (uu.nl). https://uu.nl
 [39] Computational Mechanics of Cellular Automata: An Example | SFI. https://santafe.edu
 [40] Cosma Shalizi’s Thesis (bactra.org). https://bactra.org
 [41] Local Causal States and Discrete Coherent Structures | *Chaos* (AIP). https://pubs.aip.org
 [42] Complete Characterization of Structure of Rule 54 - ResearchGate. https://researchgate.net
 [43] Computational mechanics of cellular automata: An example for *Physica D* - IBM Research. https://research.ibm.com
 [44] The Evolutionary Design of Collective Computation in … (SFI/AWS). https://sfi-edu.s3.amazonaws.com
 [45] Optimizing Parallel Computing: Mastering Task Dependency Graphs … (ones.com). https://ones.com
 [46] Amdahl’s Law and Task Dependency Graph … (Medium). https://medium.com
 [47] Lecture 4: Principles of Parallel Algorithm Design (part 1) (ND). https://www3.nd.edu
 [48] Parallel dynamical systems over special digraph classes - ResearchGate. https://researchgate.net
 [49] Causal Network Generated by a Mobile Automaton - Wolfram Demonstrations. https://demonstrations.wolfram.com
 [50] Time and Causal Networks: *A New Kind of Science* (online) - Wolfram. https://wolframscience.com
 [51] Causal Graph — Wolfram MathWorld. https://mathworld.wolfram.com
 [52] *A New Kind of Science* — YouTube lecture. https://youtube.com
 [53] Software in the natural world … - arXiv. https://arxiv.org
 [54] Reconstruction of Epsilon-Machines in Predictive Frameworks - arXiv/website. https://arxiv.org
 [55] Local Causal States and Discrete Coherent Structures - arXiv. https://arxiv.org
 [56] Reconstructing cellular automata rules from observations at nonconsecutive times - APS. https://link.aps.org
 [57] Learning Graph Cellular Automata - NeurIPS Proceedings. https://proceedings.neurips.cc
 [58] Cellular Automata on Graphs: ER graphs evolved toward low-entropy dynamics - MDPI. https://mdpi.com
 [59] Graph Cellular Automata with Relation-Based Neighbourhoods … - MDPI. https://mdpi.com
 [60] Developmental Graph Cellular Automata - MIT Press Direct. https://direct.mit.edu
 [61] Mining Frequent Patterns in 2D+t Grid Graphs for CA Analysis - CNRS. https://perso.liris.cnrs.fr
 [62] Graphs, Topology and the Game of Life - Eoin Davey. https://vey.ie
 [63] *Game of Life Cellular Automata* - ResearchGate (book). https://researchgate.net
 [64] Asynchronous cellular automaton - Wikipedia. https://en.wikipedia.org
 [65] Chapter 7 - Asynchronous Automata - l’IRIF. https://irif.fr
 [66] Asynchronous, finite dynamical systems - ResearchGate. https://researchgate.net
 [67] A graph-theoretic approach to quantum cellular design and analysis - AIP. https://pubs.aip.org
 [68] Index theory of 1D quantum walks and cellular automata - arXiv/ETHZ. https://arxiv.org
 [69] An overview of quantum cellular automata - Semantic Scholar (indexed). https://semanticscholar.org
 [70] [0910.3675] Index theory of 1D quantum walks and cellular automata - arXiv. https://arxiv.org
 [71] Quantum-inspired identification of complex cellular automata - Univ. of Manchester. https://research.manchester.ac.uk
 [72] Creating Network Motifs with Developmental Graph Cellular Automata - MIT Press Direct. https://direct.mit.edu
 [73] Motif modeling for cell signaling networks - ResearchGate. https://researchgate.net
 [74] Network discovery with DCM - PubMed Central. https://pmc.ncbi.nlm.nih.gov