# Literature Review Guide

> Until Oct 16

## 1. General Cellular Automata and Classification

These papers provide the foundational concepts for Elementary Cellular Automata (ECA), including the Wolfram coding system, the classification of rules into four dynamical classes, and the reduction of the 256 rules into 88 equivalence classes based on geometric symmetries.

[**Wolfram, S. (1983). Statistical Mechanics of Cellular Automata.** *Reviews of Modern Physics, 55*(3), 601–644.](statistical-mechanics-cellular-automata.pdf)

**Contribution: **This is a foundational paper that systematically introduced the study of elementary cellular automata. It established the Wolfram coding scheme for the 256 rules and proposed the influential qualitative classification of their behavior into four classes (Uniform, Periodic, Chaotic, Complex).

[**Li, W., & Packard, N. H. (1990). The Structure of the Elementary Cellular Automata Rule Space.** *Complex Systems, 4*(3), 281-297.](The Structure of the Elementary Cellular Automata Rule Space.pdf) 

**Contribution:** This paper provides a detailed analysis of the ECA "rule space." It formally discusses the equivalence of rules under reflection and complementation, leading to the identification of the 88 fundamental (non-equivalent) rule classes.

[**Martínez, G. J., de Jesús-Farfán, E., & Seck-Tuoh-Mora, J. C. (2023). \*A study on the composition of elementary cellular automata\*. arXiv preprint arXiv:2305.02947.**](A study on the composition of elementary cellular automata.pdf) 

**Contribution:** This work offers a modern, algebraic approach to classifying ECAs by studying the semigroup formed by their composition. It demonstrates that a rule's complexity (in Wolfram's sense) correlates with its number of "companions," providing a quantitative alternative to qualitative classification.

## 2. State Transition Graphs (STG)

These papers focus on using State Transition Graphs (also known as transition diagrams or cycle graphs) to represent the complete global dynamics of a finite CA system. They analyze the graph's topology to understand attractors, basins of attraction, and symmetries.

[**Martin, O., Odlyzko, A. M., & Wolfram, S. (1984). Algebraic properties of cellular automata.** *Communications in Mathematical Physics, 93*(2), 219-258.](algebraic-properties-cellular-automata.pdf) 

**Contribution:** A seminal paper that extensively uses state transition graphs to analyze the global properties of ECAs, particularly additive rules. It formally defines how cycles, transients, and basins of attraction are represented in the graph and links these topological features to the algebraic properties of the rules.

[**McIntosh, H. V. (2009). Automorphisms of Transition Graphs for Elementary Cellular Automata.** *In Artificial Life, Transfer, and Simulation.*](Automorphisms of transition graphs for elementary cellular automata.pdf)

**Contribution:** This work introduces the idea of studying the symmetries of the global dynamics by analyzing the automorphisms (symmetries) of the state transition graph. It proposes a classification of ECAs based on how the number of automorphisms scales with the system size, successfully identifying linear and chaotic rules.

## 3. De Bruijn Graphs (DBG)

These papers are central to the use of De Bruijn Graphs for compactly encoding the local rules of a CA. They demonstrate how this representation enables efficient algorithmic analysis of rule properties like reversibility and surjectivity.

[**Sutner, K. (1991). De Bruijn graphs and linear cellular automata.** *Complex Systems, 5*(1), 19-30.](De Bruijn Graphs and Linear Cellular Automata.pdf)

**Contribution: **This is the definitive paper on applying De Bruijn graphs to the analysis of cellular automata. Sutner shows how to represent an ECA rule using an edge-labeled DBG and develops polynomial-time algorithms to decide reversibility and surjectivity by transforming these properties into graph reachability problems on the DBG and its derivatives (like the Pair Graph).

[**Hernández, S. C., et al. (2018). Graphs Related to Reversibility and Complexity in Cellular Automata.** *In Encyclopedia of Complexity and Systems Science.*](graphs_related_to_reversibility_and_complexity in cellular automata.pdf) 

**Contribution: **This entry provides a comprehensive overview of various graphs used in CA analysis, with a strong focus on the De Bruijn graph, Pair graph, and Subset graph. It clearly explains their construction and application in studying reversibility and complexity, serving as an excellent summary of the DBG-based approach.

## 4. Graph Cellular Automata (GCA) and Graph Neural Networks (GNN)

These papers represent the modern generalization of CAs to arbitrary network structures and the paradigm shift towards learning CA rules using machine learning.

[**Grattarola, D., et al. (2021). Learning Graph Cellular Automata.** *Advances in Neural Information Processing Systems (NeurIPS), 34*.](Learning Graph Cellular Automata.pdf) 

**Contribution:** This paper formally introduces Graph Neural Cellular Automata (GNCA). It establishes the parallel between GCA and Graph Neural Networks (GNNs) and proposes a general architecture for using GNNs to learn the transition rule of a GCA from data. This marks a significant shift from analyzing predefined rules to synthesizing new ones for desired behaviors.

[**Balloccu, D., et al. (2023). E(n)-Equivariant Graph Neural Cellular Automata.** *arXiv preprint arXiv:2301.10497*.](E(n)-equivariant Graph Neural Cellular Automata.pdf)

**Contribution:** This work improves upon the initial GNCA concept by incorporating physical symmetries (equivariance to translation, rotation, etc.) into the GNN architecture. This allows for the learning of more robust and generalizable isotropic rules, avoiding issues of anisotropy inherent in arbitrary graph structures.