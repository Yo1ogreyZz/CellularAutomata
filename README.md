# CellularAutomata

GitHub Repository for Project: *Cellular Automata Spatiotemporal Pattern Classification using Graph Neural Networks.*

## Installation

Clone this repository and install locally using pip:

```bash
$ git clone https://github.com/Yo1ogreyZz/CellularAutomata.git
$ cd CellularAutomata
$ pip install -e .
```

## Reproduction of [*Progress, gaps and obstacles in the classification of cellular automata* (Vispoel et. al., 2022)](https://www.sciencedirect.com/science/article/pii/S0167278921002311)

Clone this repository and run the `sanity_check.py` script to reproduce the key results from the paper.

```bash
$ cd CellularAutomata
$ python src/eca_gnn/sanity_check.py \
  --rules 0,90,30,160,36,8,76,50,18,110 \
  --N 256 \
  --out ./data/ca_report.html \
  --assets ./data/ca_report_assets
```

This will generate an HTML report at `./data/ca_report.html` with associated assets in `./data/ca_report_assets`