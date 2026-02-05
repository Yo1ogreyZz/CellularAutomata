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

## 6-view fusion training

Use the new 6-view fusion trainer (supports any number of views) and pass the graph roots you generated in `output/`:

```bash
python gnns/train_6views_gnn.py \
  --view-roots output/pyg_radius2_debruijn_std output/pyg_radius2_symbol01_std output/pyg_radius2_dependency_std \
  --view-names dbg sym dep \
  --hidden 96 \
  --class-weight-power 0.4
```

### Tips to improve accuracy

- Keep `--balanced-sampler` enabled to reduce class bias; adjust `--class-weight-power` between `0.3` and `0.7` for best macro-F1.
- Try larger `--hidden` (e.g. 128) and slightly lower `--dropout` (e.g. 0.1) if the model underfits.
- Increase `--min-epochs` and `--patience` for more stable early stopping on macro-F1.
