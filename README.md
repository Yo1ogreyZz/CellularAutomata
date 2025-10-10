# CellularAutomata_Sem1_2025

GitHub Repository for CellularAutomata research project.

## Reproduction of [*Progress, gaps and obstacles in the classification of cellular automata* (Vispoel et. al., 2022)](https://www.sciencedirect.com/science/article/pii/S0167278921002311)

Clone this repository and run the `sanity_check.py` script to reproduce the key results from the paper.

```bash
$ cd CellularAutomata_Sem1_2025
$ python sanity_check.py \
  --rules 0,50,90,110,30 \
  --N 256 \
  --out ./data/ca_report.html \
  --assets ./data/ca_report_assets
```

This will generate an HTML report at `./data/ca_report.html` with associated assets in `./data/ca_report_assets`.