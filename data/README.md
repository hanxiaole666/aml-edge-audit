# Data directory

This folder contains only small, redistributable datasets. Large datasets are retrieved by
the scripts in `scripts/`.

| File | Source | License | Notes |
|---|---|---|---|
| `real_cats_eth.csv` | Real-CATS (GitHub) | CC BY-NC-SA | 28,591 ETH address profiles, 49 features |
| `real_text.csv` | Derived from Real-CATS | CC BY-NC-SA | Behavioral textualization |
| `phrasebank_real.csv` | Financial PhraseBank (HF mirror) | Check upstream | 2,264 sentences, binary sentiment as risk proxy |
| `german_credit.csv`, `synthetic_*.csv` | UCI / generated | Standard | Control and pipeline tests |
| `real_cats/real_graph_*.csv` | Derived from Real-CATS token interactions | CC BY-NC-SA | 1,500 nodes, 6,942 edges |
| `elliptic2/*.csv` | Elliptic2 (Kaggle) | CC BY-NC-ND | Labeled subgraph topology; features via `22_cache_node_features.py` |

The Elliptic2 background files (5.3 GB node features, 24 GB background edges) are **not**
included. Use `scripts/fast_download.py` and `scripts/15_elliptic2_extract.py`.
