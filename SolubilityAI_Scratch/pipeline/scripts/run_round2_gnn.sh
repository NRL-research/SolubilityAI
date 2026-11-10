#!/bin/bash
# Round 2: extra GNN seeds (identical to round-1 gnn_hybrid except the seed).
cd "$(dirname "$0")/.."
python3 -u scripts/05_gnn.py gnn_hybrid_s43 1 80 43 > logs/r2_gnn_hybrid_s43.out 2>&1
python3 -u scripts/05_gnn.py gnn_hybrid_s44 1 80 44 > logs/r2_gnn_hybrid_s44.out 2>&1
echo DONE > logs/r2_gnn_done.flag
