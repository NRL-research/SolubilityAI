#!/bin/bash
# Round 2: extra seed for the graph-only GNN (seed averaging helped the hybrid: 0.986 -> 0.959 with 2 seeds).
cd "$(dirname "$0")/.."
while [ ! -f logs/r2_gnn_done.flag ]; do sleep 30; done
python3 -u scripts/05_gnn.py gnn_mpnn_s43 0 80 43 > logs/r2_gnn_mpnn_s43.out 2>&1
echo DONE > logs/r2_gnn2_done.flag
