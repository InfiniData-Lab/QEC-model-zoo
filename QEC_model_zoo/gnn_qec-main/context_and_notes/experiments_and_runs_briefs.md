# Experiments and Runs Briefs

This file is a handoff for agents running additional experiments. It separates **real completed runs** from **recommended next runs**. Do not treat planned or hypothetical results as completed results.

## Current Codebase State

The project implements a minimal QEC decoder pipeline:

```text
Stim surface-code memory circuit
    -> detector-event samples + logical observable labels
    -> graph tensors
    -> TensorFlow/Keras GNN decoder
    -> PyMatching MWPM baseline
    -> metrics comparison
```

Main package files:

```text
qec_gnn/generate_data.py       Generate raw Stim detector samples and labels
qec_gnn/graph_builder.py       Build active-defect or fixed detector graph tensors
qec_gnn/model.py               GCN and residual GCN models
qec_gnn/train_gnn.py           Train GNN and tune validation threshold
qec_gnn/eval_mwpm.py           Evaluate PyMatching MWPM on same raw test split
qec_gnn/compare.py             Compare GNN and MWPM metrics
qec_gnn/utils/visualize_input_data.py  Visualize raw detector data and graph layout
```

Environment:

```bash
source .venv/bin/activate
```

The venv was created with `/opt/homebrew/bin/python3.10` because TensorFlow did not support the default Python 3.14.

## What The Data Is

Each example is one **Stim shot**, i.e. one simulated run of the same surface-code memory circuit.

For each shot, Stim gives:

```text
detectors: binary detector-event vector
label:     logical observable flip, 0 or 1
```

Important: the detector vector is **not raw qubit measurement data**. Stim already computes detector events from the circuit's `DETECTOR` annotations. A detector event is a consistency-check failure, often derived from comparing stabilizer/check measurements across rounds.

Current default physical setup used in real runs:

```text
circuit = surface_code:rotated_memory_z
distance = 3
rounds = 5
p = 0.02
num_detectors = 40
split = 70% train / 15% val / 15% test
```

## Real Completed Runs

### Run 1: Active-Defect GNN

Purpose: prove the end-to-end pipeline works.

Data:

```text
raw data = data/generated/raw_d3_r5_p02.npz
graph data = data/generated/graphs_d3_r5_p02.npz
shots = 2000
positive label fraction = 0.379500
test positive label fraction = about 0.396667
```

Graph representation:

```text
graph type = active_defect_knn
nodes = only detectors that fired in that shot
node features = normalized x, y, t, detector_index
edges = k-nearest-neighbor edges in detector coordinate space
max_nodes = 64
k = 4
```

Model:

```text
3 dense GCN layers
masked mean pool
masked max pool
masked flattened graph summary
dense classifier head
sigmoid logical-flip output
```

Sanity check:

```text
tiny-overfit accuracy = 0.953125
tiny-overfit error rate = 0.046875
```

Results:

```text
GNN accuracy = 0.560000
GNN error rate = 0.440000
GNN balanced accuracy = 0.540369

MWPM accuracy = 0.740000
MWPM error rate = 0.260000
MWPM balanced accuracy = 0.714007
```

Critical interpretation:

```text
This run was useful as a pipeline sanity check, but the model was not competitive.
The active-defect representation throws away inactive detector information.
The kNN graph is heuristic and not physically faithful.
The GNN underperformed even the majority baseline on this small split if judged by raw accuracy.
Do not present Run 1 as a successful decoder, only as a first baseline.
```

### Run 2: Fixed Detector-Location Residual GNN

Purpose: improve representation and test whether more data + more physical graph structure helps.

Data:

```text
raw data = data/generated/raw_run2_d3_r5_p02_s50000.npz
graph data = data/generated/graphs_fixed_run2_d3_r5_p02_s50000.npz
shots = 50000
positive label fraction = 0.374760
test positive label fraction = 0.367333
num_detectors = 40
```

Graph representation:

```text
graph type = fixed_detector_dem
nodes = all 40 possible detector locations for every shot
node features = detector_bit, x, y, t, detector_index, dem_degree, boundary_count
edges = detector-error-model-derived detector-detector edges
DEM detector-detector edges = 102
boundary-connected detectors = 40
```

Important graph-builder fix:

```text
Stim detector error model instructions can contain separated target groups like:
  D1 D5 ^ D4

These groups should not be treated as one clique.
The graph builder was fixed to respect target_groups().
Before this fix, the graph was over-connected with 182 edges.
After the fix, the graph has 102 detector-detector edges.
```

Model:

```text
model type = residual_gcn
hidden_dim = 128
dropout = 0.05
learning_rate = 0.0005
batch_size = 256
epochs requested = 40
epochs ran = 33
early stopping patience = 8
threshold = validation-tuned, 0.520
```

Sanity check:

```text
tiny-overfit accuracy = 0.976562
tiny-overfit error rate = 0.023438
```

Results:

```text
GNN accuracy = 0.693467
GNN error rate = 0.306533
GNN balanced accuracy = 0.632461
TP = 1109
TN = 4092
FP = 653
FN = 1646

MWPM accuracy = 0.760800
MWPM error rate = 0.239200
MWPM balanced accuracy = 0.737585
TP = 1791
TN = 3915
FP = 830
FN = 964
```

Artifacts:

```text
GNN metrics = outputs/metrics/gnn_residual_fixed_run2_demfix_d3_r5_p02_s50000.json
MWPM metrics = outputs/metrics/mwpm_run2_d3_r5_p02_s50000.json
Comparison = outputs/metrics/compare_run2_d3_r5_p02_s50000.json
Model = outputs/models/gnn_residual_fixed_run2_demfix_d3_r5_p02_s50000.keras
```

Critical interpretation:

```text
Run 2 is genuinely better than Run 1: 0.560 -> 0.693 accuracy.
It still does not beat MWPM: 0.693 vs 0.761 accuracy.
This suggests the issue is not only data volume or model size.
The GNN is still mostly a generic graph classifier, while MWPM has strong decoder-specific structure.
The GNN particularly has many false negatives: it misses many true logical flips.
Balanced accuracy is much lower than raw accuracy, so report both.
```

## Do Not Confuse With Hypothetical Material

`RESEARCH_JOURNEY.md` contains a polished narrative with a hypothetical/made-up Run 3 that reaches `0.800000`. That result is **not real** and does not correspond to saved metrics.

Do not use that file as evidence for completed experiments. If a poster/report includes an 80% run, it must be clearly labeled as planned/hypothetical unless a real run is executed and metrics are saved.

## Highest-Value Next Runs

These are recommended in priority order. The goal is to produce useful poster/final-report comparison points, not to randomly train larger GNNs.

### Priority 1: Simple Non-Graph Baselines

Why this matters:

```text
We need to know whether graph structure is actually helping.
If an MLP or logistic regression on the 40 detector bits matches the GNN, then the current GNN is not adding much.
```

Run on the Run 2 raw/fixed dataset using the same train/val/test split.

Baselines to implement:

```text
majority baseline: always predict 0
logistic regression: input = raw 40 detector bits
MLP: input = raw 40 detector bits
random forest or gradient boosting if sklearn is available
```

Expected majority baseline:

```text
Run 2 test positive fraction = 0.367333
majority/no-flip accuracy = 1 - 0.367333 = 0.632667
```

Critical recommendation:

```text
Do this before training more GNNs.
If MLP gets close to 0.69, current graph model is not very meaningful.
If MLP beats the GNN, the graph/model design is actively hurting.
If GNN beats MLP by a clear margin, the graph representation is useful.
```

Suggested output file names:

```text
outputs/metrics/majority_run2_d3_r5_p02_s50000.json
outputs/metrics/logreg_run2_d3_r5_p02_s50000.json
outputs/metrics/mlp_run2_d3_r5_p02_s50000.json
outputs/metrics/baseline_compare_run2_d3_r5_p02_s50000.json
```

### Priority 2: Dataset Size Sweep

Question:

```text
Is the GNN data-limited?
```

Suggested sizes:

```text
10k shots
50k shots already done
100k shots
250k shots only if training time is acceptable
```

Critical recommendation:

```text
Generating Stim shots is cheap; training is the bottleneck.
Do not jump to 250k before checking whether 100k improves validation/test accuracy.
If 100k barely improves over 50k, more data alone is probably not the fix.
```

Suggested commands:

```bash
.venv/bin/python -m qec_gnn.generate_data \
  --distance 3 --rounds 5 --p 0.02 --shots 100000 \
  --out data/generated/raw_run3_d3_r5_p02_s100000.npz

.venv/bin/python -m qec_gnn.graph_builder \
  --input data/generated/raw_run3_d3_r5_p02_s100000.npz \
  --graph-type fixed \
  --out data/generated/graphs_fixed_run3_d3_r5_p02_s100000.npz

.venv/bin/python -m qec_gnn.eval_mwpm \
  --raw-data data/generated/raw_run3_d3_r5_p02_s100000.npz \
  --out outputs/metrics/mwpm_run3_d3_r5_p02_s100000.json

.venv/bin/python -m qec_gnn.train_gnn \
  --data data/generated/graphs_fixed_run3_d3_r5_p02_s100000.npz \
  --model-type residual_gcn \
  --class-weight-mode none \
  --hidden-dim 128 \
  --dropout 0.05 \
  --learning-rate 0.0005 \
  --epochs 40 \
  --batch-size 256 \
  --patience 8 \
  --out outputs/models/gnn_residual_fixed_run3_d3_r5_p02_s100000.keras \
  --metrics-out outputs/metrics/gnn_residual_fixed_run3_d3_r5_p02_s100000.json
```

### Priority 3: Noise Probability Sweep

Question:

```text
Does performance depend strongly on detector-event density / class balance?
```

Suggested settings:

```text
p = 0.005
p = 0.01
p = 0.02 already done
p = 0.03
```

Critical recommendation:

```text
This is useful for understanding the problem, but poster time may be better spent on simple baselines first.
At low p, labels may be highly imbalanced and raw accuracy can be misleading.
Always report positive label fraction, balanced accuracy, and confusion matrix.
```

### Priority 4: Feature Ablation On Fixed Graph

Question:

```text
Which node features actually help?
```

Current features:

```text
detector_bit, x, y, t, detector_index, dem_degree, boundary_count
```

Ablations:

```text
A: detector_bit only
B: detector_bit + x,y,t
C: detector_bit + x,y,t + detector_index
D: detector_bit + x,y,t + dem_degree
E: detector_bit + x,y,t + boundary_count
F: full feature set
```

Critical recommendation:

```text
This requires code changes unless the graph builder is extended with feature-selection flags.
Do not run this manually by editing arrays ad hoc unless results are clearly recorded.
This is high scientific value because it can reveal whether DEM degree/boundary_count are useful or just noise.
```

### Priority 5: Edge-Weighted / Edge-Feature GNN

Question:

```text
Can the GNN use edge reliability like MWPM does?
```

Current limitation:

```text
The GNN adjacency is effectively binary/normalized.
MWPM uses weighted paths from error probabilities.
The GNN does not currently see the actual DEM edge probability/weight as an edge feature.
```

Recommendation:

```text
Implement edge weights from DEM probabilities before trying much larger hidden dimensions.
A weighted adjacency or true edge-feature message passing is more justified than making the current GCN wider.
```

Critical note:

```text
This is a real architecture change. It may not be fast enough for tomorrow's poster, but it is a strong final-report direction.
```

### Priority 6: Explicit Boundary Nodes

Question:

```text
Can the GNN model detector-to-boundary matching more like MWPM?
```

Current limitation:

```text
Boundary information is compressed into a scalar boundary_count feature.
MWPM can explicitly match detectors to boundaries.
```

Recommendation:

```text
Add one or more boundary nodes and connect detectors with single-detector DEM mechanisms to boundary nodes.
This is more decoder-like than boundary_count.
```

Critical note:

```text
This is likely more useful than adding generic depth/width.
However, it changes graph size and feature semantics; document carefully.
```

## Lower-Priority Runs

### Bigger Residual GCN

Not recommended as first next step.

Reason:

```text
The model already overfits tiny subsets.
The bottleneck is probably not raw capacity.
Larger hidden_dim may increase runtime without addressing matching/path/boundary reasoning.
```

If tried anyway:

```text
hidden_dim = 256
dropout = 0.1
epochs = 40
```

Only keep this if it beats the 128-dim baseline on validation and test, not just train.

### More Epochs

Not very promising by itself.

Reason:

```text
Run 2 early-stopped at 33/40 epochs.
Validation loss stopped improving.
More epochs without architecture/data changes likely overfit.
```

### Active-Defect Graph Variants

Low priority.

Reason:

```text
Active-defect graphs lose inactive detector information.
They are useful pedagogically, but fixed detector-location graphs are a better default for the current setup.
```

## Poster-Friendly Progress Claims

Safe completed claims:

```text
Built full Stim -> graph -> GNN -> MWPM comparison pipeline.
Verified detector-event data comes from real Stim surface-code simulations.
Implemented active-defect graph baseline.
Implemented fixed detector-location graph with detector bits as node features.
Derived graph edges from Stim detector error model.
Fixed a DEM parsing issue involving separated target groups.
Added residual GCN model, pooling, and validation-tuned thresholding.
Improved GNN accuracy from 0.56 to 0.693.
MWPM remains stronger at 0.761 on the same test split.
```

Do not claim:

```text
Do not claim the GNN beat MWPM unless a real run proves it.
Do not claim 80% accuracy as completed. That exists only in narrative/hypothetical notes.
Do not claim the issue is solved by more data; this has not been shown.
```

Good framing:

```text
Preliminary results show that graph representation matters:
active-defect GNN: 0.56 accuracy
fixed detector-location residual GNN: 0.693 accuracy
MWPM baseline: 0.761 accuracy

Next work focuses on understanding whether the remaining gap is due to:
1. lack of simple baseline comparison,
2. lack of edge weights/boundary nodes,
3. insufficient data,
4. using graph classification instead of a more decoder-shaped message-passing architecture.
```

## Suggested Parallel Work Queue

Agent A: Simple baselines

```text
Implement/evaluate majority, logistic regression, MLP on raw detector bits.
Use existing Run 2 raw data and saved split indices.
Produce one JSON per baseline and one comparison summary.
```

Agent B: 100k dataset run

```text
Generate 100k Stim shots at d=3, r=5, p=0.02.
Build fixed detector graph.
Evaluate MWPM.
Train same residual GCN config as Run 2.
Compare to 50k result.
```

Agent C: Visualization/poster support

```text
Use qec_gnn/utils/visualize_input_data.py to generate clear detector-vector and graph plots.
Prefer outputs/plots/input_visualization_fixed_v2 style, not stale smoke plots.
Make sure captions explain: white nodes are inactive detector-event locations, red nodes are fired detector events.
```

Agent D: Design next architecture

```text
Plan edge-weighted GNN or boundary-node GNN.
Do not run until interface is clear and metrics from simple baselines are available.
```

## Final Critical Recommendation

For the immediate poster, the best honest story is:

```text
The current GNN attempts are not yet competitive with MWPM, but they demonstrate a working pipeline and a meaningful representation improvement.
The strongest next evidence would be simple non-graph baselines. Without them, it is hard to say whether the GNN is learning graph structure or just acting like a generic classifier on detector bits.
After baselines, prioritize edge weights and explicit boundary nodes over simply making the GNN larger.
```
