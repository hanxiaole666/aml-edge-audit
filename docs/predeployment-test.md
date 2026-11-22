# Pre-deployment test for cross-split AML evaluation

This checklist is the portable deliverable of the Journal manuscript. It is deliberately a
decision procedure, not a formula: the paper's leave-one-graph-out regression has negative
R^2 in every direction, so no correction factor is claimed.

## Step 1 - measure three structural quantities

On the graph you are about to use for evaluation, report:

| Symbol | Quantity | How to compute |
|---|---|---|
| `n_cross` | number of edges joining train and test nodes | count edges with endpoints in different split sets |
| `d_cross` | mean number of crossing neighbours per test node | mean test-node degree restricted to crossing edges |
| `excess` | same-label rate of crossing edges minus independence | `P(same) - [p_train*p_test + (1-p_train)(1-p_test)]` |

Labels are used only to measure `excess`; they never enter the model as features.

## Step 2 - decide which channel dominates

- **`d_cross` >= 1** (several crossing neighbours per test node): the **degree channel** dominates.
  Removing or adding these edges changes how much neighbour averaging the test node receives.
  On our Real-CATS graph `d_cross = 8.15`, and at least 90% of the 0.115 PR-AUC gain from C1 to
  C3 is attributable to degree; the composition term is bounded by 0.012 PR-AUC.
- **`d_cross` < 1** (fewer than one crossing neighbour per node): the **composition channel**
  dominates. On Elliptic no edge crosses a step boundary (`d_cross = 0`), and the sign of the
  C1-C3 contrast follows the homophily of the test-test edges that a condition retains.

## Step 3 - read the sign correctly

- `excess < 0`: crossing edges are anti-homophilous; removing them raises PR-AUC.
- `excess > 0`: crossing edges are homophilous; removing them lowers PR-AUC.

The three degree-matched composition estimates in the paper disagree in sign (-0.005 to +0.011),
so the procedure returns a direction and a bound, not a point value.

## Step 4 - check index alignment for temporal models

Before trusting a causal/memory model on a temporal split, ask whether the **time index used by
the split is the same index the model consumes**. With intervention strength fixed at 25%,
TGN-lite's own optimism is +0.007 under a step split and +0.075 under a node-rank cut: the split
sets the **sign** of that optimism. But TGN scores below a static tree by 0.061 / 0.068 / 0.056
under step-25% / step-100% / node-rank-25% -- a stable offset of about 0.06 AP. The architecture
therefore supplies a bounded protection; index alignment decides whether the split's optimism
overwhelms it. Check both, not one.

## Step 5 - state the deployment mode

- **Batch / transductive scoring**: the scored entities and their mutual edges are observable
  at scoring time (condition C1).
- **Streaming / inductive scoring**: only edges whose endpoints already exist are available
  (condition C2).

Never compare a batch number with a streaming number without stating which one is reported.

## Step 6 - apply the leakage gates

Six gates are described in Section 6 of the manuscript. Two are structural and four are
dataset-specific. In particular, check label provenance columns: on Real-CATS, three columns
were flagged and two carry label information at |r| = 0.705 and 0.553; removing them changes
tabular AP by 0.155 to 0.226.

## Reproduce the numbers

```bash
python scripts/71_paper_numbers.py
python scripts/72_paper_consistency.py   # [PASS] 177/177 (v15)
```

Every tabulated value and every abstract number is generated from `results/*.json`.
