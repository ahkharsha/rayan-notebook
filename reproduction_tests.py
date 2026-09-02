"""Fast invariant tests for the recovery notebooks and shared implementation."""
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

from paper_reproduction_core import (Config, DGHeteroGNN, FocalLoss, HeteroGraphSAGE,
                                     EXPECTED_EDGE_TYPES, _device_view, flatten_graph,
                                     grl, load_all, project_root, select_threshold,
                                     transaction_split)

graphs, table = load_all(project_root())
assert table.shape == (4, 5)
assert all(set(graph.edge_types) == EXPECTED_EDGE_TYPES for graph in graphs.values())
assert not any(edge[1].startswith("rev_") for graph in graphs.values() for edge in graph.edge_types)

# Gradient reversal sign and focal-loss hand calculation.
x = torch.ones((2, 1), requires_grad=True)
grl(x, 1.0).sum().backward()
assert torch.equal(x.grad, -torch.ones_like(x))
loss = FocalLoss(.25, 2.0)(torch.tensor([0.0]), torch.tensor([1.0]))
expected = .25 * .25 * float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([0.0]), torch.tensor([1.0])))
assert abs(float(loss) - expected) < 1e-7

# Disjoint stratified source masks and model output dimensions.
train, valid = transaction_split(graphs["amlsim"], 42, .2)
assert not set(train.tolist()).intersection(valid.tolist())
model = DGHeteroGNN(graphs["amlsim"].metadata(), Config(), n_domains=3)
logits, domains = model(graphs["amlsim"].x_dict, graphs["amlsim"].edge_index_dict)
assert logits.shape == (graphs["amlsim"]["transaction"].num_nodes,)
assert domains.shape == (graphs["amlsim"]["transaction"].num_nodes, 3)
assert torch.isfinite(logits).all()

# Exact threshold regression, including duplicate probabilities and tie breaking.
rng = np.random.default_rng(3407)
for size in range(1, 100):
    labels = rng.integers(0, 2, size)
    probabilities = rng.choice([0.0, .1, .2, .5, .9, 1.0], size)
    candidates = np.unique(np.r_[0., 1., probabilities])
    expected = float(candidates[np.argmax([f1_score(labels, probabilities >= t, zero_division=0) for t in candidates])])
    assert select_threshold(labels, probabilities) == expected

# Device views do not mutate CPU graphs and adaptive feature views contain no y.
original_device = graphs["amlsim"]["transaction"].x.device
feature_view = _device_view(graphs["amlsim"], torch.device("cpu"), include_labels=False)
assert graphs["amlsim"]["transaction"].x.device == original_device
assert "y" not in feature_view["transaction"]

# Homogeneous conversion preserves every archived directed edge once.
_, flat_edges, _ = flatten_graph(graphs["amlsim"])
assert flat_edges.size(1) == sum(graphs["amlsim"][edge].edge_index.size(1) for edge in graphs["amlsim"].edge_types)

# HeteroGraphSAGE is structurally independent from all DG machinery.
baseline = HeteroGraphSAGE(graphs["amlsim"].metadata(), Config())
assert not hasattr(baseline, "domain_classifier")
assert not any("domain" in name.lower() or "grl" in name.lower() for name, _ in baseline.named_modules())

# The adaptive trainer exposes target graph features, never target labels, to its training signature/body.
from paper_reproduction_core import train_proposed
source = inspect.getsource(train_proposed)
assert "target_features" in source and 'target_features["transaction"].y' not in source and "target_features.y" not in source

# Both notebooks embed, rather than import, the reviewed implementation.
for notebook_name in ("DG_Hetero_GNN_Paper_Minimal_Diff.ipynb", "DG_Hetero_GNN_Paper_Clean_Reproduction.ipynb"):
    notebook = json.loads((Path(__file__).parent / notebook_name).read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "def select_threshold" in code and "def run_suite" in code
    assert "from paper_reproduction_core import" not in code
print("PASS: topology, Table 1, threshold equivalence, device isolation, leakage guard, baseline independence, and embedded notebooks")
