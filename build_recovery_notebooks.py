"""Build the two auditable DG-Hetero-GNN recovery notebooks.

This generator is intentionally checked into the recovery workspace so the
notebook JSON is reproducible rather than hand-edited.
"""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parent

shared_setup = r'''# Runtime, paths, and paper-controlled configuration
from __future__ import annotations
import os, sys, json, time, random, hashlib, zipfile, tempfile, platform
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)

# Install the documented environment before a clean-kernel run when needed:
#   python -m pip install -r requirements-reproduction.txt
# XGBoost is intentionally imported lazily by the baseline runner. The suite
# stops with an explicit import error instead of silently skipping that model.

# This cell resolves paths from the notebook's location, not its current directory.
ROOT = Path.cwd().resolve()
if not (ROOT / "recovery-source" / "Datasets").exists():
    ROOT = next(p for p in [ROOT, *ROOT.parents] if (p / "recovery-source" / "Datasets").exists())
DATASET_DIR = ROOT / "recovery-source" / "Datasets"
ARTIFACTS = ROOT / "artifacts"
RUN_MODE = os.environ.get("DG_HETERO_RUN_MODE", "SMOKE_TEST") # FULL_PAPER_RUN requires deliberate opt-in
SEEDS = [42, 123, 3407, 2025, 9999]
EXPECTED_SHA256 = {
 "elliptic": "0a86c749a1c388be50ca2a828485532216dd1cf29f39e9a5a69159c66514ee29",
 "ieee": "e73460dffa01a807cdf113a768c535f8d21d9bf6a94e041ca863449d220bf761",
 "dgraphfin": "c7726f1dfd085e548ac86ad1cae62d751c87df95248260499ab431b9bfbb9e21",
 "amlsim": "99dd1eb32ad2a71fd6eb95eaa942951599f88418c1fafe884957a143491acb1e",
}
ARCHIVES = {"elliptic":"elliptic_standardized.zip", "ieee":"ieee_standardized.zip",
            "dgraphfin":"dgraphfin_reduced.zip", "amlsim":"amlsim_final_unified.zip"}
TABLE_1 = {"elliptic":(46564,93128,128,222880), "ieee":(590540,14962,128,2952700),
           "dgraphfin":(250000,324822,128,1036164), "amlsim":(45,35,128,443)}
SPEC = {
 "architecture": "type-specific Linear-LayerNorm-ReLU-Dropout; 3 relation-specific GraphSAGE layers; deep transaction head",
 "dimensions": {"transaction":128,"entity":64,"hidden":48,"layers":3},
 "training": {"optimizer":"Adam","lr":0.0005,"epochs":40,"seeds":SEEDS},
 "lodo": {"LODO0":["ieee","dgraphfin","elliptic"], "LODO1":["elliptic","dgraphfin","ieee"], "LODO2":["elliptic","ieee","dgraphfin"]},
 "ood": "sources elliptic/ieee/dgraphfin -> AMLSim; no target feature or label training access",
 "adaptive_ood": "transductive unlabeled AMLSim features are domain-classifier inputs only; AMLSim labels are final-evaluation-only",
 "metrics": ["ROC-AUC","PR-AUC","Precision","Recall","F1","threshold","ECE","Brier","TP","TN","FP","FN"],
}
AMBIGUITIES = pd.DataFrame([
 ["focal alpha", "Focal loss is named but alpha omitted", 0.25, "standard conservative class weighting", True],
 ["focal gamma", "Focal loss is named but gamma omitted", 2.0, "canonical focusing value", True],
 ["domain-loss lambda", "Equation gives lambda but no numeric value", 0.10, "keeps adversarial term subordinate", True],
 ["GRL strength", "GRL is described without schedule", 1.0, "constant, auditable reversal", True],
 ["dropout", "Dropout is named without probability", 0.15, "small regularization", True],
 ["validation fraction", "Validation procedure unspecified", 0.20, "stratified source-only holdout", True],
 ["weight decay", "Not specified", 0.0, "Adam exactly as stated", True],
 ["classifier widths", "Deep head width omitted", "64->32", "minimal decreasing head", True],
 ["standard deviation", "Convention omitted", "population (ddof=0)", "matches common seed summaries", False],
 ["adaptive OOD", "target use is internally tense", "unlabeled target features only", "prevents target-label leakage", True],
], columns=["item","paper statement / omission","assumption","conservative rationale","sensitivity check"])
print("Run mode:", RUN_MODE, "| root:", ROOT)
display(pd.DataFrame([(k,json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in SPEC.items()], columns=["Paper specification","Implementation"])); display(AMBIGUITIES)
'''

loading = r'''# Archive integrity, safe extraction, normalization, and invariants
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def safe_member_path(base, member):
    target=(base/member).resolve()
    if base.resolve() not in target.parents: raise ValueError(f"unsafe archive member: {member}")
    return target

def load_graph(name):
    archive=DATASET_DIR/ARCHIVES[name]
    actual=sha256(archive)
    assert actual == EXPECTED_SHA256[name], f"{name} SHA-256 mismatch: {actual}"
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as td:
        base=Path(td)
        for member in z.infolist():
            out=safe_member_path(base, member.filename); out.parent.mkdir(parents=True,exist_ok=True)
            if not member.is_dir():
                with z.open(member) as source, open(out,"wb") as dest: dest.write(source.read())
        pts=list(base.rglob("*.pt")); assert len(pts)==1, f"expected one .pt in {archive}"
        obj=torch.load(pts[0], weights_only=False, map_location="cpu")
    if isinstance(obj,list): assert len(obj)==1, f"{name}: serialized list must contain one graph"; obj=obj[0]
    assert isinstance(obj,HeteroData), f"{name}: expected HeteroData"
    for t in ("account","merchant"):
        x=obj[t].x
        if x.size(1)==32: obj[t].x=torch.cat((x, torch.zeros((x.size(0),32),dtype=x.dtype)),dim=1)
        assert obj[t].x.size(1)==64, f"{name} {t} width is not 64"
    return obj

def add_reverse_relations(g):
    # Original directions remain untouched; reverse relations are exact transposes.
    for s,r,d in list(g.edge_types):
        if (d, f"rev_{r}", s) not in g.edge_types:
            g[(d,f"rev_{r}",s)].edge_index=g[(s,r,d)].edge_index.flip(0)
    return g

def validate_graph(name,g):
    assert set(("transaction","account","merchant")) <= set(g.node_types)
    y=g["transaction"].y.reshape(-1)
    assert g["transaction"].x.size(1)==128 and set(y.unique().tolist()) <= {0,1}
    for t in g.node_types: assert torch.isfinite(g[t].x).all()
    for et in g.edge_types:
        ei=g[et].edge_index; assert ei.ndim==2 and ei.size(0)==2
        assert ((ei[0]>=0)&(ei[0]<g[et[0]].num_nodes)).all() and ((ei[1]>=0)&(ei[1]<g[et[2]].num_nodes)).all()
    n=g["transaction"].num_nodes; e=sum(g[et].edge_index.size(1) for et in g.edge_types if not et[1].startswith("rev_"))
    entities=g["account"].num_nodes+g["merchant"].num_nodes
    assert (n,entities,128,e)==TABLE_1[name], f"Table 1 mismatch for {name}: {(n,entities,128,e)}"
    if name=="elliptic": assert int((y==0).sum())==42019 and int((y==1).sum())==4545
    return {"Dataset":name,"Transaction nodes":n,"Entity nodes":entities,"Feature size":128,"Total edges":e}

GRAPHS={k:add_reverse_relations(load_graph(k)) for k in ARCHIVES}
TABLE1=pd.DataFrame([validate_graph(k,v) for k,v in GRAPHS.items()])
display(TABLE1)
'''

model = r'''# DG-Hetero-GNN: no global-device movement in forward; source-only fraud loss
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark=False

class GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,strength): ctx.strength=strength; return x.view_as(x)
    @staticmethod
    def backward(ctx,grad): return -ctx.strength*grad, None
def grl(x,strength=1.0): return GRLFn.apply(x,strength)

class FocalLoss(nn.Module):
    def __init__(self,alpha=.25,gamma=2.): super().__init__(); self.alpha,self.gamma=alpha,gamma
    def forward(self,logits,y):
        bce=F.binary_cross_entropy_with_logits(logits,y.float(),reduction="none")
        pt=torch.exp(-bce); alpha_t=torch.where(y>0,self.alpha,1-self.alpha)
        return (alpha_t*(1-pt).pow(self.gamma)*bce).mean()

class DGHeteroGNN(nn.Module):
    def __init__(self,metadata,n_domains=4,dropout=.15):
        super().__init__(); hidden=48
        self.project=nn.ModuleDict({"transaction":nn.Sequential(nn.Linear(128,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(dropout)),"account":nn.Sequential(nn.Linear(64,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(dropout)),"merchant":nn.Sequential(nn.Linear(64,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(dropout))})
        self.layers=nn.ModuleList([HeteroConv({e:SAGEConv((hidden,hidden),hidden) for e in metadata[1]},aggr="sum") for _ in range(3)])
        self.classifier=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,1))
        self.domain=nn.Sequential(nn.Linear(hidden,16),nn.ReLU(),nn.Linear(16,n_domains))
    def forward(self,x_dict,edge_index_dict):
        h={t:self.project[t](x.float()) for t,x in x_dict.items()}
        for layer in self.layers: h={t:F.relu(x) for t,x in layer(h,edge_index_dict).items()}
        tx=h["transaction"]
        return self.classifier(tx).flatten(), self.domain(grl(tx))

def source_split(y,seed,frac=.2):
    idx=np.arange(len(y)); a,b=train_test_split(idx,test_size=frac,stratify=y,random_state=seed)
    assert not set(a)&set(b); return torch.tensor(a),torch.tensor(b)
def select_threshold(y,p):
    candidates=np.unique(np.r_[0.,1.,p]); scores=[f1_score(y,p>=t,zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); total=0.0
    for i in range(bins):
        mask=(p>=edges[i]) & (p < (edges[i+1] if i+1 < bins else 1.000001))
        if mask.any(): total += abs(y[mask].mean()-p[mask].mean()) * mask.mean()
    return float(total)
def metrics(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"ROC-AUC":roc_auc_score(y,p),"PR-AUC":average_precision_score(y,p),"Precision":precision_score(y,pred,zero_division=0),"Recall":recall_score(y,pred,zero_division=0),"F1":f1_score(y,pred,zero_division=0),"threshold":t,"ECE":ece(y,p),"Brier":brier_score_loss(y,p),"TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn)}
'''

run = r'''# Protocol guard and execution manifest. Full suite is intentionally opt-in.
def environment_manifest():
    return {"python":sys.version,"torch":torch.__version__,"pyg":torch_geometric.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"numpy":np.__version__,"pandas":pd.__version__,"configuration":SPEC,"dataset_hashes":{k:sha256(DATASET_DIR/v) for k,v in ARCHIVES.items()},"run_mode":RUN_MODE}

def write_json_atomic(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8"); tmp.replace(path)

def preflight():
    x=torch.ones((2,1),requires_grad=True); grl(x).sum().backward(); assert torch.all(x.grad==-1)
    assert torch.isfinite(FocalLoss()(torch.tensor([0.,1.]),torch.tensor([0.,1.])))
    write_json_atomic(ARTIFACTS/"manifest.json",environment_manifest())
    write_json_atomic(ARTIFACTS/"graph_statistics.json",TABLE1.to_dict(orient="records"))
    print("PASS: hashes, graph invariants, GRL sign, focal-loss finiteness, and manifest.")

def protocol_notice():
    scenarios={"LODO0":(["ieee","dgraphfin"],"elliptic"),"LODO1":(["elliptic","dgraphfin"],"ieee"),"LODO2":(["elliptic","ieee"],"dgraphfin"),"OOD":(["elliptic","ieee","dgraphfin"],"amlsim"),"Adaptive OOD":(["elliptic","ieee","dgraphfin"],"amlsim")}
    display(pd.DataFrame([{"scenario":k,"sources":", ".join(v[0]),"target":v[1],"target labels in training":False,"target features in training":k=="Adaptive OOD"} for k,v in scenarios.items()]))
    if RUN_MODE!="FULL_PAPER_RUN": print("SMOKE_TEST is deliberately non-paper-valid. Set DG_HETERO_RUN_MODE=FULL_PAPER_RUN only to launch all five-seed runs.")

preflight(); protocol_notice()
'''

def make_notebook(clean=False):
    # Embed the reviewed engine verbatim. This prevents an imported module from
    # silently differing from the code a reviewer sees in the notebook.
    core_source = (ROOT / "paper_reproduction_core.py").read_text(encoding="utf-8")
    n=nbf.v4.new_notebook()
    title = "DG-Hetero-GNN Paper Clean Reproduction" if clean else "DG-Hetero-GNN Paper Minimal-Diff Recovery"
    n.cells=[nbf.v4.new_markdown_cell(f"# {title}\n\nThe attached paper is the sole normative specification. Commit 9 supplies only the recovered processed archives. The complete reviewed implementation is embedded below and is the code this notebook executes.")]
    if clean:
        tree = ast.parse(core_source)
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        def start_line(node):
            return min([node.lineno, *[item.lineno for item in getattr(node, "decorator_list", [])]])
        first_definition = min(start_line(node) for node in definitions)
        chunks = ["\n".join(core_source.splitlines()[:first_definition-1])]
        lines = core_source.splitlines()
        for index, node in enumerate(definitions):
            start = start_line(node)
            end = start_line(definitions[index+1])-1 if index+1 < len(definitions) else len(lines)
            chunks.append("\n".join(lines[start-1:end]))
        n.cells.append(nbf.v4.new_markdown_cell("## Clean modular implementation\n\nDefinitions are separated into auditable cells while retaining byte-for-byte function bodies from the reviewed engine."))
        n.cells.extend(nbf.v4.new_code_cell(chunk) for chunk in chunks if chunk.strip())
    else:
        n.cells.append(nbf.v4.new_markdown_cell("## Minimal-diff executable implementation\n\nThe reviewed engine remains in one contiguous cell to minimize notebook-level restructuring."))
        n.cells.append(nbf.v4.new_code_cell(core_source))
    n.cells.extend([nbf.v4.new_markdown_cell("## Artifact-backed experiment suite\n\nProgress is flushed at every scenario/model/seed and epoch. A stable run ID resumes atomic per-seed checkpoints; different modes and notebook implementations cannot share artifacts."), nbf.v4.new_code_cell(r'''ROOT = project_root(Path.cwd())
RUN_MODE = os.environ.get("DG_HETERO_RUN_MODE", "SMOKE_TEST")
RUN_ID = os.environ.get("DG_HETERO_RUN_ID", "interactive-smoke")
IMPLEMENTATION_ID = "clean-notebook" if "Clean" in "''' + title + r'''" else "minimal-diff-notebook"
CORE_CONFIG = Config(run_mode=RUN_MODE)
SUITE_RESULTS = run_suite(ROOT, CORE_CONFIG, run_id=RUN_ID, implementation_id=IMPLEMENTATION_ID)
summary = pd.DataFrame([{"scenario": r["scenario"], "model": r["model"], "F1 mean": r["aggregate"]["F1"]["mean"], "run mode": r["run_mode"]} for r in SUITE_RESULTS])
display(summary)
print("Completed run", RUN_ID, "for", IMPLEMENTATION_ID)
''')])
    n.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}}
    return n

for name, clean in [("DG_Hetero_GNN_Paper_Minimal_Diff.ipynb",False),("DG_Hetero_GNN_Paper_Clean_Reproduction.ipynb",True)]:
    nbf.write(make_notebook(clean),ROOT/name)
    print("wrote",ROOT/name)
