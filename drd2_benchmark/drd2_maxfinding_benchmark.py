from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from tdc import Oracle
from tdc.generation import MolGen

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@dataclass
class GraphMol:
    node_feat: torch.Tensor
    adj: torch.Tensor
    num_nodes: int


def atom_features(atom: Chem.Atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge()
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    hybrid = atom.GetHybridization()
    hyb_sp = 1.0 if hybrid == Chem.rdchem.HybridizationType.SP else 0.0
    hyb_sp2 = 1.0 if hybrid == Chem.rdchem.HybridizationType.SP2 else 0.0
    hyb_sp3 = 1.0 if hybrid == Chem.rdchem.HybridizationType.SP3 else 0.0
    total_h = atom.GetTotalNumHs()
    return [
        atomic_num / 100.0,
        degree / 6.0,
        formal_charge / 5.0,
        aromatic,
        hyb_sp,
        hyb_sp2,
        hyb_sp3,
        total_h / 8.0,
    ]


def smiles_to_graph(smiles: str) -> GraphMol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_nodes = mol.GetNumAtoms()
    if num_nodes == 0:
        return None

    node_feat = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
    )
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    adj = adj + torch.eye(num_nodes, dtype=torch.float32)

    return GraphMol(node_feat=node_feat, adj=adj, num_nodes=num_nodes)


def collate_graphs(
    graphs: Sequence[GraphMol], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_nodes = max(g.num_nodes for g in graphs)
    feat_dim = graphs[0].node_feat.shape[1]
    batch_size = len(graphs)

    x = torch.zeros(
        (batch_size, max_nodes, feat_dim), dtype=torch.float32, device=device
    )
    adj = torch.zeros(
        (batch_size, max_nodes, max_nodes), dtype=torch.float32, device=device
    )
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.float32, device=device)

    for idx, g in enumerate(graphs):
        n = g.num_nodes
        node_feat = (
            g.node_feat
            if g.node_feat.device == device
            else g.node_feat.to(device, non_blocking=True)
        )
        adj_mat = (
            g.adj if g.adj.device == device else g.adj.to(device, non_blocking=True)
        )
        x[idx, :n, :] = node_feat
        adj[idx, :n, :n] = adj_mat
        mask[idx, :n] = 1.0

    deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adj = adj / deg
    return x, adj, mask


class GraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.bmm(adj, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConvLayer(hidden_dim, dropout) for _ in range(depth)]
        )
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, emb = self.forward_logits(x, adj, mask)
        y_hat = torch.sigmoid(logits)
        return y_hat, emb

    def forward_logits(
        self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            # Residual update mitigates over-smoothing in deeper message passing.
            h = h + layer(h, adj)

        mask_exp = mask.unsqueeze(-1)
        h = h * mask_exp
        pooled = h.sum(dim=1) / mask_exp.sum(dim=1).clamp_min(1.0)
        emb = F.relu(self.readout(pooled))
        logits = self.out(emb)
        return logits.squeeze(-1), emb


@dataclass
class BenchmarkConfig:
    seed: int = 123
    pool_size: int = 2500
    seed_size: int = 64
    rounds: int = 15
    batch_k: int = 32
    train_epochs: int = 25
    train_batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 384
    depth: int = 3
    dropout: float = 0.1
    prior_mean: float = 0.95
    shortlist_size: int = 5000
    dpp_candidates: int = 256
    tau: float = 0.08
    ucb_beta: float = 1.0
    ucb_mc_passes: int = 16
    pred_batch_size: int = 2048
    use_amp: bool = True
    use_tf32: bool = True
    cache_graphs_on_device: bool = True
    auto_tune_cuda_batch_sizes: bool = True
    ucb_uncertainty_collapse_eps: float = 1e-6
    flat_score_std_eps: float = 1e-6
    log_level: str = "INFO"


def method_seed(base_seed: int, method: str) -> int:
    # Deterministic method-specific offset to avoid coupling trajectories.
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(method))
    return int(base_seed + offset)


@dataclass
class MethodResult:
    method: str
    queries: List[int]
    best_so_far: List[float]
    simple_regret: List[float]
    threshold_q95: int | None
    threshold_q98: int | None
    diversity_mean_similarity: float


def load_candidate_pool(pool_size: int, seed: int) -> List[str]:
    data = MolGen(name="ZINC").get_data()
    if "smiles" not in data.columns:
        raise ValueError("Expected a smiles column in TDC MolGen ZINC dataset")

    smiles_raw = data["smiles"].dropna().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(smiles_raw)

    dedup: List[str] = []
    seen = set()
    invalid_smiles = 0
    duplicate_smiles = 0
    for smi in smiles_raw:
        can = canonicalize_smiles(smi)
        if can is None:
            invalid_smiles += 1
            continue
        if can in seen:
            duplicate_smiles += 1
            continue
        seen.add(can)
        dedup.append(can)
        if len(dedup) >= pool_size:
            break

    if invalid_smiles > 0 or duplicate_smiles > 0:
        logger.warning(
            "Candidate pool preprocessing filtered entries: invalid=%d duplicate=%d"
            " kept=%d",
            invalid_smiles,
            duplicate_smiles,
            len(dedup),
        )

    if len(dedup) < pool_size:
        raise ValueError(
            f"Only found {len(dedup)} valid unique molecules, requested {pool_size}"
        )
    return dedup


def build_graph_cache(smiles_pool: Sequence[str]) -> List[GraphMol]:
    graphs: List[GraphMol] = []
    for smi in smiles_pool:
        g = smiles_to_graph(smi)
        if g is None:
            raise ValueError(f"Failed to featurize SMILES: {smi}")
        graphs.append(g)
    return graphs


def score_pool_with_oracle(smiles_pool: Sequence[str]) -> np.ndarray:
    oracle = Oracle(name="DRD2")
    scores = np.array([float(oracle(smi)) for smi in smiles_pool], dtype=np.float32)
    nonfinite = ~np.isfinite(scores)
    if np.any(nonfinite):
        logger.warning(
            "Oracle returned non-finite scores: count=%d (replacing with 0.0)",
            int(np.sum(nonfinite)),
        )
        scores = np.where(nonfinite, 0.0, scores)
    return np.clip(scores, 0.0, 1.0)


def maybe_move_graph_cache_to_device(
    graph_cache: Sequence[GraphMol],
    cfg: BenchmarkConfig,
    device: torch.device,
) -> List[GraphMol]:
    if device.type != "cuda" or not cfg.cache_graphs_on_device:
        return list(graph_cache)

    moved: List[GraphMol] = []
    for g in graph_cache:
        moved.append(
            GraphMol(
                node_feat=g.node_feat.to(device, non_blocking=True),
                adj=g.adj.to(device, non_blocking=True),
                num_nodes=g.num_nodes,
            )
        )
    return moved


def tune_cuda_batch_sizes(
    cfg: BenchmarkConfig, device: torch.device
) -> BenchmarkConfig:
    if device.type != "cuda" or not cfg.auto_tune_cuda_batch_sizes:
        return cfg

    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    tuned = BenchmarkConfig(**vars(cfg))

    if total_gb >= 20:
        tuned.train_batch_size = max(tuned.train_batch_size, 3072)
        tuned.pred_batch_size = max(tuned.pred_batch_size, 8192)
    elif total_gb >= 12:
        tuned.train_batch_size = max(tuned.train_batch_size, 2048)
        tuned.pred_batch_size = max(tuned.pred_batch_size, 4096)
    elif total_gb >= 10:
        tuned.train_batch_size = max(tuned.train_batch_size, 1536)
        tuned.pred_batch_size = max(tuned.pred_batch_size, 4096)
    elif total_gb >= 6:
        tuned.train_batch_size = max(tuned.train_batch_size, 1024)
        tuned.pred_batch_size = max(tuned.pred_batch_size, 3072)
    else:
        tuned.train_batch_size = max(tuned.train_batch_size, 512)
        tuned.pred_batch_size = max(tuned.pred_batch_size, 1024)

    return tuned


def train_model(
    model: GNN,
    graph_cache: Sequence[GraphMol],
    labels: np.ndarray,
    train_idx: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> None:
    model.train()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.grad_scaler.GradScaler(
        enabled=(device.type == "cuda" and cfg.use_amp)
    )

    for _ in range(cfg.train_epochs):
        np.random.shuffle(train_idx)
        for start in range(0, len(train_idx), cfg.train_batch_size):
            batch_idx = train_idx[start : start + cfg.train_batch_size]
            batch_graphs = [graph_cache[i] for i in batch_idx]
            x, adj, mask = collate_graphs(batch_graphs, device)
            y_true = torch.tensor(labels[batch_idx], dtype=torch.float32, device=device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and cfg.use_amp),
            ):
                y_pred, _ = model(x, adj, mask)
                loss = F.mse_loss(y_pred, y_true)

            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite loss detected during training; skipping optimizer step."
                    " batch_start=%d batch_size=%d",
                    start,
                    len(batch_idx),
                )
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


def predict_mean_and_emb(
    model: GNN,
    graph_cache: Sequence[GraphMol],
    idx: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    means: List[np.ndarray] = []
    embs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(idx), cfg.pred_batch_size):
            batch_idx = idx[start : start + cfg.pred_batch_size]
            batch_graphs = [graph_cache[i] for i in batch_idx]
            x, adj, mask = collate_graphs(batch_graphs, device)
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and cfg.use_amp),
            ):
                y, emb = model(x, adj, mask)
            means.append(y.detach().float().cpu().numpy())
            embs.append(emb.detach().float().cpu().numpy())

    return np.concatenate(means), np.concatenate(embs)


def predict_logits_and_emb(
    model: GNN,
    graph_cache: Sequence[GraphMol],
    idx: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: List[np.ndarray] = []
    embs: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(idx), cfg.pred_batch_size):
            batch_idx = idx[start : start + cfg.pred_batch_size]
            batch_graphs = [graph_cache[i] for i in batch_idx]
            x, adj, mask = collate_graphs(batch_graphs, device)
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda" and cfg.use_amp),
            ):
                logit, emb = model.forward_logits(x, adj, mask)
            logits.append(logit.detach().float().cpu().numpy())
            embs.append(emb.detach().float().cpu().numpy())

    return np.concatenate(logits), np.concatenate(embs)


def predict_mc_dropout(
    model: GNN,
    graph_cache: Sequence[GraphMol],
    idx: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    model.train()
    with torch.no_grad():
        for _ in range(cfg.ucb_mc_passes):
            pass_preds = []
            for start in range(0, len(idx), cfg.pred_batch_size):
                batch_idx = idx[start : start + cfg.pred_batch_size]
                batch_graphs = [graph_cache[i] for i in batch_idx]
                x, adj, mask = collate_graphs(batch_graphs, device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda" and cfg.use_amp),
                ):
                    y, _ = model(x, adj, mask)
                pass_preds.append(y.detach().float().cpu().numpy())
            preds.append(np.concatenate(pass_preds))
    pred_mat = np.stack(preds, axis=0)
    if not np.all(np.isfinite(pred_mat)):
        logger.warning(
            "MC-dropout predictions contain non-finite values: nan=%d +inf=%d -inf=%d",
            int(np.isnan(pred_mat).sum()),
            int(np.isposinf(pred_mat).sum()),
            int(np.isneginf(pred_mat).sum()),
        )
    return pred_mat.mean(axis=0), pred_mat.std(axis=0)


def predict_mc_dropout_logits(
    model: GNN,
    graph_cache: Sequence[GraphMol],
    idx: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    model.train()
    with torch.no_grad():
        for _ in range(cfg.ucb_mc_passes):
            pass_preds = []
            for start in range(0, len(idx), cfg.pred_batch_size):
                batch_idx = idx[start : start + cfg.pred_batch_size]
                batch_graphs = [graph_cache[i] for i in batch_idx]
                x, adj, mask = collate_graphs(batch_graphs, device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda" and cfg.use_amp),
                ):
                    logit, _ = model.forward_logits(x, adj, mask)
                pass_preds.append(logit.detach().float().cpu().numpy())
            preds.append(np.concatenate(pass_preds))
    pred_mat = np.stack(preds, axis=0)
    if not np.all(np.isfinite(pred_mat)):
        logger.warning(
            "MC-dropout logit predictions contain non-finite values: nan=%d +inf=%d"
            " -inf=%d",
            int(np.isnan(pred_mat).sum()),
            int(np.isposinf(pred_mat).sum()),
            int(np.isneginf(pred_mat).sum()),
        )
    return pred_mat.mean(axis=0), pred_mat.std(axis=0)


def avg_pairwise_similarity(emb: np.ndarray, sigma: float | None = None) -> float:
    if len(emb) < 2:
        return 0.0
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1)
    if sigma is None:
        sigma = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    s = np.exp(-(d**2) / (sigma**2 + 1e-8))
    tri = np.triu_indices(len(emb), k=1)
    return float(np.mean(s[tri]))


def greedy_k_dpp(
    scores: np.ndarray,
    emb: np.ndarray,
    k: int,
    tau: float,
) -> np.ndarray:
    m = len(scores)
    if m <= k:
        logger.warning(
            "k-DPP candidate pool smaller than requested batch: m=%d k=%d; returning"
            " all.",
            m,
            k,
        )
        return np.arange(m, dtype=int)

    if not np.all(np.isfinite(scores)):
        logger.warning(
            "k-DPP scores contain non-finite values: nan=%d +inf=%d -inf=%d",
            int(np.isnan(scores).sum()),
            int(np.isposinf(scores).sum()),
            int(np.isneginf(scores).sum()),
        )
        scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)

    if not np.all(np.isfinite(emb)):
        logger.warning(
            "k-DPP embeddings contain non-finite values: nan=%d +inf=%d -inf=%d",
            int(np.isnan(emb).sum()),
            int(np.isposinf(emb).sum()),
            int(np.isneginf(emb).sum()),
        )
        emb = np.nan_to_num(emb, nan=0.0, posinf=1e4, neginf=-1e4)

    dist = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1)
    sigma = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    if sigma <= 1e-12:
        logger.warning(
            "k-DPP similarity scale sigma is very small (sigma=%.3e); selection may be"
            " unstable.",
            sigma,
        )
    s = np.exp(-(dist**2) / (sigma**2 + 1e-8))

    stabilized = (scores - scores.max()) / max(tau, 1e-4)
    q = np.exp(np.clip(stabilized, -30.0, 30.0))
    l = (q[:, None] * s) * q[None, :]
    l += np.eye(m) * 1e-6

    selected: List[int] = []
    remaining = set(range(m))
    non_pos_def_hits = 0
    for _ in range(k):
        best_i = None
        best_val = -np.inf
        for i in remaining:
            idx = selected + [i]
            sub = l[np.ix_(idx, idx)]
            sign, logdet = np.linalg.slogdet(sub)
            val = logdet if sign > 0 else -1e18
            if sign <= 0:
                non_pos_def_hits += 1
            if val > best_val:
                best_val = val
                best_i = i
        if best_i is None:
            logger.warning(
                "k-DPP greedy selection terminated early at size=%d (requested k=%d).",
                len(selected),
                k,
            )
            break
        selected.append(best_i)
        remaining.remove(best_i)

    if non_pos_def_hits > 0:
        logger.warning(
            "k-DPP encountered non-positive-definite submatrices during greedy search:"
            " count=%d",
            non_pos_def_hits,
        )
    return np.array(selected, dtype=int)


def queries_to_threshold(
    best_so_far: Sequence[float], queries: Sequence[int], threshold: float
) -> int | None:
    for b, q in zip(best_so_far, queries):
        if b >= threshold:
            return int(q)
    return None


def stable_topk_indices(
    scores: np.ndarray,
    k: int,
    rng: np.random.Generator,
    collapse_std_eps: float = 1e-8,
    jitter_std: float = 1e-6,
) -> np.ndarray:
    if k <= 0 or len(scores) == 0:
        logger.debug(
            "Top-k request has empty input or non-positive k: len(scores)=%d k=%d",
            len(scores),
            k,
        )
        return np.array([], dtype=int)
    if k >= len(scores):
        # Common and expected when shortlist size exceeds current pool size.
        return np.arange(len(scores), dtype=int)

    s = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(s)
    nonfinite_count = int(np.sum(~finite_mask))
    if nonfinite_count > 0:
        logger.warning(
            "Top-k scores contain non-finite values: total=%d nan=%d +inf=%d -inf=%d",
            nonfinite_count,
            int(np.isnan(s).sum()),
            int(np.isposinf(s).sum()),
            int(np.isneginf(s).sum()),
        )
    if not np.any(finite_mask):
        logger.warning(
            "All top-k scores are non-finite; falling back to uniform random choice for"
            " k=%d.",
            k,
        )
        return rng.choice(np.arange(len(s)), size=k, replace=False)

    safe = np.where(finite_mask, s, -np.inf)
    finite_vals = safe[np.isfinite(safe)]
    finite_std = float(np.nanstd(finite_vals))
    if finite_std < collapse_std_eps:
        logger.debug(
            "Top-k score landscape is flat/collapsed (std=%.3e < %.3e, min=%.6f,"
            " max=%.6f); using random fallback.",
            finite_std,
            collapse_std_eps,
            float(np.nanmin(finite_vals)),
            float(np.nanmax(finite_vals)),
        )
        return rng.choice(np.arange(len(safe)), size=k, replace=False)

    noisy = safe + rng.normal(loc=0.0, scale=jitter_std, size=len(safe))
    return np.argpartition(-noisy, k - 1)[:k]


def run_method(
    method: str,
    graph_cache: Sequence[GraphMol],
    true_scores: np.ndarray,
    cfg: BenchmarkConfig,
    device: torch.device,
    init_idx: np.ndarray | None = None,
) -> MethodResult:
    n = len(true_scores)
    seed = method_seed(cfg.seed, method)
    rng = np.random.default_rng(seed)
    set_seed(seed)
    all_idx = np.arange(n)

    if init_idx is None:
        init_idx = rng.choice(all_idx, size=cfg.seed_size, replace=False)
    else:
        init_idx = np.asarray(init_idx, dtype=int)
    queried = set(int(i) for i in init_idx.tolist())

    queries = [cfg.seed_size]
    best = [float(true_scores[init_idx].max())]
    diversity_vals: List[float] = []
    warning_once: set[str] = set()

    def warn_once(key: str, msg: str, *args: object) -> None:
        if key in warning_once:
            return
        warning_once.add(key)
        logger.warning(msg, *args)

    for round_idx in range(cfg.rounds):
        t0 = time.time()
        train_idx = np.array(sorted(queried), dtype=int)
        unlabeled_idx = np.array(sorted(set(all_idx.tolist()) - queried), dtype=int)

        model = GNN(
            in_dim=graph_cache[0].node_feat.shape[1],
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            dropout=cfg.dropout,
        ).to(device)

        train_model(model, graph_cache, true_scores, train_idx, cfg, device)

        if method == "high_prior_dpp":
            mean_logit, emb = predict_logits_and_emb(
                model, graph_cache, unlabeled_idx, cfg, device
            )
            mean_std = float(np.nanstd(mean_logit))
            if mean_std < cfg.flat_score_std_eps:
                warn_once(
                    f"{method}:flat_mean",
                    "[%s] round=%d predictive logit mean appears flat before optimism"
                    " transform: std=%.3e min=%.6f max=%.6f",
                    method,
                    round_idx + 1,
                    mean_std,
                    float(np.nanmin(mean_logit)),
                    float(np.nanmax(mean_logit)),
                )
            beta0 = float(math.log(cfg.prior_mean / (1.0 - cfg.prior_mean)))
            optimistic_logit = mean_logit + beta0
            optimistic_std = float(np.nanstd(optimistic_logit))
            if optimistic_std < cfg.flat_score_std_eps:
                warn_once(
                    f"{method}:flat_optimistic",
                    "[%s] round=%d optimistic acquisition logit score is flat: "
                    "std=%.3e min=%.6f max=%.6f beta0=%.6f",
                    method,
                    round_idx + 1,
                    optimistic_std,
                    float(np.nanmin(optimistic_logit)),
                    float(np.nanmax(optimistic_logit)),
                    beta0,
                )
            shortlist_n = min(cfg.shortlist_size, len(unlabeled_idx))
            top_short = stable_topk_indices(optimistic_logit, shortlist_n, rng)
            short_idx = unlabeled_idx[top_short]
            short_mean = optimistic_logit[top_short]
            short_emb = emb[top_short]

            dpp_n = min(cfg.dpp_candidates, len(short_idx))
            top_dpp = stable_topk_indices(short_mean, dpp_n, rng)
            dpp_pool_idx = short_idx[top_dpp]
            dpp_pool_scores = short_mean[top_dpp]
            dpp_pool_emb = short_emb[top_dpp]

            picked_local = greedy_k_dpp(
                dpp_pool_scores, dpp_pool_emb, cfg.batch_k, cfg.tau
            )
            batch_idx = dpp_pool_idx[picked_local]

            if len(batch_idx) < min(cfg.batch_k, len(unlabeled_idx)):
                warn_once(
                    f"{method}:short_batch",
                    "[%s] round=%d selected fewer points than batch_k: selected=%d"
                    " requested=%d; filling with random remaining points.",
                    method,
                    round_idx + 1,
                    len(batch_idx),
                    cfg.batch_k,
                )
                remaining = np.array(
                    sorted(set(unlabeled_idx.tolist()) - set(batch_idx.tolist())),
                    dtype=int,
                )
                need = min(cfg.batch_k - len(batch_idx), len(remaining))
                if need > 0:
                    fill = rng.choice(remaining, size=need, replace=False)
                    batch_idx = np.concatenate([batch_idx, fill])

            if len(picked_local) > 1:
                diversity_vals.append(
                    avg_pairwise_similarity(dpp_pool_emb[picked_local])
                )

        elif method == "random":
            batch_idx = rng.choice(
                unlabeled_idx, size=min(cfg.batch_k, len(unlabeled_idx)), replace=False
            )

        elif method == "ucb_mc_dropout":
            mu, std = predict_mc_dropout_logits(
                model, graph_cache, unlabeled_idx, cfg, device
            )
            score = mu + cfg.ucb_beta * std
            mu_std = float(np.nanstd(mu))
            unc_std = float(np.nanstd(std))
            score_std = float(np.nanstd(score))
            if mu_std < cfg.flat_score_std_eps:
                warn_once(
                    f"{method}:flat_mean",
                    "[%s] round=%d predictive logit mean is flat: std=%.3e min=%.6f"
                    " max=%.6f",
                    method,
                    round_idx + 1,
                    mu_std,
                    float(np.nanmin(mu)),
                    float(np.nanmax(mu)),
                )
            if float(np.nanmax(std)) <= cfg.ucb_uncertainty_collapse_eps:
                warn_once(
                    f"{method}:unc_collapsed",
                    "[%s] round=%d MC-dropout logit uncertainty collapsed: max_std=%.3e"
                    " <= %.3e",
                    method,
                    round_idx + 1,
                    float(np.nanmax(std)),
                    cfg.ucb_uncertainty_collapse_eps,
                )
            if unc_std < cfg.flat_score_std_eps:
                warn_once(
                    f"{method}:unc_low_var",
                    "[%s] round=%d predictive logit uncertainty has low variation:"
                    " std(std)=%.3e",
                    method,
                    round_idx + 1,
                    unc_std,
                )
            if score_std < cfg.flat_score_std_eps:
                warn_once(
                    f"{method}:flat_score",
                    "[%s] round=%d UCB acquisition logit score is flat: std=%.3e"
                    " min=%.6f max=%.6f",
                    method,
                    round_idx + 1,
                    score_std,
                    float(np.nanmin(score)),
                    float(np.nanmax(score)),
                )
            k = min(cfg.batch_k, len(unlabeled_idx))
            top = stable_topk_indices(score, k, rng)
            batch_idx = unlabeled_idx[top]

        else:
            raise ValueError(f"Unknown method: {method}")

        for i in batch_idx.tolist():
            queried.add(int(i))

        curr_best = float(true_scores[np.array(sorted(queried), dtype=int)].max())
        best.append(curr_best)
        queries.append(len(queried))

        dt = time.time() - t0
        print(
            f"[{method}] round={round_idx + 1}/{cfg.rounds} queried={len(queried)}"
            f" best={curr_best:.4f} time={dt:.1f}s"
        )

    pool_max = float(true_scores.max())
    regret = [pool_max - b for b in best]
    q95 = queries_to_threshold(best, queries, 0.95 * pool_max)
    q98 = queries_to_threshold(best, queries, 0.98 * pool_max)

    return MethodResult(
        method=method,
        queries=queries,
        best_so_far=best,
        simple_regret=regret,
        threshold_q95=q95,
        threshold_q98=q98,
        diversity_mean_similarity=(
            float(np.mean(diversity_vals)) if diversity_vals else float("nan")
        ),
    )


def run_benchmark(
    cfg: BenchmarkConfig, methods: Sequence[str] | None = None
) -> Dict[str, object]:
    level_name = str(cfg.log_level).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(log_level)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and cfg.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    cfg = tune_cuda_batch_sizes(cfg, device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        print(
            "gpu_tuned "
            f"train_batch_size={cfg.train_batch_size} "
            f"pred_batch_size={cfg.pred_batch_size} "
            f"amp={cfg.use_amp} tf32={cfg.use_tf32}"
        )

    smiles_pool = load_candidate_pool(cfg.pool_size, cfg.seed)
    graph_cache = build_graph_cache(smiles_pool)
    graph_cache = maybe_move_graph_cache_to_device(graph_cache, cfg, device)
    true_scores = score_pool_with_oracle(smiles_pool)
    shared_init_idx = np.random.default_rng(cfg.seed).choice(
        np.arange(len(true_scores)), size=cfg.seed_size, replace=False
    )

    methods = (
        list(methods)
        if methods is not None
        else [
            "high_prior_dpp",
            "random",
            "ucb_mc_dropout",
        ]
    )
    results = [
        run_method(m, graph_cache, true_scores, cfg, device, init_idx=shared_init_idx)
        for m in methods
    ]

    return {
        "config": cfg,
        "smiles_pool": smiles_pool,
        "scores": true_scores,
        "results": results,
    }
