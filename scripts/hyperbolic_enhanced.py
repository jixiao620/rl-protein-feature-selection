#!/usr/bin/env python3
# hypstructure_disease_embed.py
# HyperStructure-style hyperbolic embedding learning for disease subtree (root node_id = "90")
# No classification / flat loss: only CPCC + centering on node embeddings.

import os
import json
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Hyperbolic geometry utils (Poincaré ball, curvature c)
# -----------------------

EPS = 1e-6

def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    # Ensure ||x|| < 1/sqrt(c) - eps
    with torch.no_grad():
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        max_norm = (1.0 - eps) / (c ** 0.5)
        factor = torch.minimum(torch.ones_like(norm), max_norm / norm)
    return x * factor

def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    # Exponential map at origin for Poincaré ball
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
    sqrt_c = c ** 0.5
    factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return v * factor

def poincare_distance_matrix(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    # x: [N, D] in Poincaré ball
    x2 = torch.sum(x * x, dim=-1, keepdim=True)  # [N,1]
    # pairwise squared Euclidean
    diff = x.unsqueeze(1) - x.unsqueeze(0)        # [N,N,D]
    dist2 = torch.sum(diff * diff, dim=-1)        # [N,N]
    sqrt_c = c ** 0.5
    num = 2.0 * sqrt_c * dist2
    denom = (1.0 - c * x2).clamp_min(EPS) @ (1.0 - c * x2).clamp_min(EPS).transpose(0,1)
    z = 1.0 + num / denom.clamp_min(EPS)
    return torch.acosh(z.clamp_min(1.0 + EPS))

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    # x,y: [...,D]
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    cxy2 = 2 * c * xy
    num = (1 + cxy2 + c * y2) * x + (1 - c * x2) * y
    den = 1 + cxy2 + c * c * x2 * y2
    return num / den.clamp_min(EPS)

def poincare_mean(x: torch.Tensor, c: float = 1.0, iters: int = 10) -> torch.Tensor:
    # Approximate Karcher mean in Poincaré ball via iterative Möbius averaging.
    # x: [N, D] in ball
    mu = x.mean(dim=0, keepdim=True)  # Euclidean init
    mu = project_to_ball(mu, c=c)
    for _ in range(iters):
        diff = mobius_add(x, -mu, c=c)
        mu = mobius_add(mu, diff.mean(dim=0, keepdim=True), c=c)
        mu = project_to_ball(mu, c=c)
    return mu.squeeze(0)

# -----------------------
# Dataset: subtree rooted at node_id="90"
# -----------------------

class DiseaseTreeDataset:
    """
    Build label_map (string) and tree_dist for a disease subtree rooted at root_id.
    """
    def __init__(self, df: pd.DataFrame, root_id: str = "90"):
        self.root_id = str(root_id)

        # Ensure columns present and as str
        for col in ["node_id", "parent_id"]:
            if col not in df.columns:
                raise ValueError(f"Input DataFrame must contain column '{col}'")
            df[col] = df[col].astype(str)
        df["parent_id"] = df["parent_id"].fillna("")

        # Build parent->children map from full df, then BFS to get subtree
        children = defaultdict(list)
        for _, row in df.iterrows():
            p = row["parent_id"]
            c = row["node_id"]
            if p != "":
                children[p].append(c)

        if self.root_id not in children and self.root_id not in set(df["node_id"].tolist()):
            raise ValueError(f"root_id {self.root_id} not found in node_id or as a parent_id in the DataFrame")

        subtree_nodes = []
        q = deque([self.root_id])
        seen = set()
        while q:
            nid = q.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            subtree_nodes.append(nid)
            for ch in children.get(nid, []):
                q.append(ch)

        # restrict df to subtree
        sub_df = df[df["node_id"].isin(subtree_nodes)].copy()

        # build parent_map within subtree (ignore parents outside subtree)
        parent_map = {}
        for _, row in sub_df.iterrows():
            nid = row["node_id"]
            pid = row["parent_id"]
            if pid in subtree_nodes:
                parent_map[nid] = pid
        # root has no parent in this subtree
        if self.root_id in parent_map:
            del parent_map[self.root_id]

        self.parent_map = parent_map
        self.node_list = sorted(subtree_nodes)  # deterministic order
        self.node2idx = {n: i for i, n in enumerate(self.node_list)}

        # build paths from each node up to root_id
        paths = {}
        depths = {}
        for nid in self.node_list:
            cur = nid
            path = [cur]
            while cur in self.parent_map:
                cur = self.parent_map[cur]
                path.append(cur)
            # stop when no parent (should be root)
            paths[nid] = path  # [leaf, ..., root]
            depths[nid] = len(path) - 1
        self.depths = depths
        self.paths = paths

        max_depth = max(len(p) for p in paths.values())
        # label_map_str: shape [num_nodes, max_depth], padded with empty string
        label_map = []
        for nid in self.node_list:
            p = paths[nid]
            padded = p + [""] * (max_depth - len(p))
            label_map.append(padded)
        self.label_map = np.array(label_map, dtype=object)  # string array

        # build tree_dist dictionary between all label nodes (non-empty strings)
        self.tree_dist = self._build_tree_dist()

    def _build_tree_dist(self):
        # collect all non-empty label nodes (leaf + ancestors)
        nodes_set = set(self.label_map.flatten().tolist())
        nodes_set.discard("")
        nodes = sorted(nodes_set)

        # paths to root and depths for every node
        paths_to_root = {}
        depths = {}
        for n in nodes:
            cur = n
            path = [cur]
            # follow parent_map, but if n is an ancestor that does not appear in parent_map,
            # we still treat it as root-like
            while cur in self.parent_map:
                cur = self.parent_map[cur]
                path.append(cur)
            paths_to_root[n] = path
            depths[n] = len(path) - 1

        tree_dist = {}
        for i, a in enumerate(nodes):
            pa = paths_to_root[a]
            for j, b in enumerate(nodes):
                if j <= i:
                    continue
                pb = paths_to_root[b]
                # find deepest common ancestor
                lca = None
                max_depth = -1
                for d_a, anc in enumerate(pa):
                    if anc in pb and d_a > max_depth:
                        max_depth = d_a
                        lca = anc
                if lca is None:
                    # no common ancestor inside subtree; fall back to sum of depths
                    d = depths[a] + depths[b]
                else:
                    d = depths[a] + depths[b] - 2 * depths[lca]
                tree_dist[(a, b)] = d
                tree_dist[(b, a)] = d
        return tree_dist

# -----------------------
# CPCC loss (HypStructure 风格，poincare_mean 版本)
# -----------------------

class CPCCLoss(nn.Module):
    def __init__(self, dataset: DiseaseTreeDataset, c: float = 1.0):
        super().__init__()
        self.tree_dist = dataset.tree_dist                 # dict[(str,str)] -> distance
        self.label_map_str = dataset.label_map             # np array (num_nodes, depth)
        self.label_map_int, self.str2int, self.int2str = self.map_strings_to_integers(self.label_map_str)
        self.empty_int = self.str2int.get("", -1)
        self.tree_depth = self.label_map_str.shape[1]
        self.c = c

    def map_strings_to_integers(self, string_array: np.ndarray):
        string_to_int = {}
        int_to_string = {}
        current_id = 0
        h, w = string_array.shape
        int_array = np.zeros((h, w), dtype=np.int64)
        for i in range(h):
            for j in range(w):
                s = string_array[i, j]
                if s not in string_to_int:
                    string_to_int[s] = current_id
                    int_to_string[current_id] = s
                    current_id += 1
                int_array[i, j] = string_to_int[s]
        return int_array, string_to_int, int_to_string

    def dT(self, nodes: list, device: torch.device) -> torch.Tensor:
        # build vector of tree distances for all node pairs in `nodes`
        n = len(nodes)
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                a = nodes[i]
                b = nodes[j]
                d = self.tree_dist.get((a, b), 0.0)
                dists.append(d)
        return torch.tensor(dists, dtype=torch.float32, device=device)

    def forward(self, representations: torch.Tensor, targets_fine: torch.Tensor) -> torch.Tensor:
        """
        representations: [B, D] Euclidean embeddings of nodes in this batch
        targets_fine: [B] integer indices into label_map rows (node indices)
        """
        device = representations.device
        label_map = torch.tensor(self.label_map_int, device=device)  # [N, depth]
        targets = label_map[targets_fine]                            # [B, depth]

        # list of unique integer node IDs per level (excluding empty)
        all_unique_int = []
        for col in range(targets.shape[1]):
            col_vals = targets[:, col]
            mask = col_vals != self.empty_int
            if mask.any():
                all_unique_int.append(torch.unique(col_vals[mask]))
            else:
                all_unique_int.append(torch.tensor([], dtype=torch.long, device=device))

        # Poincaré embeddings of representations
        reps_p = expmap0(representations, c=self.c)

        all_unique_str = []
        proto_list = []

        for col, uniq_vals in enumerate(all_unique_int):
            for val in uniq_vals:
                mask = targets[:, col] == val
                if not mask.any():
                    continue
                points = reps_p[mask]  # [K, D]
                proto = poincare_mean(points, c=self.c)  # [D]
                proto_list.append(proto)
                all_unique_str.append(self.int2str[int(val.item())])

        if len(proto_list) < 2:
            # not enough nodes to form pairs; return neutral loss
            return torch.tensor(0.0, device=device)

        prototypes = torch.stack(proto_list, dim=0)  # [M, D]
        prototypes = project_to_ball(prototypes, c=self.c)

        # hyperbolic pairwise distances
        dB_matrix = poincare_distance_matrix(prototypes, c=self.c)
        idx = torch.triu_indices(dB_matrix.size(0), dB_matrix.size(1), offset=1)
        dB = dB_matrix[idx[0], idx[1]]

        # tree distances with the same ordering
        dT = self.dT(all_unique_str, device=device)
        if dT.numel() != dB.numel():
            # safeguard
            min_len = min(dT.numel(), dB.numel())
            dT = dT[:min_len]
            dB = dB[:min_len]

        # Pearson correlation
        stacked = torch.stack([dB, dT], dim=0)
        corr = torch.corrcoef(stacked)[0, 1]
        if torch.isnan(corr):
            return torch.tensor(1.0, device=device)
        return 1.0 - corr

# -----------------------
# Centering loss（和 HypStructure main.py 里逻辑一致）
# -----------------------

class CenteringLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb_p = expmap0(emb, c=self.c)
        center = poincare_mean(emb_p, c=self.c)
        return torch.norm(center, p=2)

# -----------------------
# Model: node embedding matrix
# -----------------------

class HyperbolicEmbeddingModel(nn.Module):
    def __init__(self, num_nodes: int, dim: int, c: float = 1.0):
        super().__init__()
        self.c = c
        self.z = nn.Parameter(torch.randn(num_nodes, dim) * 0.01)

    def forward(self) -> torch.Tensor:
        with torch.no_grad():
            self.z.data = project_to_ball(self.z.data, c=self.c)
        return self.z

# -----------------------
# Training loop（复刻 HypStructure：只保留 CPCC + Center）
# -----------------------

def train_disease_subtree(
    tsv_path: str,
    root_id: str = "90",
    dim: int = 10,
    c: float = 1.0,
    alpha_cpcc: float = 1.0,
    beta_center: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    save_path: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    dataset = DiseaseTreeDataset(df, root_id=str(root_id))

    num_nodes = len(dataset.node_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HyperbolicEmbeddingModel(num_nodes, dim=dim, c=c).to(device)
    cpcc = CPCCLoss(dataset, c=c).to(device)
    center_loss = CenteringLoss(c=c).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_indices = torch.arange(num_nodes, dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = all_indices[torch.randperm(num_nodes)]
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_nodes, batch_size):
            batch_idx = perm[i : i + batch_size]
            if batch_idx.numel() < 2:
                continue

            emb = model()                 # [N, D]
            batch_rep = emb[batch_idx]    # [B, D]

            loss_cpcc = cpcc(batch_rep, batch_idx)
            loss_center = center_loss(batch_rep)
            loss = alpha_cpcc * loss_cpcc + beta_center * loss_center

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            epoch_loss /= n_batches
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss {epoch_loss:.4f}")

    final_emb = model().detach().cpu()
    result = {
        "node_list": dataset.node_list,
        "embeddings": final_emb.numpy().tolist(),
    }
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f)
    return dataset.node_list, final_emb

# =====================================================
# Visualization utilities
# =====================================================
import matplotlib.pyplot as plt

def poincare_to_2d(z, c=1.0):
    """
    Convert D-dimensional Poincaré ball embeddings to 2D.
    Use simple Euclidean PCA, then project back to disk.
    """
    from sklearn.decomposition import PCA

    # PCA to 2D
    z_np = z.detach().cpu().numpy()
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z_np)

    # ensure inside Poincaré ball (norm < 1/sqrt(c))
    r = np.linalg.norm(z2, axis=1, keepdims=True) + 1e-9
    max_r = (1.0 / np.sqrt(c)) - 1e-6
    scale = np.minimum(1.0, max_r / r)
    z2 = z2 * scale
    return z2


def plot_poincare_disk(z, nodes, depths, parent_map, save_path=None, c=1.0):
    """
    Draw Poincaré 2D disk with nodes colored by depth.
    """
    z2 = poincare_to_2d(z, c=c)

    depth_list = np.array([depths[n] for n in nodes])
    unique_depths = sorted(list(set(depth_list)))
    cmap = plt.get_cmap("viridis", len(unique_depths))

    plt.figure(figsize=(7, 7))

    # Draw Poincaré disk boundary
    circle = plt.Circle((0, 0), 1.0 / np.sqrt(c), color="black", fill=False)
    plt.gca().add_artist(circle)

    # Scatter nodes by depth
    for d in unique_depths:
        idx = np.where(depth_list == d)[0]
        plt.scatter(
            z2[idx, 0], z2[idx, 1],
            s=40 if d > 0 else 100,
            color=cmap(d),
            label=f"depth {d}"
        )

    # Highlight root (depth=0)
    root_idx = np.where(depth_list == 0)[0]
    if len(root_idx) > 0:
        plt.scatter(z2[root_idx,0], z2[root_idx,1], color="red", s=120, marker="*", label="root")

    plt.title("Poincaré Disk Visualization (2D Projection)")
    plt.legend()
    plt.axis("equal")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved Poincaré 2D Plot to:", save_path)

    plt.close()


def plot_depth_radius(z, nodes, depths, save_path=None, c=1.0):
    """
    Plot node radius as function of depth.
    Should see increasing radius with depth.
    """
    # convert to Poincaré ball
    z_p = expmap0(z, c=c).detach().cpu().numpy()

    radius = np.linalg.norm(z_p, axis=1)
    depth_list = np.array([depths[n] for n in nodes])

    plt.figure(figsize=(6, 4))
    plt.scatter(depth_list, radius, s=50, alpha=0.7)
    plt.xlabel("Tree Depth")
    plt.ylabel("Poincaré Radius ||x||")
    plt.title("Depth vs Radius")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved Depth-Radius Plot to:", save_path)

    plt.close()

def compute_top_branch(nodes, parent_map, root_id="90"):
    """
    Determine each node's top-level branch.
    root_id branch = itself ("root")
    child of root → branch = itself
    deeper nodes → use the ancestor which is direct child of root
    """
    top_branch = {}
    for n in nodes:
        cur = n
        prev = None
        # climb to root
        while cur in parent_map:
            prev = cur
            cur = parent_map[cur]
        # after loop:
        # cur should be root or ancestor
        if cur == root_id and prev is not None:
            top_branch[n] = prev  # direct child of root
        else:
            # for root itself
            top_branch[n] = n
    return top_branch


def plot_poincare_disk_with_branches(z, nodes, depths, parent_map,
                                     root_id="90",
                                     save_path=None, c=1.0):
    """
    Same as before, but:
    - Color by top-level branch
    - Draw parent-child edges
    """
    z2 = poincare_to_2d(z, c=c)
    z_p = expmap0(z, c=c).detach().cpu().numpy()  # true poincare coords for radius check

    # compute top-level branch id for color coding
    top_branch = compute_top_branch(nodes, parent_map, root_id)
    branches = sorted(list(set(top_branch.values())))

    branch_to_color = {
        br: plt.get_cmap("tab20")(i % 20)
        for i, br in enumerate(branches)
    }

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Draw disk boundary
    R = 1.0 / np.sqrt(c)
    circle = plt.Circle((0, 0), R, color="black", fill=False)
    ax.add_artist(circle)

    # Node ID → index in z2
    node2idx = {n: i for i, n in enumerate(nodes)}

    # ---- Draw parent–child edges ----
    for child, parent in parent_map.items():
        if child in node2idx and parent in node2idx:
            i = node2idx[child]
            j = node2idx[parent]
            x1, y1 = z2[i]
            x2, y2 = z2[j]
            plt.plot([x1, x2], [y1, y2], color="gray", linewidth=0.6, alpha=0.7)

    # ---- Draw nodes ----
    for n in nodes:
        idx = node2idx[n]
        d = depths[n]
        br = top_branch[n]
        col = branch_to_color[br]

        size = 120 if n == root_id else 40
        marker = "*" if n == root_id else "o"

        plt.scatter(z2[idx, 0], z2[idx, 1],
                    s=size,
                    color=col,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.5)

    plt.title("Poincaré Disk (Branch-colored + Parent-Child Edges)")
    plt.axis("equal")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved:", save_path)
    plt.close()


if __name__ == "__main__":
    NODE_FILE = "/home/jayding/feature_selection/data/datacode-19.tsv"
    SAVE_JSON = "/home/jayding/feature_selection/results_disease90/hypstructure_disease90_embeddings.json"
    SAVE_DIR = "/home/jayding/feature_selection/results_disease90/plots"
    os.makedirs(SAVE_DIR, exist_ok=True)

    nodes, emb = train_disease_subtree(
        tsv_path=NODE_FILE,
        root_id="90",
        dim=30,
        c=5.0,
        alpha_cpcc=5.0,
        beta_center=0.01,
        epochs=1500,
        batch_size=256,
        lr=5e-4,
        seed=42,
        save_path=SAVE_JSON,
    )

    # ---- Visualization ----
    # Reconstruct dataset for depths and parent_map
    df = pd.read_csv(NODE_FILE, sep="\t", dtype=str)
    dataset = DiseaseTreeDataset(df, root_id="90")

    plot_poincare_disk(
        z=emb,
        nodes=dataset.node_list,
        depths=dataset.depths,
        parent_map=dataset.parent_map,
        save_path=os.path.join(SAVE_DIR, "poincare_2d.png"),
        c=1.0
    )

    plot_depth_radius(
        z=emb,
        nodes=dataset.node_list,
        depths=dataset.depths,
        save_path=os.path.join(SAVE_DIR, "depth_radius.png"),
        c=1.0
    )

    plot_poincare_disk_with_branches(
    z=emb,
    nodes=dataset.node_list,
    depths=dataset.depths,
    parent_map=dataset.parent_map,
    root_id="90",
    save_path=os.path.join(SAVE_DIR, "poincare_branches.png"),
    c=1.0
    )


    print("Visualization complete!")
