import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import lightgbm as lgb
from sklearn.metrics import fbeta_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.join(REPO_ROOT, "data")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.entity_representations import build_entity_representation
from scripts.optimize_country_thresholds import UnifiedTargetStore, compute_token_idf
from scripts.test_enhanced_blocker_v4 import SuperBlockerV4
from scripts.optimize_country_thresholds import EnhancedFeatureExtractorV3

class FuzzyWeightingNet(nn.Module):
    def __init__(self, num_quality_features, num_similarity_features):
        super().__init__()
        self.quality_mlp = nn.Sequential(
            nn.Linear(num_quality_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_similarity_features),
            nn.Sigmoid()
        )
        self.predictor = nn.Linear(num_similarity_features, 1)

    def forward(self, q, x):
        w = self.quality_mlp(q)
        z = w * x
        out = self.predictor(z)
        return out, z, w

def extract_quality_features(rep, target):
    q = []
    q.append(1.0 if not rep.name_raw else 0.0)
    q.append(1.0 if not target.name_raw else 0.0)
    q.append(abs(len(rep.name_raw) - len(target.name_raw)) / 100.0)
    q.append(1.0 if not rep.address_raw else 0.0)
    q.append(1.0 if not target.address_raw else 0.0)
    return np.array(q, dtype=np.float32)

def main():
    target_files = [os.path.join(WORKSPACE_DIR, "train_source2.tsv"), os.path.join(WORKSPACE_DIR, "train_source3.tsv")]
    idf_dict, default_idf = compute_token_idf()
    target_store = UnifiedTargetStore(target_files)
    sblocker = SuperBlockerV4(target_store, idf_dict, default_idf, max_cands=30)
    
    for f_idx, tf in enumerate(target_files):
        with open(tf, 'rb') as f:
            f.readline()
            byte_pos = f.tell()
            line = f.readline()
            while line:
                parts = line.decode('utf-8', errors='replace').rstrip('\r\n').split('\t')
                if len(parts) >= 4:
                    int_id = target_store.add_target(parts[0].strip(), f_idx, byte_pos)
                    sblocker.index_target(int_id, parts[1].strip(), parts[2].strip(), parts[3].strip())
                byte_pos = f.tell()
                line = f.readline()
                
    train_gt = defaultdict(list)
    with open(os.path.join(WORKSPACE_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts: train_gt[parts[0].strip()] = parts[1].strip().split(",") if len(parts) > 1 and parts[1].strip() else []

    train_queries = []
    with open(os.path.join(WORKSPACE_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                train_queries.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
                if len(train_queries) >= 1000: break

    X_sim, X_qual, y = [], [], []
    for eid, qn, qa, qc in train_queries:
        gt_matches = set(train_gt.get(eid, []))
        q_rep = build_entity_representation(eid, qn, qa, qc)
        cands = sblocker.retrieve(qn, qa, qc)
        cand_int_ids = set(cands)
        for gt_id in gt_matches:
            int_id = target_store.id_to_int.get(gt_id)
            if int_id is not None: cand_int_ids.add(int_id)
            
        c_objs = [target_store.get_target(cid) for cid in cand_int_ids]
        for ct in c_objs:
            is_match = 1.0 if ct.entity_id in gt_matches else 0.0
            feats = EnhancedFeatureExtractorV3.extract_features(q_rep, ct, idf_dict, default_idf)
            qual = extract_quality_features(q_rep, ct)
            X_sim.append(feats)
            X_qual.append(qual)
            y.append(is_match)

    X_sim = torch.tensor(np.array(X_sim, dtype=np.float32))
    X_qual = torch.tensor(np.array(X_qual, dtype=np.float32))
    y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)

    net = FuzzyWeightingNet(X_qual.shape[1], X_sim.shape[1])
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]))
    
    for epoch in range(15):
        optimizer.zero_grad()
        out, z, w = net(X_qual, X_sim)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        _, Z, _ = net(X_qual, X_sim)
    Z_np = Z.numpy()
    y_np = y_t.numpy().flatten()
    
    lgb_data = lgb.Dataset(Z_np, label=y_np)
    lgb_params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "verbose": -1}
    booster = lgb.train(lgb_params, lgb_data, num_boost_round=100)
    
    preds = booster.predict(Z_np)
    pred_labels = (preds > 0.85).astype(int)
    f05 = fbeta_score(y_np, pred_labels, beta=0.5)
    print(f"F0.5 Score: {f05:.4f}", flush=True)

if __name__ == "__main__":
    main()