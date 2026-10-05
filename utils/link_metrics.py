"""Phase 1 metrics helpers extracted unchanged from the shared notebook.

See docs/phase1_cleanup.md for task-specific semantic differences.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def lp_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError('Labels/scores phải là vector cùng độ dài, không rỗng')
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError('Labels/scores có NaN/Inf')
    if set(np.unique(labels)) != {0,1}:
        raise ValueError('Metric yêu cầu đủ hai lớp nhị phân 0/1')
    return {'roc_auc':float(roc_auc_score(labels,scores)), 'average_precision':float(average_precision_score(labels,scores)),
            'n_positive':int(labels.sum()), 'n_negative':int((labels==0).sum())}

def cosine_scores(features, pairs, epsilon=1e-12, chunk_size=256):
    x, e = np.asarray(features,dtype=np.float64), np.asarray(pairs)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features phải là ma trận hữu hạn')
    if e.ndim != 2 or e.shape[0] != 2 or e.dtype.kind not in 'iu':
        raise ValueError('Pairs phải có shape [2,M], integer')
    if not e.shape[1]:
        raise ValueError('Không chấm điểm tập pairs rỗng')
    if e.min() < 0 or e.max() >= len(x):
        raise ValueError('Pair index ngoài miền')
    if not np.isfinite(epsilon) or epsilon <= 0 or not isinstance(chunk_size,int) or chunk_size <= 0:
        raise ValueError('epsilon/chunk_size không hợp lệ')
    scores = []
    for start in range(0,e.shape[1],chunk_size):
        u,v = x[e[0,start:start+chunk_size]], x[e[1,start:start+chunk_size]]
        den = np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),epsilon)
        scores.append(np.sum(u*v,axis=1)/den)
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError('Cosine overflow: features cần rescale')
    return result
