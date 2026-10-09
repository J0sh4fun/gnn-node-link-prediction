"""Phase 1 graph_ops helpers extracted unchanged from the shared notebook.

See docs/phase1_cleanup.md for task-specific semantic differences.
"""

from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch_geometric.datasets import Planetoid

def validate_edges(edges, num_nodes):
    e = np.asarray(edges)
    if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 0:
        raise ValueError('num_nodes phải là số nguyên không âm')
    if e.ndim != 2 or e.shape[0] != 2 or e.dtype.kind not in 'iu':
        raise ValueError('edge_index phải có shape [2,M], dtype integer')
    if e.size and (e.min() < 0 or e.max() >= num_nodes):
        raise ValueError('Node index ngoài miền [0,num_nodes)')
    return e.astype(np.int64, copy=False)

def canonicalize(edges, num_nodes):
    e = validate_edges(edges, num_nodes)
    e = np.sort(e[:, e[0] != e[1]], axis=0)
    return np.unique(e, axis=1)

def bidirectional(edges, num_nodes):
    e = canonicalize(edges, num_nodes)
    return np.concatenate([e, e[::-1]], axis=1)

def with_self_loops(edges, num_nodes):
    e = validate_edges(edges, num_nodes)
    loops = np.tile(np.arange(num_nodes, dtype=np.int64), (2,1))
    return np.unique(np.concatenate([e[:,e[0] != e[1]], loops], axis=1), axis=1)

def neighborhoods(edges, num_nodes):
    e = validate_edges(edges, num_nodes)
    return [e[0,e[1] == node].copy() for node in range(num_nodes)]

def pair_set(edges):
    return set(map(tuple, np.asarray(edges).T.tolist()))

def load_cora(root, config):
    if config['dataset'] != 'Cora' or config['public_node_split'] != 'public':
        raise ValueError('Phạm vi hiện tại chỉ hỗ trợ Cora public')
    return Planetoid(root=str(Path(root)/'data/cora'), name='Cora', split='public')[0]

def make_split(full_pairs, num_nodes, config):
    if not config['undirected'] or config['negative_ratio'] != 1:
        raise ValueError('Protocol hiện tại yêu cầu undirected, negatives 1:1')
    if config['rounding'] != 'floor_val_floor_test_train_remainder':
        raise ValueError('Quy tắc làm tròn chưa hỗ trợ')
    vr, tr = config['val_ratio'], config['test_ratio']
    if not (0 < vr < 1 and 0 < tr < 1 and vr + tr < 1):
        raise ValueError('Tỷ lệ split không hợp lệ')
    pos = canonicalize(full_pairs, num_nodes)
    m = pos.shape[1]
    nv, nt = int(np.floor(m*vr)), int(np.floor(m*tr))
    sizes = [m-nv-nt,nv,nt]
    rng = np.random.default_rng(config['seed'])
    positives = pos[:,rng.permutation(m)]
    forbidden = pair_set(pos)
    if num_nodes*(num_nodes-1)//2-len(forbidden) < m:
        raise ValueError('Không đủ negative pairs cho tỷ lệ 1:1')
    chosen, pool = set(), []
    while len(pool) < m:
        u,v = sorted(rng.integers(0,num_nodes,size=2).tolist())
        pair = (u,v)
        if u != v and pair not in forbidden and pair not in chosen:
            chosen.add(pair)
            pool.append(pair)
    negatives = np.asarray(pool, dtype=np.int64).reshape(-1,2).T
    result, start = {}, 0
    for part,size in zip(['train','val','test'],sizes):
        result[part+'_pos'] = positives[:,start:start+size].copy()
        result[part+'_neg'] = negatives[:,start:start+size].copy()
        start += size
    return result

def audit_split(arrays, full_pairs, num_nodes, adjacency):
    expected = {p+'_'+s for p in ['train','val','test'] for s in ['pos','neg']}
    assert set(arrays) == expected, 'Sai schema sáu mảng split'
    checks, sets = {}, {}
    for key,e in arrays.items():
        e = validate_edges(e, num_nodes)
        assert np.all(e[0] < e[1]), f'{key}: không canonical hoặc self-loop'
        sets[key] = pair_set(e)
        assert len(sets[key]) == e.shape[1], f'{key}: duplicate'
        checks[key+'_shape_unique_canonical'] = {'status':'passed','pairs':e.shape[1]}
    for sign in ['pos','neg']:
        a,b,c = [sets[p+'_'+sign] for p in ['train','val','test']]
        assert not (a&b or a&c or b&c), f'{sign}: giao giữa split'
        checks[sign+'_disjoint'] = {'status':'passed','overlap':0}
    full = pair_set(canonicalize(full_pairs,num_nodes))
    assert set.union(*(sets[p+'_pos'] for p in ['train','val','test'])) == full, 'Positive union sai'
    neg = set.union(*(sets[p+'_neg'] for p in ['train','val','test']))
    assert not neg & full, 'Negative trùng full positive'
    for p in ['train','val','test']:
        assert len(sets[p+'_neg']) == len(sets[p+'_pos']), 'Negative ratio sai'
    adj = validate_edges(adjacency,num_nodes)
    want = bidirectional(arrays['train_pos'],num_nodes)
    assert pair_set(adj) == pair_set(want) and adj.shape == want.shape, 'Adjacency phải đúng train hai chiều; phát hiện leakage'
    held = sets['val_pos'] | sets['test_pos']
    assert not pair_set(canonicalize(adj,num_nodes)) & held, 'Reverse-edge leakage'
    checks['positive_union_negative_exclusion_ratio'] = {'status':'passed','full_pairs':len(full),'negative_pairs':len(neg)}
    checks['train_adjacency_reverse_leakage'] = {'status':'passed','directed_columns':adj.shape[1]}
    return checks
