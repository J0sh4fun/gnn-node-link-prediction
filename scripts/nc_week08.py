"""Predeclared GCN validation search and persistent single-use final evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models import GCN
from scripts.node_classification import fit_validation
from utils.data_loader import get_cora_data
from utils.evaluate import evaluate_node_classification
from utils.experiment import environment, mapping_hash, write_json
from utils.paths import isolated_output_root, repository_root, require_cora_cache


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_hash(data):
    return mapping_hash({key: getattr(data, key).cpu().numpy() for key in
                         ('x', 'edge_index', 'train_mask', 'val_mask')})


def tune(data, protocol, output, provenance):
    """All candidates are fixed before fitting; held-out labels are unused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    rows = []
    for index, override in enumerate(protocol['candidates']):
        params = {**protocol['base'], **override}
        config = {**protocol, 'models': {'gcn': params}}
        row = fit_validation(data, 'gcn', config, output / f'candidate_{index:02}', provenance)
        rows.append({'candidate': index, 'params': params, 'best_val_loss': row['best_val_loss'],
                     'validation_score': row['validation_score'], 'best_epoch': row['best_epoch'],
                     'checkpoint_sha256': row['checkpoint_sha256'], 'test_score': None})
        write_json(output / 'search.json', rows)
    winner = min(rows, key=lambda row: (row['best_val_loss'], row['candidate']))
    selection = {**winner, 'model': 'GCN', 'dataset': 'Cora', 'seed': protocol['seed'],
                 'graph_hash': graph_hash(data), 'protocol_sha256': digest(output / 'protocol.json'),
                 'source_hashes': provenance.get('source_hashes', {}),
                 'checkpoint': f"candidate_{winner['candidate']:02}/best_model.pt"}
    write_json(output / 'selection.json', selection)
    return selection


def final_gcn(data, run, output, provenance):
    """Claim the frozen run before accessing test labels; refuse any later attempt.

    A failed attempt leaves its claim for manual audit, never automatic retry.
    Results and predictions are written from the same single forward pass.
    """
    run, output = Path(run), Path(output)
    selection = json.loads((run / 'selection.json').read_text(encoding='utf-8'))
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    checkpoint = run / selection['checkpoint']
    if (digest(checkpoint) != selection['checkpoint_sha256'] or
            digest(run / 'protocol.json') != selection['protocol_sha256'] or
            graph_hash(data) != selection['graph_hash']):
        raise ValueError('Selected checkpoint, protocol or graph has changed')
    if selection['source_hashes'] != provenance.get('source_hashes', {}):
        raise ValueError('Sources changed after selection; audit before final evaluation')
    params = selection['params']
    torch.set_num_threads(protocol['num_threads'])
    model = GCN(data.x.size(1), params['hidden_dim'], 7,
                dropout=params['dropout'], num_layers=params['num_layers'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    mask = data.test_mask
    if (mask.dtype != torch.bool or mask.shape != data.y.shape or not mask.any() or
            (mask & (data.train_mask | data.val_mask)).any()):
        raise ValueError('Invalid or overlapping test mask')
    # Exclusive file creation persists across processes and alternate output dirs.
    with (run / 'final_claim.json').open('x', encoding='utf-8') as handle:
        json.dump({'selection_sha256': digest(run / 'selection.json'),
                   'output': str(output.resolve())}, handle)
    output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite final logits')
        metrics = evaluate_node_classification(logits, data.y, mask)
    np.savez_compressed(output / 'predictions.npz', node_id=mask.nonzero().flatten().numpy(),
                        y_true=data.y[mask].numpy(), y_pred=logits[mask].argmax(-1).numpy(),
                        logits=logits[mask].numpy())
    np.savez_compressed(output / 'masks.npz', **{key: getattr(data, key).numpy() for key in
                                              ('train_mask', 'val_mask', 'test_mask')})
    summary = {**selection, 'test_score': metrics, 'evaluation_split': 'test',
               'test_mask_hash': mapping_hash({'test_mask': mask.numpy()}),
               'predictions_sha256': digest(output / 'predictions.npz'),
               'masks_sha256': digest(output / 'masks.npz'), 'num_test_nodes': int(mask.sum())}
    write_json(output / 'summary.json', summary)
    write_json(output / 'environment.json', provenance)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('tune', 'final-gcn'))
    parser.add_argument('--config', type=Path, default=repository_root() / 'configs/node_classification_week08.json')
    parser.add_argument('--data-root', type=Path, default=repository_root() / 'data')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = isolated_output_root(args.output_dir, repository_root())
    require_cora_cache(args.data_root)
    data = get_cora_data(str(args.data_root), normalize_features=True).cpu()
    provenance = environment(repository_root())
    if args.stage == 'tune':
        protocol = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if protocol['dataset'] != 'Cora':
            parser.error('Cora is required')
        result = tune(data, protocol, output, provenance)
    else:
        if args.run is None:
            parser.error('--run is required for final-gcn')
        result = final_gcn(data, args.run, output, provenance)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
