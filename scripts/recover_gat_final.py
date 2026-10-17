"""Recover GAT artifacts with an explicitly authorized repeat evaluation.

This is a protocol exception, never a claim that test was evaluated only once.
The previously selected configuration is fixed; no search is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import GAT
from scripts.nc_week08 import digest, graph_hash
from train.tune_gat import fit_gat, set_seed
from utils.data_loader import get_cora_data
from utils.evaluate import evaluate_node_classification
from utils.experiment import environment, write_json
from utils.paths import repository_root, isolated_output_root, require_cora_cache


def recover(data, original, output, provenance, claim):
    """Rebuild only the selected checkpoint, verify validation, then export once."""
    original, output, claim = Path(original), Path(output), Path(claim)
    report = json.loads(original.read_text(encoding='utf-8'))
    best = min(report['configurations'], key=lambda row: (row['best_val_loss'], row['config_id']))
    if best != report['best_configuration'] or 'final_test_evaluation' not in report:
        raise ValueError('Invalid original selection or absent original final result')
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open('x', encoding='utf-8') as handle:
        json.dump({'authorization': 'User explicitly requested: chạy lại test',
                   'original_report_sha256': digest(original),
                   'output': str(output.resolve()), 'protocol_exception': True}, handle)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'original_report.json', report)
    write_json(output / 'environment.json', provenance)
    config = dict(model='GAT', dataset='Cora', seed=report['seed'], num_layers=2,
                  hidden_dim=best['hidden_dim'], num_heads=best['heads'], output_heads=1,
                  dropout=best['dropout'], attention_dropout=best['dropout'], negative_slope=0.2,
                  learning_rate=best['learning_rate'], weight_decay=best['weight_decay'],
                  max_epochs=200, patience=20, num_threads=1, device='cpu',
                  decoder=None, checkpoint_metric='val_loss', checkpoint_mode='min',
                  split_id='cora_planetoid_public', normalize_features=True,
                  graph_hash=graph_hash(data), original_report_sha256=digest(original),
                  protocol_exception=True, authorization='User: chạy lại test',
                  test_score=None)
    write_json(output / 'config.json', config)
    set_seed(report['seed'])
    model = GAT(in_channels=data.x.size(1), hidden_dim=best['hidden_dim'], num_classes=7,
                heads=best['heads'], out_heads=1, dropout=best['dropout'],
                feature_dropout=best['dropout'], attention_dropout=best['dropout'])
    # Remove all labels outside train/validation during reconstruction.
    labels = data.y.clone()
    labels[~(data.train_mask | data.val_mask)] = -999
    training, state = fit_gat(model, data.x, data.edge_index, labels,
                              data.train_mask, data.val_mask,
                              torch.optim.Adam(model.parameters(), lr=best['learning_rate'],
                                               weight_decay=best['weight_decay']),
                              torch.nn.CrossEntropyLoss(), max_epochs=200, patience=20)
    write_json(output / 'validation.json', training)
    # Do not proceed to test if reconstruction does not reproduce selection.
    if (training['best_epoch'] != best['best_epoch'] or
            abs(training['best_val_loss'] - best['best_val_loss']) > 1e-6 or
            abs(training['val_accuracy'] - best['val_accuracy']) > 1e-12 or
            abs(training['val_macro_f1'] - best['val_macro_f1']) > 1e-12):
        raise ValueError('Reconstructed validation differs from the original; no test evaluated')
    checkpoint = output / 'best_gat_val.pt'
    torch.save(state, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    selection = dict(config, **training, checkpoint_sha256=digest(checkpoint))
    write_json(output / 'selection.json', selection)
    mask = data.test_mask
    if (mask.dtype != torch.bool or mask.shape != data.y.shape or not mask.any() or
            (mask & (data.train_mask | data.val_mask)).any()):
        raise ValueError('Invalid test mask')
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite final logits')
        metrics = evaluate_node_classification(logits, data.y, mask)
    np.savez_compressed(output / 'predictions.npz', node_id=mask.nonzero().flatten().numpy(),
                        y_true=data.y[mask].numpy(), y_pred=logits[mask].argmax(-1).numpy(),
                        logits=logits[mask].numpy())
    np.savez_compressed(output / 'masks.npz', **{key: getattr(data, key).numpy()
                        for key in ('train_mask', 'val_mask', 'test_mask')})
    summary = dict(selection, evaluation_split='test', test_score=metrics,
                   original_test_score=report['final_test_evaluation'],
                   num_test_nodes=int(mask.sum()),
                   evaluation_note='Repeated final at user request; original final already reported by B',
                   predictions_sha256=digest(output / 'predictions.npz'),
                   masks_sha256=digest(output / 'masks.npz'))
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-repeat', action='store_true', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = repository_root()
    output = isolated_output_root(args.output_dir, root)
    require_cora_cache(root / 'data')
    data = get_cora_data(str(root / 'data'), normalize_features=True).cpu()
    result = recover(data, root / 'results/week08/gat_tuning.json', output,
                     environment(root), root / 'runs/week08-gat-final-recovery.claim.json')
    print(json.dumps({'test_score': result['test_score'], 'protocol_exception': True}))


if __name__ == '__main__':
    main()
