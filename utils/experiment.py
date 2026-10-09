"""Phase 1 runtime helpers extracted from the shared notebook.

New runs also hash Python sources and may use an isolated output root.
Original seed, hash, publication schema and default paths remain unchanged.
"""

from pathlib import Path
import csv, hashlib, importlib.metadata, json, platform, random, subprocess, uuid
from datetime import datetime, timezone
import numpy as np
import torch

def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}. Xem notebooks/README.md để chạy notebook tạo input.")
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def array_hash(value):
    a = np.asarray(value)
    dtype = a.dtype.newbyteorder('<')
    a = np.ascontiguousarray(a.astype(dtype, copy=False))
    header = json.dumps({'shape':list(a.shape), 'dtype':a.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + a.tobytes()).hexdigest()

def mapping_hash(arrays):
    hashes = {k:array_hash(v) for k,v in sorted(arrays.items())}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

def environment(root):
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    sources = {}
    for pattern in ['notebooks/**/*.ipynb', 'configs/*.json',
                    'models/**/*.py', 'layers/**/*.py', 'train/**/*.py',
                    'utils/**/*.py', 'tasks/**/*.py', 'scripts/**/*.py']:
        for path in sorted(Path(root).glob(pattern)):
            if path.suffix == '.ipynb':
                obj = read_json(path)
                content = json.dumps([c['source'] for c in obj['cells']], ensure_ascii=False).encode()
            else:
                content = path.read_bytes()
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    return {'python':platform.python_version(), 'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['torch','torch-geometric','numpy','scipy','scikit-learn','matplotlib','ipython','nbformat','nbclient','ipykernel']},
            'code_revision':git('rev-parse','HEAD'), 'dirty_worktree':bool(git('status','--porcelain')),
            'branch':git('branch','--show-current'), 'source_hashes':sources,
            'cuda_available':torch.cuda.is_available()}

def new_run(root, task, config, *, output_root=None):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + task + '_' + uuid.uuid4().hex[:8]
    path = Path(root if output_root is None else output_root) / 'runs' / run_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / 'config.json', config)
    write_json(path / 'environment.json', environment(root))
    return path

def publish_result(root, run, relative_path, payload, config, week, task, model, dataset, manifest=None):
    env = read_json(run / 'environment.json')
    result = dict(payload, run_id=run.name, config=config, config_path=f'runs/{run.name}/config.json',
                  environment=env, week=week, task=task, model=model, dataset=dataset,
                  split_id=manifest['split_id'] if manifest else None,
                  split_hash=manifest['split_hash'] if manifest else None, test_value=None)
    write_json(run / 'metrics.json', result)
    write_json(Path(root) / relative_path, result)
    fields = 'run_id week owner task model dataset split_id split_hash seed config_path code_revision dirty_worktree validation_metric validation_value test_value result_path'.split()
    index = Path(root) / 'results/experiment_index.csv'
    rows = []
    if index.exists():
        with index.open(encoding='utf-8', newline='') as f:
            rows = [r for r in csv.DictReader(f) if r['task'] != task]
    row = {k:result.get(k) for k in fields}
    row.update(owner='B', seed=config.get('seed'), code_revision=env['code_revision'], dirty_worktree=env['dirty_worktree'], result_path=relative_path)
    rows.append(row)
    with index.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result

def expect_error(function, exceptions=(ValueError, AssertionError)):
    try:
        function()
    except exceptions as exc:
        return {'status':'passed', 'observed':str(exc)}
    raise AssertionError('Input sai đã không bị từ chối')
