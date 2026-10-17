"""Compare custom GATLayer and GATConv on identical synthetic sparse graphs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def validate(config):
    for key, minimum in (("threads", 1), ("warmup", 1), ("repeats", 2)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config.get("dtype") not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(config.get("seed")) is not int or not config.get("cases"):
        raise ValueError("An integer seed and nonempty cases are required")
    names = set()
    for case in config["cases"]:
        for key in ("num_nodes", "in_channels", "out_channels", "heads"):
            if type(case[key]) is not int or case[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(case["undirected_pairs"]) is not int or not (
            0 <= case["undirected_pairs"] <= case["num_nodes"] * (case["num_nodes"] - 1) // 2
        ):
            raise ValueError("Invalid number of unique undirected pairs")
        if type(case["concat"]) is not bool or not case["name"] or case["name"] in names:
            raise ValueError("Cases require unique names and boolean concat")
        names.add(case["name"])


def synthetic_inputs(case, seed, dtype, device):
    import torch
    generator = torch.Generator().manual_seed(seed)
    n, count = case["num_nodes"], case["undirected_pairs"]
    pairs = set()
    # Sample O(E) storage; never construct a dense adjacency matrix.
    while len(pairs) < count:
        candidates = torch.randint(n, (max(32, 2 * (count - len(pairs))), 2), generator=generator)
        for u, v in candidates.tolist():
            if u != v:
                pairs.add((min(u, v), max(u, v)))
            if len(pairs) == count:
                break
    edges = torch.tensor(sorted(pairs), dtype=torch.long).reshape(-1, 2).t().contiguous()
    edges = torch.cat((edges, edges.flip(0)), dim=1).to(device)
    x = torch.randn(n, case["in_channels"], generator=generator, dtype=dtype).to(device)
    return x, edges


def matched_layers(case, dtype, device):
    from models import GATLayer
    from torch_geometric.nn import GATConv
    import torch
    kwargs = {key: case[key] for key in ("in_channels", "out_channels", "heads", "concat")}
    kwargs.update(dropout=0.0, negative_slope=0.2, add_self_loops=True, bias=True)
    custom = GATLayer(**kwargs).to(device=device, dtype=dtype)
    reference = GATConv(**kwargs, residual=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        for name, parameter in custom.named_parameters():
            dict(reference.named_parameters())[name].copy_(parameter)
    return {"custom": custom, "pyg": reference}


def check_parity(layers, x, edges):
    import torch
    custom, reference = layers["custom"], layers["pyg"]
    custom.eval()
    reference.eval()
    custom.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    left, right = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
    output, expected = custom(left, edges), reference(right, edges)
    tolerance = 1e-5 if x.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(output)
    (output * upstream).mean().backward()
    (expected * upstream).mean().backward()
    torch.testing.assert_close(left.grad, right.grad, atol=tolerance, rtol=tolerance)
    ref_parameters = dict(reference.named_parameters())
    for name, parameter in custom.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, ref_parameters[name].grad,
                                   atol=tolerance, rtol=tolerance)
    return {"output_and_gradient_match": True, "atol": tolerance, "rtol": tolerance,
            "max_output_absolute_error": (output - expected).abs().max().item()}


def timed_steps(layers, x, edges, mode, warmup, repeats):
    import torch
    training = mode == "forward_backward"
    for layer in layers.values():
        layer.train(training)

    def step(layer):
        if training:
            layer.zero_grad(set_to_none=True)
            layer(x, edges).square().mean().backward()
        else:
            with torch.no_grad():
                layer(x, edges)

    def synchronize():
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)

    for _ in range(warmup):
        for layer in layers.values():
            step(layer)
    synchronize()
    samples = {name: [] for name in layers}
    # Alternate first runner to reduce systematic order/thermal bias.
    for iteration in range(repeats):
        order = ("custom", "pyg") if iteration % 2 == 0 else ("pyg", "custom")
        for name in order:
            synchronize()
            start = time.perf_counter_ns()
            step(layers[name])
            synchronize()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    result = {}
    for name, values in samples.items():
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        result[name] = {"median_ms": statistics.median(values),
                        "p25_ms": quartiles[0], "p75_ms": quartiles[2],
                        "min_ms": min(values), "max_ms": max(values), "samples_ms": values}
    result["custom_over_pyg_median"] = result["custom"]["median_ms"] / result["pyg"]["median_ms"]
    return result


def run_benchmark(config):
    import torch
    import torch_geometric
    validate(config)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent device fallback")
    previous_threads = torch.get_num_threads()
    results = []
    try:
        torch.set_num_threads(config["threads"])
        devices = [torch.cuda.current_device()] if config["device"] == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config["seed"])
            dtype = getattr(torch, config["dtype"])
            for index, case in enumerate(config["cases"]):
                x, edges = synthetic_inputs(case, config["seed"] + index, dtype, config["device"])
                layers = matched_layers(case, dtype, config["device"])
                parity = check_parity(layers, x, edges)
                timings = {mode: timed_steps(layers, x, edges, mode, config["warmup"], config["repeats"])
                           for mode in ("inference", "forward_backward")}
                results.append({"case": case, "directed_edges_before_loops": edges.size(1),
                                "edges_after_loops": edges.size(1) + case["num_nodes"],
                                "parity": parity, "timings": timings})
    finally:
        torch.set_num_threads(previous_threads)
    return {"schema_version": 1, "config": config, "results": results,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "pyg": torch_geometric.__version__, "os": platform.platform(),
                            "processor": platform.processor(),
                            "device_name": torch.cuda.get_device_name() if config["device"] == "cuda" else "CPU"},
            "protocol": {"graph": "synthetic undirected unique pairs; no Cora data/labels loaded",
                         "identical_weights": True, "dropout": 0.0, "self_loops": True,
                         "timer": "perf_counter_ns; CUDA synchronized before/after each sample",
                         "order": "alternate custom/pyg first each repetition",
                         "inference": "eval + no_grad",
                         "forward_backward": "train; zero_grad + forward + mean(square(output)) + backward; no optimizer or input gradients"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gat_benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gat-benchmark/result.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate(config)
    result = run_benchmark(config)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = [ROOT / "models/gat_layer.py", Path(__file__).resolve(), args.config.resolve(), ROOT / "requirements.txt"]
    result["source_hash_format"] = "SHA256 of UTF-8 text normalized to LF newlines"
    result["source_sha256"] = {path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path):
                               hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                               for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["results"]:
        for mode, timing in row["timings"].items():
            print(f"{row['case']['name']} {mode}: custom={timing['custom']['median_ms']:.3f}ms "
                  f"pyg={timing['pyg']['median_ms']:.3f}ms ratio={timing['custom_over_pyg_median']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
