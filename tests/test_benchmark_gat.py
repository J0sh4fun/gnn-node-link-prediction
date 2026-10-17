"""Verify benchmark fairness and sampling, without performance thresholds."""

import copy
import torch
import pytest

from scripts.benchmark_gat import run_benchmark, synthetic_inputs, validate


@pytest.fixture
def config():
    return {"seed": 42, "device": "cpu", "dtype": "float32", "threads": 1,
            "warmup": 1, "repeats": 3,
            "cases": [{"name": "small", "num_nodes": 6, "undirected_pairs": 4,
                       "in_channels": 3, "out_channels": 2, "heads": 2, "concat": True}]}


def test_synthetic_graph_is_reproducible_unique_and_undirected(config):
    first = synthetic_inputs(config["cases"][0], 42, torch.float32, "cpu")
    second = synthetic_inputs(config["cases"][0], 42, torch.float32, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    edges = set(map(tuple, first[1].t().tolist()))
    assert len(edges) == 8
    assert all(u != v and (v, u) in edges for u, v in edges)


@pytest.mark.parametrize("concat", [True, False])
def test_benchmark_matches_outputs_gradients_and_collects_samples(config, concat):
    config["cases"][0]["concat"] = concat
    old_threads = torch.get_num_threads()
    old_rng = torch.get_rng_state().clone()
    result = run_benchmark(config)
    assert torch.get_num_threads() == old_threads
    assert torch.equal(torch.get_rng_state(), old_rng)
    row = result["results"][0]
    assert row["parity"]["output_and_gradient_match"]
    assert row["edges_after_loops"] == 14
    for timing in row["timings"].values():
        for name in ("custom", "pyg"):
            assert len(timing[name]["samples_ms"]) == 3
            assert min(timing[name]["samples_ms"]) > 0
        assert timing["custom_over_pyg_median"] > 0


@pytest.mark.parametrize("change", [
    {"warmup": 0}, {"repeats": 1}, {"threads": 0}, {"dtype": "float16"},
])
def test_invalid_protocol_is_rejected(config, change):
    config.update(change)
    with pytest.raises(ValueError):
        validate(config)


def test_impossible_graph_and_duplicate_case_names_are_rejected(config):
    invalid = copy.deepcopy(config)
    invalid["cases"][0]["undirected_pairs"] = 16
    with pytest.raises(ValueError):
        validate(invalid)
    config["cases"].append(copy.deepcopy(config["cases"][0]))
    with pytest.raises(ValueError):
        validate(config)
