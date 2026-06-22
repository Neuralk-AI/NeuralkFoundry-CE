import pytest
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import json
import joblib

from neuralk_foundry_ce.workflow import WorkFlow, Step, Field


# Define a minimal DummyStep for testing
class DummyStep(Step):
    inputs = [Field('x', '')]
    outputs = [Field('x', ''), Field('y', '')]

    def __init__(self, name, increment):
        super().__init__()
        self.name = name
        self.increment = increment

    def _execute(self, inputs):
        x = inputs["x"] + self.increment
        self.output("x", x)
        self.output("y", x)
        self.log_metric("increment", self.increment)


# TESTS

def test_workflow_runs_and_produces_expected_output():
    wf = WorkFlow(
        steps=[
            DummyStep(name="add2", increment=2),
            DummyStep(name="add3", increment=3)
        ],
        cache_dir=Path(tempfile.mkdtemp())
    )
    result, metrics = wf.run({"x": 5})
    assert result["y"] == 10
    assert metrics["add3"]["increment"] == 3
    shutil.rmtree(wf.cache_dir)

def test_workflow_caching_mechanism():
    cache_dir = Path(tempfile.mkdtemp())
    wf = WorkFlow(steps=[DummyStep(name="add5", increment=5)], cache_dir=cache_dir)
    _ = wf.run({"x": 10})
    # Ensure cache directory exists
    assert (cache_dir / "0_add5").exists()
    assert (cache_dir / "0_add5" / "_scalars.json").exists()
    shutil.rmtree(cache_dir)

def test_check_consistency_raises_on_missing_input():
    wf = WorkFlow(steps=[DummyStep(name="needs_x", increment=1)], cache_dir=Path(tempfile.mkdtemp()))
    with pytest.warns(UserWarning, match="requires unavailable fields"):
        result = wf.check_consistency(init_keys={})
    assert (not result)
    shutil.rmtree(wf.cache_dir)

def test_workflow_metrics_are_collected_per_step():
    wf = WorkFlow(
        steps=[
            DummyStep(name="step1", increment=1),
            DummyStep(name="step2", increment=2),
        ],
        cache_dir=Path(tempfile.mkdtemp())
    )
    _, metrics = wf.run({"x": 0})
    assert metrics["step1"]["increment"] == 1
    assert metrics["step2"]["increment"] == 2
    shutil.rmtree(wf.cache_dir)

def test_cache_files_are_created_correctly():
    cache_dir = Path(tempfile.mkdtemp())
    wf = WorkFlow(
        steps=[DummyStep(name="adder", increment=4)],
        cache_dir=cache_dir
    )
    _ = wf.run({"x": np.arange(10)})
    step_dir = cache_dir / "0_adder"
    assert step_dir.exists()
    assert any(f.name.startswith("y") for f in step_dir.iterdir() if f.name.endswith(".npy") or f.name.endswith(".json"))
    shutil.rmtree(cache_dir)


class _EmitX(Step):
    """A step that goes through the normal cache machinery (uses _execute)."""
    inputs = [Field('seed', '')]
    outputs = [Field('X', '')]

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    def _execute(self, inputs):
        self.output('X', np.full(4, inputs['seed'] + self.value))


def test_workflow_gc_postpones_overlapping_outputs():
    """Two steps both emit 'X'. GC drops step 0's X.npy, records 'postponed'
    in its marker, and a second run replays from cache without re-executing."""
    cache_dir = Path(tempfile.mkdtemp())
    try:
        wf = WorkFlow(steps=[_EmitX('first', 1), _EmitX('second', 10)], cache_dir=cache_dir)
        out, _ = wf.run({'seed': 0})
        assert (out['X'] == 10).all()

        # Caches are nested under the previous step. After GC, step 0 keeps the
        # marker but loses X.npy; step 1 keeps X.npy.
        first_dir = cache_dir / '0_first'
        second_dir = first_dir / '1_second'
        assert (first_dir / '_executed.json').exists()
        assert not (first_dir / 'X.npy').exists()
        assert (second_dir / 'X.npy').exists()
        marker = json.loads((first_dir / '_executed.json').read_text())
        assert marker['postponed'] == {'X': '1_second'}

        # Second run: both steps trust their marker (no re-execute).
        wf2 = WorkFlow(steps=[_EmitX('first', 1), _EmitX('second', 10)], cache_dir=cache_dir)
        out2, _ = wf2.run({'seed': 0})
        assert (out2['X'] == 10).all()
    finally:
        shutil.rmtree(cache_dir)
