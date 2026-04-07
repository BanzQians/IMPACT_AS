#!/usr/bin/env python3
"""End-to-end tests for the learning module: LoRA, State Loss, AdaptationSampleBuffer, BackgroundRefinerTrainer."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch import nn

# ── helpers ──────────────────────────────────────────────────────────────────

HIDDEN = 32
NUM_CLASSES = 4
NUM_STATES = 3
INPUT_DIM = 8  # 4 feature + 4 channels
WINDOW_RADIUS = 4
SEQ_LEN = 2 * WINDOW_RADIUS + 1


def _make_model(num_states: int = 0):
    from tools.train_local_refiner import TinyLocalBoundaryModel
    return TinyLocalBoundaryModel(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN,
        dropout=0.0,
        num_states=num_states,
    )


def _random_batch(n: int = 4, with_states: bool = False):
    batch = {
        "x": torch.randn(n, SEQ_LEN, INPUT_DIM),
        "uncertain": torch.rand(n, SEQ_LEN),
        "left": torch.rand(n, SEQ_LEN),
        "right": torch.rand(n, SEQ_LEN),
        "boundary_index": torch.randint(0, SEQ_LEN, (n,)),
        "left_label": torch.randint(0, NUM_CLASSES, (n,)),
        "right_label": torch.randint(0, NUM_CLASSES, (n,)),
    }
    if with_states:
        batch["left_state"] = torch.randint(-1, NUM_STATES, (n,))
        batch["right_state"] = torch.randint(-1, NUM_STATES, (n,))
        batch["state_before"] = batch["left_state"].clone()
        batch["state_after"] = batch["right_state"].clone()
    return batch


passed = 0
failed = 0


def _report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


# ── Test 1: LoRA inject / forward / merge ────────────────────────────────────

def test_lora_inject_merge():
    print("\n=== Test 1: LoRA inject / forward / merge ===")
    from core.local_boundary_refiner import inject_lora, merge_lora, collect_lora_state_dict, load_lora_state_dict, LoRALinear

    model = _make_model()
    x = torch.randn(2, SEQ_LEN, INPUT_DIM)
    unc = torch.rand(2, SEQ_LEN)
    lft = torch.rand(2, SEQ_LEN)
    rgt = torch.rand(2, SEQ_LEN)

    # Baseline output
    model.eval()
    with torch.no_grad():
        out_before = model(x, unc, lft, rgt)
        bl_before = out_before["boundary_logits"].clone()

    # Inject LoRA
    wrapped = inject_lora(model, rank=4, alpha=1.0)
    _report("inject_lora returns wrapped layers", len(wrapped) > 0, f"{len(wrapped)} layers")

    # Check that base weights are frozen
    frozen_ok = all(not p.requires_grad for name, p in model.named_parameters() if "lora" not in name.lower())
    lora_trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    _report("base weights frozen", frozen_ok)
    _report("LoRA params trainable", len(lora_trainable) > 0, f"{len(lora_trainable)} params")

    # Forward still works (B=0 so output should be identical)
    model.eval()
    with torch.no_grad():
        out_after = model(x, unc, lft, rgt)
        bl_after = out_after["boundary_logits"]
    diff = float((bl_before - bl_after).abs().max())
    _report("output unchanged after inject (B=0)", diff < 1e-5, f"max_diff={diff:.2e}")

    # Collect LoRA state dict
    lora_sd = collect_lora_state_dict(model)
    _report("collect_lora_state_dict", len(lora_sd) > 0, f"{len(lora_sd)} keys")

    # Train a tiny bit to change LoRA weights
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    for _ in range(3):
        out = model(x, unc, lft, rgt)
        loss = out["boundary_logits"].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        out_trained = model(x, unc, lft, rgt)
        bl_trained = out_trained["boundary_logits"]
    diff_trained = float((bl_before - bl_trained).abs().max())
    _report("output changed after LoRA training", diff_trained > 1e-3, f"max_diff={diff_trained:.4f}")

    # Merge LoRA
    merged = merge_lora(model)
    _report("merge_lora", len(merged) > 0, f"{len(merged)} layers merged")

    # Check no LoRA attributes remain
    has_lora_after = any(LoRALinear.has_lora(m) for m in model.modules() if isinstance(m, nn.Linear))
    _report("no LoRA attrs after merge", not has_lora_after)

    # Output should be same as post-training
    model.eval()
    with torch.no_grad():
        out_merged = model(x, unc, lft, rgt)
        bl_merged = out_merged["boundary_logits"]
    diff_merge = float((bl_trained - bl_merged).abs().max())
    _report("output preserved after merge", diff_merge < 1e-4, f"max_diff={diff_merge:.2e}")

    # Load LoRA into a fresh model
    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=True)
    inject_lora(model2, rank=4, alpha=1.0)
    lora_sd2 = collect_lora_state_dict(model)  # from merged (should be empty now)
    # Actually test load into injected model with original lora_sd
    model3 = _make_model()
    # Load base weights only
    base_sd = {k: v for k, v in model.state_dict().items()}
    model3.load_state_dict(base_sd, strict=True)
    inject_lora(model3, rank=4, alpha=1.0)
    loaded = load_lora_state_dict(model3, lora_sd)
    _report("load_lora_state_dict", loaded > 0, f"{loaded} layers loaded")


# ── Test 2: State Loss ───────────────────────────────────────────────────────

def test_state_loss():
    print("\n=== Test 2: State Loss computation ===")
    from tools.train_local_refiner import _compute_state_loss

    model = _make_model(num_states=NUM_STATES)
    model.eval()
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()

    batch = _random_batch(8, with_states=True)
    with torch.no_grad():
        out = model(batch["x"], batch["uncertain"], batch["left"], batch["right"])

    _report("model has state heads", "left_state_logits" in out and "right_state_logits" in out)
    _report("left_state_logits shape", out["left_state_logits"].shape == (8, NUM_STATES),
            f"{out['left_state_logits'].shape}")
    _report("right_state_logits shape", out["right_state_logits"].shape == (8, NUM_STATES),
            f"{out['right_state_logits'].shape}")

    loss = _compute_state_loss(out, batch, criterion, device)
    _report("state loss computes", loss.dim() == 0 and not torch.isnan(loss), f"loss={loss.item():.4f}")

    # All -1 targets -> loss should be 0
    batch_no_state = dict(batch)
    batch_no_state["left_state"] = torch.full((8,), -1, dtype=torch.long)
    batch_no_state["right_state"] = torch.full((8,), -1, dtype=torch.long)
    loss_no = _compute_state_loss(out, batch_no_state, criterion, device)
    _report("state loss=0 when all ignored", float(loss_no.item()) == 0.0, f"loss={loss_no.item():.6f}")

    # Model without state heads -> loss 0
    model_no = _make_model(num_states=0)
    model_no.eval()
    with torch.no_grad():
        out_no = model_no(batch["x"], batch["uncertain"], batch["left"], batch["right"])
    _report("no state heads in base model", "left_state_logits" not in out_no)
    loss_base = _compute_state_loss(out_no, batch, criterion, device)
    _report("state loss=0 for base model", float(loss_base.item()) == 0.0)


# ── Test 3: AdaptationSampleBuffer serialization ─────────────────────────────

def test_adaptation_buffer():
    print("\n=== Test 3: AdaptationSampleBuffer ===")
    from core.action_corrections import AdaptationSample, AdaptationSampleBuffer

    buf = AdaptationSampleBuffer(max_size=10)
    _report("empty buffer", len(buf) == 0 and not buf.ready())

    for i in range(7):
        sample = AdaptationSample(
            window_features=np.random.randn(SEQ_LEN, 4).astype(np.float32),
            scribble_channels={
                "uncertain": np.random.rand(SEQ_LEN).astype(np.float32),
                "left": np.random.rand(SEQ_LEN).astype(np.float32),
                "right": np.random.rand(SEQ_LEN).astype(np.float32),
            },
            boundary_energy=np.random.rand(SEQ_LEN).astype(np.float32),
            boundary_frame=50 + i,
            window_start=50 + i - WINDOW_RADIUS,
            window_end=50 + i + WINDOW_RADIUS,
            left_label=f"action_{i % 3}",
            right_label=f"action_{(i + 1) % 3}",
            left_state=f"state_{i % 2}",
            right_state=f"state_{(i + 1) % 2}",
        )
        buf.add(sample)

    _report("buffer has 7 samples", len(buf) == 7)
    _report("buffer ready", buf.ready(min_samples=5))

    # Serialization roundtrip
    payload = buf.to_jsonable()
    _report("to_jsonable is dict", isinstance(payload, dict))

    buf2 = AdaptationSampleBuffer(max_size=10)
    loaded = buf2.load_jsonable(payload)
    _report("load_jsonable count", loaded == 7, f"loaded={loaded}")

    # Check data integrity
    s0 = buf.samples[0]
    s0_rt = buf2.samples[0]
    feat_ok = np.allclose(s0.window_features, s0_rt.window_features, atol=1e-5)
    _report("feature roundtrip", feat_ok)
    _report("label roundtrip", s0.left_label == s0_rt.left_label and s0.right_label == s0_rt.right_label)
    _report("state roundtrip", s0.left_state == s0_rt.left_state and s0.right_state == s0_rt.right_state)

    # Export examples
    examples = buf.export_examples()
    _report("export_examples count", len(examples) == 7)
    _report("export has window_features", "window_features" in examples[0])

    # Max size eviction
    buf3 = AdaptationSampleBuffer(max_size=5)
    for i in range(10):
        sample = AdaptationSample(
            window_features=np.zeros((SEQ_LEN, 4), dtype=np.float32),
            scribble_channels={"uncertain": np.zeros(SEQ_LEN, dtype=np.float32),
                               "left": np.zeros(SEQ_LEN, dtype=np.float32),
                               "right": np.zeros(SEQ_LEN, dtype=np.float32)},
            boundary_energy=np.zeros(SEQ_LEN, dtype=np.float32),
            boundary_frame=i, window_start=0, window_end=SEQ_LEN - 1,
            left_label="a", right_label="b",
        )
        buf3.add(sample)
    _report("max_size eviction", len(buf3) == 5)

    # discard_prefix
    buf.discard_prefix(3)
    _report("discard_prefix", len(buf) == 4)


# ── Test 4: Checkpoint save/load with LoRA + state ──────────────────────────

def test_checkpoint_roundtrip():
    print("\n=== Test 4: Checkpoint save/load with LoRA + state ===")
    from core.local_boundary_refiner import (
        inject_lora, merge_lora, collect_lora_state_dict,
        CheckpointLocalBoundaryRefiner,
    )

    model = _make_model(num_states=NUM_STATES)
    label_to_idx = {f"action_{i}": i for i in range(NUM_CLASSES)}
    idx_to_label = {i: f"action_{i}" for i in range(NUM_CLASSES)}
    state_to_idx = {f"state_{i}": i for i in range(NUM_STATES)}

    # Inject LoRA, train briefly, collect
    inject_lora(model, rank=2, alpha=0.5)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(2, SEQ_LEN, INPUT_DIM)
    unc = torch.rand(2, SEQ_LEN)
    lft = torch.rand(2, SEQ_LEN)
    rgt = torch.rand(2, SEQ_LEN)
    model.train()
    for _ in range(3):
        out = model(x, unc, lft, rgt)
        out["boundary_logits"].sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    lora_sd = collect_lora_state_dict(model)
    merge_lora(model)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        ckpt = {
            "state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN,
            "dropout": 0.0,
            "window_radius": WINDOW_RADIUS,
            "lora_state_dict": lora_sd,
            "lora_rank": 2,
            "lora_alpha": 0.5,
            "state_to_idx": state_to_idx,
            "num_states": NUM_STATES,
        }
        torch.save(ckpt, ckpt_path)
        _report("checkpoint saved", os.path.isfile(ckpt_path))

        # Load via CheckpointLocalBoundaryRefiner
        refiner = CheckpointLocalBoundaryRefiner(ckpt_path, device="cpu")
        _report("refiner loaded", refiner.ready, f"error={refiner.load_error}")

        # Inference
        from core.local_boundary_refiner import LocalRefinerInput
        inp = LocalRefinerInput(
            window_start=0,
            window_end=SEQ_LEN - 1,
            boundary_energy=np.random.rand(SEQ_LEN).astype(np.float32),
            scribble_channels={
                "uncertain": np.random.rand(SEQ_LEN).astype(np.float32),
                "left": np.random.rand(SEQ_LEN).astype(np.float32),
                "right": np.random.rand(SEQ_LEN).astype(np.float32),
            },
            left_candidates=[("action_0", 0.8), ("action_1", 0.2)],
            right_candidates=[("action_2", 0.7), ("action_3", 0.3)],
            window_features=np.random.randn(SEQ_LEN, INPUT_DIM - 4).astype(np.float32),
        )
        result = refiner.refine(inp)
        _report("refine produces output", result.boundary_frame is not None)
        _report("boundary in range", 0 <= result.boundary_frame < SEQ_LEN)
        _report("confidence > 0", result.confidence > 0.0, f"conf={result.confidence:.4f}")
        _report("mode is checkpoint", result.extras.get("mode") == "checkpoint")

        # reload_checkpoint
        reloaded = refiner.reload_checkpoint(ckpt_path)
        _report("reload_checkpoint", reloaded)

    finally:
        os.unlink(ckpt_path)


# ── Test 5: Full training loop (tiny) ───────────────────────────────────────

def test_training_loop():
    print("\n=== Test 5: Training loop with LoRA + State Loss ===")
    from core.local_boundary_refiner import inject_lora, merge_lora, collect_lora_state_dict

    label_to_idx = {f"action_{i}": i for i in range(NUM_CLASSES)}
    state_to_idx = {f"state_{i}": i for i in range(NUM_STATES)}

    # Create base model and save checkpoint
    base_model = _make_model(num_states=0)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        base_path = f.name
    base_ckpt = {
        "state_dict": base_model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": {v: k for k, v in label_to_idx.items()},
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN,
        "dropout": 0.0,
        "window_radius": WINDOW_RADIUS,
    }
    torch.save(base_ckpt, base_path)

    try:
        # Build model with state heads, load base (strict=False)
        model = _make_model(num_states=NUM_STATES)
        model.load_state_dict(base_ckpt["state_dict"], strict=False)
        _report("base checkpoint loaded (strict=False)", True)

        # Inject LoRA
        wrapped = inject_lora(model, rank=2, alpha=1.0)
        _report("LoRA injected for fine-tune", len(wrapped) > 0)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        _report("trainable << total", trainable < total, f"{trainable}/{total}")

        # Mini training
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        criterion = nn.CrossEntropyLoss()

        from tools.train_local_refiner import _compute_state_loss

        # Use a fixed batch for convergence test (same data each step)
        fixed_batch = _random_batch(16, with_states=True)
        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(fixed_batch["x"], fixed_batch["uncertain"], fixed_batch["left"], fixed_batch["right"])
            loss_b = criterion(out["boundary_logits"], fixed_batch["boundary_index"])
            loss_l = criterion(out["left_logits"], fixed_batch["left_label"])
            loss_r = criterion(out["right_logits"], fixed_batch["right_label"])
            loss_s = _compute_state_loss(out, fixed_batch, criterion, torch.device("cpu"))
            loss = loss_b + 0.5 * (loss_l + loss_r) + 0.5 * loss_s
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        _report("training loss decreases", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")

        # Collect LoRA, merge, save
        lora_sd = collect_lora_state_dict(model)
        merge_lora(model)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            adapted_path = f.name
        adapted_ckpt = dict(base_ckpt)
        adapted_ckpt["state_dict"] = model.state_dict()
        adapted_ckpt["lora_state_dict"] = lora_sd
        adapted_ckpt["lora_rank"] = 2
        adapted_ckpt["lora_alpha"] = 1.0
        adapted_ckpt["state_to_idx"] = state_to_idx
        adapted_ckpt["num_states"] = NUM_STATES
        torch.save(adapted_ckpt, adapted_path)
        _report("adapted checkpoint saved", os.path.isfile(adapted_path))

        # Load adapted checkpoint
        loaded_ckpt = torch.load(adapted_path, map_location="cpu")
        _report("has lora_state_dict", "lora_state_dict" in loaded_ckpt)
        _report("has state_to_idx", "state_to_idx" in loaded_ckpt)
        _report("lora_rank=2", loaded_ckpt.get("lora_rank") == 2)

        os.unlink(adapted_path)
    finally:
        os.unlink(base_path)


# ── Test 6: BackgroundRefinerTrainer (functional, no QThread) ────────────────

def test_background_trainer():
    print("\n=== Test 6: BackgroundRefinerTrainer end-to-end ===")
    from core.action_corrections import AdaptationSample, AdaptationSampleBuffer

    label_to_idx = {f"action_{i}": i for i in range(NUM_CLASSES)}
    state_to_idx = {f"state_{i}": i for i in range(NUM_STATES)}

    # Create base checkpoint
    base_model = _make_model(num_states=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "base.pt")
        base_ckpt = {
            "state_dict": base_model.state_dict(),
            "label_to_idx": label_to_idx,
            "idx_to_label": {v: k for k, v in label_to_idx.items()},
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN,
            "dropout": 0.0,
            "window_radius": WINDOW_RADIUS,
        }
        torch.save(base_ckpt, base_path)

        # Build samples
        buf = AdaptationSampleBuffer(max_size=20)
        for i in range(10):
            sample = AdaptationSample(
                window_features=np.random.randn(SEQ_LEN, INPUT_DIM - 4).astype(np.float32),
                scribble_channels={
                    "uncertain": np.random.rand(SEQ_LEN).astype(np.float32),
                    "left": np.random.rand(SEQ_LEN).astype(np.float32),
                    "right": np.random.rand(SEQ_LEN).astype(np.float32),
                },
                boundary_energy=np.random.rand(SEQ_LEN).astype(np.float32),
                boundary_frame=WINDOW_RADIUS,
                window_start=0,
                window_end=SEQ_LEN - 1,
                left_label=f"action_{i % NUM_CLASSES}",
                right_label=f"action_{(i + 1) % NUM_CLASSES}",
                left_state=f"state_{i % NUM_STATES}",
                right_state=f"state_{(i + 1) % NUM_STATES}",
            )
            buf.add(sample)

        samples = buf.export_examples()
        _report("samples exported", len(samples) == 10)

        # Run trainer directly (call _train, skip QThread)
        from core.background_trainer import BackgroundRefinerTrainer

        trainer = BackgroundRefinerTrainer(
            samples=samples,
            base_checkpoint=base_path,
            output_dir=tmpdir,
            label_to_idx=label_to_idx,
            state_to_idx=state_to_idx,
            lora_rank=2,
            lora_alpha=1.0,
            epochs=3,
            lr=1e-3,
            device="cpu",
        )

        # Capture signals
        progress_msgs = []
        finished_path = [None]
        trainer.progress = type("Sig", (), {"emit": lambda self, m: progress_msgs.append(m)})()
        trainer.training_finished = type("Sig", (), {"emit": lambda self, p: finished_path.__setitem__(0, p)})()

        trainer._train()

        _report("progress messages emitted", len(progress_msgs) > 0, f"{len(progress_msgs)} messages")
        _report("finished with path", finished_path[0] is not None and finished_path[0] != "")

        out_path = finished_path[0]
        if out_path and os.path.isfile(out_path):
            _report("output checkpoint exists", True)
            ckpt = torch.load(out_path, map_location="cpu")
            _report("has state_dict", "state_dict" in ckpt)
            _report("has lora_state_dict", "lora_state_dict" in ckpt)
            _report("has state_to_idx", "state_to_idx" in ckpt)
            _report("adaptation_samples recorded", ckpt.get("adaptation_samples", 0) > 0,
                    f"n={ckpt.get('adaptation_samples')}")

            # Verify the checkpoint can be loaded by CheckpointLocalBoundaryRefiner
            from core.local_boundary_refiner import CheckpointLocalBoundaryRefiner
            refiner = CheckpointLocalBoundaryRefiner(out_path, device="cpu")
            _report("adapted refiner loads", refiner.ready, f"error={refiner.load_error}")
        else:
            _report("output checkpoint exists", False, f"path={out_path}")

        bootstrap_trainer = BackgroundRefinerTrainer(
            samples=samples,
            base_checkpoint="",
            output_dir=tmpdir,
            label_to_idx=label_to_idx,
            state_to_idx=state_to_idx,
            epochs=2,
            lr=1e-3,
            hidden_dim=48,
            dropout=0.0,
            device="cpu",
        )
        bootstrap_progress = []
        bootstrap_path = [None]
        bootstrap_trainer.progress = type("Sig", (), {"emit": lambda self, m: bootstrap_progress.append(m)})()
        bootstrap_trainer.training_finished = type("Sig", (), {"emit": lambda self, p: bootstrap_path.__setitem__(0, p)})()
        bootstrap_trainer._train()

        _report("bootstrap trainer emitted progress", len(bootstrap_progress) > 0, f"{len(bootstrap_progress)} messages")
        _report("bootstrap trainer produced path", bool(bootstrap_path[0]), f"path={bootstrap_path[0]}")
        if bootstrap_path[0] and os.path.isfile(bootstrap_path[0]):
            bootstrap_ckpt = torch.load(bootstrap_path[0], map_location="cpu")
            _report("bootstrap checkpoint exists", True)
            _report("bootstrap checkpoint marks cold start", bool(bootstrap_ckpt.get("bootstrap_from_scratch", False)))
            _report("bootstrap checkpoint has label map", "idx_to_label" in bootstrap_ckpt)
            _report("bootstrap checkpoint omits lora weights", "lora_state_dict" not in bootstrap_ckpt)
            from core.local_boundary_refiner import CheckpointLocalBoundaryRefiner
            bootstrap_refiner = CheckpointLocalBoundaryRefiner(bootstrap_path[0], device="cpu")
            _report("bootstrapped refiner loads", bootstrap_refiner.ready, f"error={bootstrap_refiner.load_error}")
        else:
            _report("bootstrap checkpoint exists", False, f"path={bootstrap_path[0]}")


# ── Test 7: Consistency loss / action loss / inline dataset ─────────────────

def test_consistency_action_and_dataset():
    print("\n=== Test 7: Consistency loss / action loss / inline dataset ===")
    from tools.train_local_refiner import (
        ScribbleTrainingDataset,
        _compute_action_loss,
        _compute_boundary_loss,
        _compute_consistency_loss,
        _compute_query_loss,
        _compute_side_loss,
        _compute_struct_loss,
        _normalize_window_arrays,
    )

    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    label_to_idx = {"left": 0, "right": 1}
    state_to_idx = {"0|0": 0, "1|0": 1}

    examples = [
        {
            "window_features": np.random.randn(SEQ_LEN, INPUT_DIM - 4).astype(np.float32),
            "scribble_channels": {
                "uncertain": np.zeros(SEQ_LEN, dtype=np.float32),
                "left": np.concatenate(
                    [np.ones(WINDOW_RADIUS - 1, dtype=np.float32), np.zeros(SEQ_LEN - (WINDOW_RADIUS - 1), dtype=np.float32)]
                ),
                "right": np.concatenate(
                    [np.zeros(WINDOW_RADIUS + 1, dtype=np.float32), np.ones(SEQ_LEN - (WINDOW_RADIUS + 1), dtype=np.float32)]
                ),
            },
            "boundary_energy": np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32),
            "boundary_frame": WINDOW_RADIUS,
            "window_start": 0,
            "window_end": SEQ_LEN - 1,
            "left_label": "left",
            "right_label": "right",
            "state_before": "0|0",
            "state_after": "1|0",
        }
    ]
    dataset = ScribbleTrainingDataset(
        examples,
        features_dir="",
        features_map={},
        window_radius=WINDOW_RADIUS,
        label_to_idx=label_to_idx,
        state_to_idx=state_to_idx,
    )
    sample = dataset[0]
    _report("inline dataset loads", tuple(sample["x"].shape) == (SEQ_LEN, INPUT_DIM))
    _report("action labels generated", tuple(sample["action_labels"].shape) == (SEQ_LEN,))
    _report("state_before present", int(sample["state_before"].item()) >= 0)
    _report("state_after present", int(sample["state_after"].item()) >= 0)

    state_only_dataset = ScribbleTrainingDataset(
        [
            {
                "window_features": np.random.randn(SEQ_LEN, INPUT_DIM - 4).astype(np.float32),
                "scribble_channels": {
                    "uncertain": np.zeros(SEQ_LEN, dtype=np.float32),
                    "left": np.zeros(SEQ_LEN, dtype=np.float32),
                    "right": np.zeros(SEQ_LEN, dtype=np.float32),
                },
                "boundary_energy": np.zeros(SEQ_LEN, dtype=np.float32),
                "boundary_frame": WINDOW_RADIUS,
                "window_start": 0,
                "window_end": SEQ_LEN - 1,
                "left_label": "",
                "right_label": "",
                "state_before": "0|0",
                "state_after": "1|0",
                "boundary_valid": False,
                "side_valid": False,
                "action_valid": False,
                "sample_kind": "state_repair_accept",
            }
        ],
        features_dir="",
        features_map={},
        window_radius=WINDOW_RADIUS,
        label_to_idx=label_to_idx,
        state_to_idx=state_to_idx,
    )
    state_only_sample = state_only_dataset[0]
    _report("state-only sample loads without action labels", tuple(state_only_sample["x"].shape) == (SEQ_LEN, INPUT_DIM))
    _report("state-only sample keeps side loss disabled", bool(state_only_sample["side_valid"].item()) is False)
    _report("state-only sample keeps action labels ignored", int((state_only_sample["action_labels"] >= 0).sum().item()) == 0)

    batch_bad = {
        "left": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32),
        "right": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float32),
        "left_label": torch.tensor([0], dtype=torch.long),
        "right_label": torch.tensor([1], dtype=torch.long),
        "state_before": torch.tensor([0], dtype=torch.long),
        "state_after": torch.tensor([0], dtype=torch.long),
        "action_labels": torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long),
        "query_utility": torch.tensor([1.0], dtype=torch.float32),
    }
    out_bad = {
        "boundary_logits": torch.tensor([[6.0, 5.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "left_logits": torch.tensor([[0.1, 2.2]], dtype=torch.float32),
        "right_logits": torch.tensor([[2.2, 0.1]], dtype=torch.float32),
        "left_state_logits": torch.tensor([[3.0, 0.1]], dtype=torch.float32),
        "right_state_logits": torch.tensor([[2.5, 0.2]], dtype=torch.float32),
        "action_logits": torch.tensor(
            [[[0.1, 2.2], [0.1, 2.2], [2.2, 0.1], [2.2, 0.1], [2.2, 0.1]]],
            dtype=torch.float32,
        ),
        "query_utility_logits": torch.tensor([-4.0], dtype=torch.float32),
    }
    batch_good = dict(batch_bad)
    batch_good["state_after"] = torch.tensor([1], dtype=torch.long)
    out_good = dict(out_bad)
    out_good["boundary_logits"] = torch.tensor([[0.0, 0.2, 5.0, 0.1, 0.0]], dtype=torch.float32)
    out_good["left_logits"] = torch.tensor([[4.0, 0.1]], dtype=torch.float32)
    out_good["right_logits"] = torch.tensor([[0.1, 4.0]], dtype=torch.float32)
    out_good["left_state_logits"] = torch.tensor([[4.0, 0.1]], dtype=torch.float32)
    out_good["right_state_logits"] = torch.tensor([[0.1, 4.0]], dtype=torch.float32)
    out_good["action_logits"] = torch.tensor(
        [[[4.0, 0.1], [4.0, 0.1], [0.1, 4.0], [0.1, 4.0], [0.1, 4.0]]],
        dtype=torch.float32,
    )
    out_good["query_utility_logits"] = torch.tensor([4.0], dtype=torch.float32)

    consistency_bad = _compute_consistency_loss(out_bad, batch_bad, device)
    consistency_good = _compute_consistency_loss(out_good, batch_good, device)
    _report(
        "consistency penalizes anchor/state mismatch",
        float(consistency_bad.item()) > float(consistency_good.item()),
        f"{consistency_bad.item():.4f} > {consistency_good.item():.4f}",
    )

    action_bad = _compute_action_loss(out_bad, batch_bad, device)
    action_good = _compute_action_loss(out_good, batch_good, device)
    _report(
        "action loss improves for correct dense labels",
        float(action_good.item()) < float(action_bad.item()),
        f"{action_good.item():.4f} < {action_bad.item():.4f}",
    )

    struct_bad = _compute_struct_loss(out_bad, batch_bad, device)
    struct_good = _compute_struct_loss(out_good, batch_good, device)
    _report(
        "struct loss improves for consistent dense decode",
        float(struct_good.item()) < float(struct_bad.item()),
        f"{struct_good.item():.4f} < {struct_bad.item():.4f}",
    )

    query_bad = _compute_query_loss(out_bad, batch_bad, device)
    query_good = _compute_query_loss(out_good, batch_good, device)
    _report(
        "query loss improves when utility target is matched",
        float(query_good.item()) < float(query_bad.item()),
        f"{query_good.item():.4f} < {query_bad.item():.4f}",
    )

    interval_batch = {
        "boundary_index": torch.tensor([2], dtype=torch.long),
        "uncertain": torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        "boundary_valid": torch.tensor([True], dtype=torch.bool),
    }
    interval_good = {
        "boundary_logits": torch.tensor([[0.1, 4.0, 4.0, 0.1, 0.1]], dtype=torch.float32),
    }
    interval_bad = {
        "boundary_logits": torch.tensor([[4.0, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32),
    }
    boundary_interval_good = _compute_boundary_loss(interval_good, interval_batch, criterion, device)
    boundary_interval_bad = _compute_boundary_loss(interval_bad, interval_batch, criterion, device)
    _report(
        "boundary loss prefers interval probability mass",
        float(boundary_interval_good.item()) < float(boundary_interval_bad.item()),
        f"{boundary_interval_good.item():.4f} < {boundary_interval_bad.item():.4f}",
    )

    masked_side = {
        "left_label": torch.tensor([0], dtype=torch.long),
        "right_label": torch.tensor([1], dtype=torch.long),
        "side_valid": torch.tensor([False], dtype=torch.bool),
    }
    masked_side_loss = _compute_side_loss(out_bad, masked_side, criterion, device)
    _report(
        "side loss respects side_valid mask",
        abs(float(masked_side_loss.item())) < 1e-8,
        f"loss={masked_side_loss.item():.4f}",
    )

    norm_feat, norm_channels, norm_energy, norm_bi = _normalize_window_arrays(
        np.random.randn(5, INPUT_DIM - 4).astype(np.float32),
        {
            "uncertain": np.ones(5, dtype=np.float32),
            "left": np.zeros(5, dtype=np.float32),
            "right": np.zeros(5, dtype=np.float32),
        },
        0,
        window_radius=WINDOW_RADIUS,
    )
    _report("window normalization pads/crops to fixed length", tuple(norm_feat.shape) == (SEQ_LEN, INPUT_DIM - 4))
    _report("window normalization recenters boundary", int(norm_bi) == WINDOW_RADIUS, f"bi={norm_bi}")
    _report("window normalization keeps channels aligned", tuple(norm_channels["uncertain"].shape) == (SEQ_LEN,))
    _report("window normalization recomputes energy length", tuple(norm_energy.shape) == (SEQ_LEN,))


# ── Test 8: Query utility + escape labels + distill helpers ─────────────────

def test_query_escape_and_distill_helpers():
    print("\n=== Test 8: Query utility / escape labels / distill helpers ===")
    from core.query_planner import (
        QueryCandidate,
        QueryType,
        QueryUtilityModel,
        choose_query,
    )
    from core.structured_decode import ConfirmedWindow, decode_frame_labels_with_constraints
    from tools.distill_global_model import _build_training_data, _compute_sequence_struct_loss
    from ui.action_window import _default_local_refiner_checkpoint_candidates

    history = [
        {
            "kind": "boundary_accept",
            "meta": {
                "query_type": "boundary_scribble",
                "point_type": "boundary_scribble_accept",
                "feedback_start": 10,
                "feedback_end": 18,
                "changed_frame_count": 6,
                "accepted": True,
                "score_terms": {"uncertainty": 0.7, "energy": 0.9, "state_conflict": 0.1},
                "estimated_cost": 0.5,
            },
            "steps": 2,
            "changed": True,
            "committed_at": "2026-04-07T00:00:00Z",
        },
        {
            "kind": "label_accept",
            "meta": {
                "query_type": "label_review",
                "point_type": "label_review_accept",
                "feedback_start": 20,
                "feedback_end": 30,
                "changed_frame_count": 4,
                "accepted": True,
                "score_terms": {"uncertainty": 0.6, "energy": 0.0, "state_conflict": 0.0},
                "estimated_cost": 0.4,
            },
            "steps": 1,
            "changed": True,
            "committed_at": "2026-04-07T00:00:01Z",
        },
        {
            "kind": "state_accept",
            "meta": {
                "query_type": "state_repair",
                "point_type": "state_repair_accept",
                "feedback_start": 32,
                "feedback_end": 40,
                "changed_frame_count": 5,
                "accepted": True,
                "state_conflicts_before": 3,
                "state_conflicts_after": 0,
                "score_terms": {"uncertainty": 0.0, "energy": 0.0, "state_conflict": 0.8},
                "estimated_cost": 0.8,
            },
            "steps": 3,
            "changed": True,
            "committed_at": "2026-04-07T00:00:02Z",
        },
    ]
    model = QueryUtilityModel()
    for item in history:
        model.update_from_summary(item)
    mse = model.fit(lr=0.05, epochs=30)
    cand = QueryCandidate(
        query_id="q1",
        query_type=QueryType.BOUNDARY_SCRIBBLE,
        start_frame=12,
        end_frame=18,
        score_terms={"uncertainty": 0.75, "energy": 0.85, "state_conflict": 0.0},
        estimated_cost=0.5,
    )
    pred_before = model.predict(cand)
    snap = model.snapshot()
    model2 = QueryUtilityModel()
    model2.load_snapshot(snap)
    pred_after = model2.predict(cand)
    _report("query model fit returns mse", mse >= 0.0, f"mse={mse:.4f}")
    _report("query model predicts bounded utility", 0.0 <= pred_before <= 1.0, f"pred={pred_before:.4f}")
    _report("query model snapshot roundtrip", abs(pred_before - pred_after) < 1e-6, f"{pred_before:.4f} ~ {pred_after:.4f}")
    starter_candidates = _default_local_refiner_checkpoint_candidates(ROOT)
    _report(
        "starter checkpoint candidates include default path",
        any(str(path).endswith("configs/models/starter_local_refiner.pt") for path in starter_candidates),
        f"{starter_candidates}",
    )

    escape_candidate = QueryCandidate(
        query_id="escape",
        query_type=QueryType.LABEL_REVIEW,
        start_frame=0,
        end_frame=10,
        score_terms={"uncertainty": 0.9},
        estimated_cost=0.1,
        payload={"current_label": "Unknown"},
    )
    boundary_candidate = QueryCandidate(
        query_id="boundary",
        query_type=QueryType.BOUNDARY_SCRIBBLE,
        start_frame=11,
        end_frame=20,
        score_terms={"uncertainty": 0.4, "energy": 0.5},
        estimated_cost=0.4,
    )
    mixed_escape_boundary = QueryCandidate(
        query_id="boundary_escape",
        query_type=QueryType.BOUNDARY_SCRIBBLE,
        start_frame=21,
        end_frame=30,
        score_terms={"uncertainty": 0.95, "energy": 0.95},
        estimated_cost=0.1,
        payload={"left_label": "Unknown", "right_label": "Assemble", "contains_escape_label": True},
    )
    decision = choose_query([escape_candidate, mixed_escape_boundary, boundary_candidate])
    _report(
        "escape-labelled query ignored by planner",
        decision is not None and decision.candidate.query_id == "boundary",
        f"chosen={None if decision is None else decision.candidate.query_id}",
    )

    decoded, diag = decode_frame_labels_with_constraints(
        frame_labels={},
        windows=[ConfirmedWindow(start_frame=0, end_frame=3, boundary_frame=2, left_label="Unknown", right_label="Assemble")],
        soft_constraints=[],
        label_vocabulary=["Assemble"],
        frame_start=0,
        frame_end=3,
    )
    _report("structured decode keeps escape label legal", decoded.get(0) == "Unknown" and decoded.get(2) == "Assemble", f"{decoded}")
    _report("structured decode has no hard violations", int(diag.get("hard_violations_after", 1)) == 0)

    features = np.zeros((10, 4), dtype=np.float32)
    _, targets, mask = _build_training_data(
        features,
        {
            "view_start": 4,
            "view_end": 6,
            "per_frame_labels": ["Assemble", "Assemble", "Assemble"],
            "confirmed_windows": [],
        },
        {"Assemble": 0},
    )
    _report("distill helper maps view-relative labels onto full sequence", np.all(targets[4:7] == 0))
    _report("distill helper mask matches labels", int(mask.sum()) == 3, f"sum={int(mask.sum())}")

    sparse_features = np.zeros((3, 4), dtype=np.float32)
    _, sparse_targets, sparse_mask = _build_training_data(
        sparse_features,
        {
            "view_start": 4,
            "view_end": 8,
            "per_frame_labels": ["A", "B", "C", "D", "E"],
            "confirmed_windows": [
                {
                    "start_frame": 4,
                    "end_frame": 8,
                    "boundary_frame": 7,
                    "left_label": "A",
                    "right_label": "E",
                }
            ],
        },
        {"A": 0, "C": 1, "E": 2},
        frame_map=np.asarray([4, 6, 8], dtype=np.int64),
    )
    _report(
        "distill helper aligns sparse frame_map to raw frames",
        sparse_targets.tolist() == [0, 1, 2],
        f"targets={sparse_targets.tolist()}",
    )
    _report("distill helper sparse mask matches mapped frames", int(sparse_mask.sum()) == 3, f"sum={int(sparse_mask.sum())}")

    _, masked_targets, masked_mask = _build_training_data(
        np.zeros((3, 4), dtype=np.float32),
        {
            "view_start": 0,
            "view_end": 2,
            "per_frame_labels": ["A", "A", "A"],
            "confirmed_frame_mask": [False, True, False],
            "confirmed_windows": [],
        },
        {"A": 0},
    )
    _report(
        "distill helper respects confirmed-only frame mask",
        masked_targets.tolist() == [-1, 0, -1] and masked_mask.tolist() == [False, True, False],
        f"targets={masked_targets.tolist()} mask={masked_mask.tolist()}",
    )

    distill_good = torch.tensor(
        [
            [4.0, 0.1],
            [4.0, 0.1],
            [0.1, 4.0],
            [0.1, 4.0],
        ],
        dtype=torch.float32,
    )
    distill_bad = torch.tensor(
        [
            [0.1, 4.0],
            [0.1, 4.0],
            [4.0, 0.1],
            [4.0, 0.1],
        ],
        dtype=torch.float32,
    )
    struct_good = _compute_sequence_struct_loss(
        distill_good,
        [
            {
                "start_frame": 0,
                "end_frame": 3,
                "boundary_frame": 2,
                "left_label": "A",
                "right_label": "E",
            }
        ],
        {"A": 0, "E": 1},
        np.arange(4, dtype=np.int64),
        torch.device("cpu"),
    )
    struct_bad = _compute_sequence_struct_loss(
        distill_bad,
        [
            {
                "start_frame": 0,
                "end_frame": 3,
                "boundary_frame": 2,
                "left_label": "A",
                "right_label": "E",
            }
        ],
        {"A": 0, "E": 1},
        np.arange(4, dtype=np.int64),
        torch.device("cpu"),
    )
    _report(
        "distill structured loss prefers consistent windows",
        float(struct_good.item()) < float(struct_bad.item()),
        f"{struct_good.item():.4f} < {struct_bad.item():.4f}",
    )


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_lora_inject_merge()
    test_state_loss()
    test_adaptation_buffer()
    test_checkpoint_roundtrip()
    test_training_loop()
    test_background_trainer()
    test_consistency_action_and_dataset()
    test_query_escape_and_distill_helpers()

    print(f"\n{'='*60}")
    print(f"  TOTAL: {passed + failed} tests | PASSED: {passed} | FAILED: {failed}")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
