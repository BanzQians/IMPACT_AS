# IMPACT_AS

IMPACT_AS (IMPACT-Scribe) is a PyQt5 desktop tool for interactive action segmentation review and assembly-state annotation. It combines machine-generated baselines with a query-driven correction loop that learns from every user interaction.

## Key Features

### Interactive Review
- Multi-view video sessions (up to 5 synchronized cameras)
- Coarse and fine action timelines with direct drag-to-annotate
- ASOT-based pre-labeling for initial baseline segmentation
- Query-driven review via `Suggest Query` with boundary and label suggestions
- Temporal scribble interaction for boundary refinement
- Structured decoding with configurable anchor context

### Three-Layer Correction Learning
The system improves continuously as the user corrects the segmentation:

1. **Immediate** — Label prototype store: online running-mean embeddings per label with cosine-similarity scoring. Updated on every segment confirmation.
2. **Periodic** — LoRA local refiner (`tools/train_local_refiner.py`): lightweight boundary model fine-tuned on accumulated scribble corrections. Supports 8 loss components: boundary, side, consistency, action, query utility, trace conflict, structured, and L2 regularization.
3. **Offline** — Stage C distillation (`tools/distill_global_model.py`): global model fine-tuning using confirmed pseudo-labels and structured decode constraints.

### Query Utility Learning
The query planner includes a `TrainableQueryUtilityModel` (numpy linear regression) that learns to predict which queries are most valuable based on correction outcomes. Trained via `tools/train_query_model.py`.

### Escape Labels
Three reserved labels — **Unknown**, **Other**, **Background** — are always available in the label palette. They are excluded from query scoring and handled specially in structured decode, allowing users to mark uncertain segments without affecting the learning loop.

### Assembly State Review
- PSR / ASR / ASD annotation workflows
- State conflict detection integrated into query scoring

## Quick Start

### Conda environment

Create the dedicated environment in-repo:

```bash
conda env create -f environment.yml -p ./.conda/envs/impact_as
```

Then launch with the bundled wrapper:

```bash
./run.sh
```

`run.sh` uses `./.conda/envs/impact_as`, prepares local runtime directories, and performs a small GUI preflight check before starting the Qt application.

If you prefer to launch Python directly:

```bash
./.conda/envs/impact_as/bin/python app.py
```

Optional operation logging:

```bash
./run.sh --oplog
```

If you only need the minimum pip package list, it remains available in `requirements.txt`.

## Typical Workflow

1. Open a video or session.
2. Generate a baseline with `ASOT Pre-label`, or import an existing segment file.
3. Use `Suggest Query` to step through review targets ranked by utility.
4. Resolve **Label Review** items by accepting, rejecting, or choosing another label.
5. Resolve **Boundary** items by accepting/rejecting the proposed split, or entering `Boundary Scribble` to refine.
6. The system updates label prototypes and accumulates training data after each correction.
7. Periodically run LoRA fine-tuning for improved boundary predictions.
8. Optionally run Stage C distillation for offline global model improvement.
9. Save the updated annotation JSON and optional logs.

## Project Structure

- `app.py` — application entry point
- `run.sh` — local launcher for the in-repo Conda environment
- `ui/` — GUI windows, panels, dialogs, and timeline widgets
- `core/` — query planning, scribble logic, structured decoding, background training, and state helpers
  - `query_planner.py` — query scoring, utility model, escape label filtering
  - `structured_decode.py` — constrained frame-label decoding with anchor context
  - `background_trainer.py` — LoRA-based background refiner with configurable loss weights
- `tools/` — training, feature extraction, baseline inference, and evaluation scripts
  - `train_local_refiner.py` — local boundary model training (8 loss components)
  - `train_query_model.py` — query utility model training from correction history
  - `distill_global_model.py` — Stage C pseudo-label distillation
- `utils/` — constants, escape label helpers, and shared utilities
- `tests/` — unit tests for the learning module (90 tests)
- `docs/` — design documents, user study guides, and workflow notes

## Outputs and Logs

The tool can generate:

- Annotation JSON outputs
- Confirmed pseudo-label exports (per-frame labels + confirmed windows)
- Optional operation logs: `*.ops.log.csv`
- Validation summaries: `*.validation.log.txt`
- LoRA checkpoints and query model snapshots

## License

See `LICENSE`.
