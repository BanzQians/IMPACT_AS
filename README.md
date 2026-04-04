# IMPACT_AS

IMPACT_AS is a PyQt5 desktop annotation tool for action segmentation and assembly-state review.

## Current user-facing workflows

- Action Segmentation
- Assembly State annotation and review (`PSR / ASR / ASD`)

The current main window no longer exposes the older Transcript or HOI workspaces. The repository now focuses on the action-segmentation workbench and the assembly-state workflow built on top of it.

## Action Segmentation in this version

The current Action Segmentation workspace supports:

- multi-view video sessions
- coarse and fine action timelines
- manual segmentation and boundary editing
- ASOT-based pre-labeling for an initial baseline segmentation
- query-driven review with `Suggest Query`
- actionable `Label Review` suggestions with `Accept`, `Reject`, or manual label override
- `Boundary Scribble` review, including direct boundary accept/reject or scribble-based refinement
- optional validation logs and operation logs

This repository also includes the current IMPACT-Scribe-oriented interaction modules under `core/` and `ui/`, such as query planning, temporal scribble handling, structured decoding, and local boundary refinement.

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

## Typical action-segmentation flow

1. Open a video or session.
2. Generate a baseline with `ASOT Pre-label`, or import an existing segment file.
3. Use `Suggest Query` to step through lightweight review targets.
4. Resolve `Label Review` items by accepting, rejecting, or choosing another label in the label panel.
5. Resolve `Boundary` items by directly accepting/rejecting the proposed split, or entering `Boundary Scribble` to refine it.
6. Save the updated annotation JSON and optional logs.

## Quick Start guide in the UI

The main window includes a `Quick Start` entry that opens the in-app guide for the current review workflow, including baseline generation, query suggestions, label review, and boundary scribble usage.

## Project structure

- `app.py`: application entry point
- `run.sh`: local launcher for the in-repo Conda environment
- `ui/`: GUI windows, panels, dialogs, and timeline widgets
- `core/`: query planning, scribble logic, structured decoding, and state helpers
- `tools/`: feature extraction, baseline inference, conversion, repair, and evaluation scripts
- `docs/`: project notes and workflow documentation

## Outputs and logs

The tool can generate:

- annotation JSON outputs
- optional operation logs: `*.ops.log.csv`
- validation summaries: `*.validation.log.txt`
- imported or generated baseline segment files for review workflows

## License

See `LICENSE`.
