# IMPACT_AS

IMPACT_AS is a desktop annotation toolkit for video understanding tasks.

## Supported workflows

- Action Segmentation
- Transcript workspace support
- Assembly State annotation (PSR/ASR/ASD)
- HandOI / HOI annotation

The GUI is built with PyQt5 and OpenCV.

## Public release scope

This public repository focuses on annotation workflow, data preparation, and utility tooling.
Internal research notes and unpublished experimental details are intentionally omitted.

## Quickstart

1. Create the dedicated Conda environment:

```bash
conda env create -f environment.yml -p ./.conda/envs/impact_as
source ./activate_impact_env.sh
```

If you prefer pip/venv, the minimum package list remains available in `requirements.txt`.

2. Install dependencies into an existing environment only if needed:

```bash
pip install -r requirements.txt
```

3. Launch the application:

```bash
python app.py
```

4. Optional operation logging:

```bash
python app.py --oplog
```

## Project structure

- `app.py`: application entry point
- `ui/`: GUI windows, panels, and timeline widgets
- `core/`: core logic and data processing
- `tools/`: conversion, repair, and evaluation scripts
- `configs/`: configuration files
- `docs/`: public documentation
- `test_data/`: sample data and format checks

## Outputs and logs

The tool can generate annotation outputs and optional logs.

- Annotation JSON outputs
- Operation log (optional): `*.ops.log.csv`
- Validation summary: `*.validation.log.txt`

## Public packaging

To prepare a clean public copy, run:

```bash
python tools/prepare_public_release.py --out-dir ..\\IMPACT_AS_public
```

## License

See `LICENSE`.
