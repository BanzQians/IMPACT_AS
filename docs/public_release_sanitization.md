# Public Release Sanitization Checklist

Updated: 2026-03-25

Use this checklist before making a repository public.

## 1. Build a sanitized copy

Run:

```bash
python tools/prepare_public_release.py --out-dir ..\\IMPACT_AS_public
```

Publish the generated copy instead of the working research tree.

## 2. Verify sensitive paths are excluded

Default exclusions include:

- `backups/`
- `tools/east/`
- `core/east_online_update.py`
- `core/east_online_adapter_train.py`
- `core/east_label_fusion.py`
- `core/east_shared_assets.py`

## 3. Scan for leakage terms in the export directory

From the sanitized directory, run a term scan for internal mechanism keywords and confirm no private method descriptions remain.

## 4. Re-run app smoke test

In sanitized directory:

```bash
python app.py
```

Confirm startup and core UI tasks still run.

## 5. Confirm docs scope

Public docs should include only:

- setup and usage
- annotation workflow
- data format descriptions
- reproducible utility scripts

Private method design notes, future ablations, and unpublished roadmap details should remain outside the public repository.
