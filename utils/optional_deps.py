import importlib
import importlib.util
import os
import sys
from typing import Iterable


class MissingOptionalDependency(RuntimeError):
    def __init__(
        self,
        feature_name: str,
        module_name: str,
        missing_name: str,
        install_hint: str = "",
        custom_message: str = "",
    ):
        self.feature_name = str(feature_name or "").strip() or "This feature"
        self.module_name = str(module_name or "").strip()
        self.missing_name = str(missing_name or "").strip() or self.module_name
        self.install_hint = str(install_hint or "").strip()
        self.custom_message = str(custom_message or "").strip()
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self.custom_message:
            return self.custom_message
        lines = [
            f"{self.feature_name} is unavailable because '{self.missing_name}' is not installed."
        ]
        if self.install_hint:
            lines.append(self.install_hint)
        return "\n".join(lines)


def import_optional_module(
    module_name: str,
    *,
    feature_name: str,
    install_hint: str = "",
):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependency(
            feature_name=feature_name,
            module_name=module_name,
            missing_name=getattr(exc, "name", "") or module_name,
            install_hint=install_hint,
        ) from exc


def import_module_from_path(module_name: str, module_path: str):
    abs_path = os.path.abspath(os.path.expanduser(str(module_path or "").strip()))
    if not abs_path or not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def import_optional_local_module(
    module_name: str,
    *,
    module_path: str,
    feature_name: str,
    install_hint: str = "",
):
    try:
        return import_optional_module(
            module_name,
            feature_name=feature_name,
            install_hint=install_hint,
        )
    except MissingOptionalDependency as exc:
        requested_name = str(module_name or "").strip()
        if exc.missing_name != requested_name:
            raise

        abs_path = os.path.abspath(os.path.expanduser(str(module_path or "").strip()))
        if not abs_path or not os.path.isfile(abs_path):
            raise MissingOptionalDependency(
                feature_name=feature_name,
                module_name=module_name,
                missing_name=requested_name,
                install_hint=install_hint,
                custom_message=(
                    f"{feature_name} is unavailable because the project file "
                    f"'{abs_path or module_path}' is missing from this copy.\n"
                    "Re-copy or re-upload the full project before using this feature."
                ),
            ) from exc

        try:
            return import_module_from_path(requested_name, abs_path)
        except ModuleNotFoundError as inner_exc:
            raise MissingOptionalDependency(
                feature_name=feature_name,
                module_name=module_name,
                missing_name=getattr(inner_exc, "name", "") or requested_name,
                install_hint=install_hint,
            ) from inner_exc


def format_missing_dependency_message(exc: MissingOptionalDependency) -> str:
    return str(exc)


def ensure_optional_modules(
    modules: Iterable[str],
    *,
    feature_name: str,
    install_hint: str = "",
) -> None:
    for module_name in modules or ():
        name = str(module_name or "").strip()
        if not name:
            continue
        import_optional_module(
            name,
            feature_name=feature_name,
            install_hint=install_hint,
        )
