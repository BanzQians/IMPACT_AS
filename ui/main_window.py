from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStackedLayout, QSizePolicy, QVBoxLayout, QWidget

from ui.action_window import ActionWindow
from ui.psr_window import PSRWindow
from ui.task_host import TaskHost
from utils.op_logger import OperationLogger
from utils.shortcut_settings import load_logging_policy, save_logging_policy


class MainWindow(QWidget):
    """Standalone host for Action Segmentation and PSR/ASR/ASD."""

    def __init__(self, logger: OperationLogger = None):
        super().__init__()
        self._app_title = "IMPACT AS"
        self.setWindowTitle(self._app_title)
        self.setGeometry(100, 80, 1200, 780)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry() if screen else None
            if geom:
                min_w = min(800, max(640, geom.width() - 80))
                min_h = min(600, max(480, geom.height() - 80))
                self.setMinimumSize(min_w, min_h)
                target_w = min(1200, geom.width() - 60)
                target_h = min(780, geom.height() - 60)
                self.resize(max(min_w, target_w), max(min_h, target_h))
            else:
                self.setMinimumSize(800, 600)
        except Exception:
            self.setMinimumSize(800, 600)
        try:
            self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, False)
        except Exception:
            pass

        self.op_logger = logger or OperationLogger(False)
        initial_logging = load_logging_policy(
            default_ops_enabled=bool(getattr(self.op_logger, "enabled", False)),
            default_validation_summary_enabled=True,
        )
        self._validation_summary_enabled = bool(
            initial_logging.get("validation_summary_enabled", True)
        )
        self.op_logger.enabled = bool(initial_logging.get("ops_csv_enabled", False))
        self.task_items = [
            "Action Segmentation",
            "Assembly State (PSR/ASR/ASD)",
        ]

        root = QVBoxLayout(self)
        self.view_stack = QStackedLayout()
        root.addLayout(self.view_stack)

        self.action_window = ActionWindow(
            logger=self.op_logger,
            on_switch_task=self._on_task_changed,
            tasks=self.task_items,
            on_shortcuts_updated=self._apply_shortcuts_everywhere,
            on_logging_policy_updated=self._set_logging_policy_everywhere,
        )
        self.action_host = TaskHost(self)
        self.action_host.set_body(self.action_window)
        self.view_stack.addWidget(self.action_host)

        self.psr_window = PSRWindow(
            self,
            on_activate=self.action_window._on_psr_asr_asd_activated,
            on_load_components=self.action_window._load_psr_components,
            on_save_components=self.action_window._save_psr_components,
            on_load_rules=self.action_window._load_psr_rules,
            on_edit_rules=self.action_window._open_psr_rules_editor,
            on_export_rules=self.action_window._export_psr_rules,
            on_apply_rules=self.action_window._apply_psr_rules,
            on_learn_rules=self.action_window._psr_learn_rules_from_edits,
            on_batch_convert=self.action_window._psr_batch_convert_dataset,
            on_state_changed=self.action_window._psr_on_state_changed,
            on_reset_segment=self.action_window._psr_reset_selected_segment,
            on_invert_segment=self.action_window._psr_invert_selected_segment,
            on_merge_identical=self.action_window._psr_merge_adjacent_identical_segments,
            on_undo=self.action_window._psr_undo,
            on_redo=self.action_window._psr_redo,
            on_select_from_here=self.action_window._psr_select_from_here,
            on_select_segment=self.action_window._psr_select_segment_only,
            on_split_at_playhead=self.action_window._psr_split_at_playhead,
            on_model_type_changed=self.action_window._psr_on_model_type_changed,
            available_models=self.action_window._psr_model_specs,
            initial_model_type=self.action_window._psr_model_type,
        )
        self.action_window.set_psr_panel(self.psr_window)
        self.view_stack.addWidget(self.psr_window)

        self._current_task = "Action Segmentation"
        self.view_stack.setCurrentWidget(self.action_host)
        self.action_window.set_task(self._current_task)
        self._set_window_title_for_task(self._current_task)
        self._apply_shortcuts_everywhere()
        self._set_logging_policy_everywhere(
            bool(getattr(self.op_logger, "enabled", False)),
            bool(self._validation_summary_enabled),
        )

    def _apply_shortcuts_everywhere(self, bindings=None):
        try:
            self.action_window.apply_shortcut_settings(bindings)
        except Exception:
            pass
        try:
            self.psr_window.apply_shortcut_settings(bindings)
        except Exception:
            pass

    def _set_logging_policy_everywhere(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ):
        self.op_logger.enabled = bool(oplog_enabled)
        self._validation_summary_enabled = bool(validation_summary_enabled)
        validation_comment_prompt_enabled = bool(
            getattr(
                self.action_window,
                "_validation_comment_prompt_enabled",
                load_logging_policy().get("validation_comment_prompt_enabled", True),
            )
        )
        try:
            ok_save, path_or_err = save_logging_policy(
                {
                    "ops_csv_enabled": bool(oplog_enabled),
                    "validation_summary_enabled": bool(validation_summary_enabled),
                    "validation_comment_prompt_enabled": validation_comment_prompt_enabled,
                }
            )
            if not ok_save:
                print(f"[LOG][ERROR] Failed to save logging policy: {path_or_err}")
        except Exception:
            pass
        try:
            self.action_window.set_logging_policy(
                bool(oplog_enabled), bool(validation_summary_enabled)
            )
        except Exception:
            pass

    def _set_window_title_for_task(self, task_name: str) -> None:
        name = (task_name or "").strip()
        if name:
            self.setWindowTitle(f"{name} - {self._app_title}")
        else:
            self.setWindowTitle(self._app_title)

    def _sync_task_selectors(self, text: str) -> None:
        try:
            self.action_window.set_task(text)
        except Exception:
            pass

    def _attach_action_window(self, host) -> None:
        if not host:
            return
        try:
            widget = self.action_window
            was_visible = bool(widget.isVisible())
            try:
                widget.hide()
            except Exception:
                pass
            current_parent = widget.parent()
            if current_parent is not None and current_parent is not host:
                detach = getattr(current_parent, "set_body", None)
                if callable(detach):
                    detach(None)
            try:
                host.setUpdatesEnabled(False)
            except Exception:
                pass
            try:
                host.set_body(widget)
            finally:
                try:
                    host.setUpdatesEnabled(True)
                except Exception:
                    pass
            if was_visible:
                try:
                    widget.show()
                except Exception:
                    pass
        except Exception:
            pass

    def _restore_action_ui(self) -> None:
        self._attach_action_window(self.action_host)
        self.view_stack.setCurrentWidget(self.action_host)
        self.action_window.exit_psr_mode()

    def _on_task_changed(self, text: str) -> None:
        if text == self._current_task:
            self._sync_task_selectors(text)
            return

        self._current_task = text
        self._set_window_title_for_task(text)
        self._sync_task_selectors(text)
        self.action_window.set_review_shortcuts_enabled(True)

        if "psr/asr/asd" in (text or "").lower():
            self._attach_action_window(self.psr_window)
            self.view_stack.setCurrentWidget(self.psr_window)
            self.action_window.enter_psr_mode()
        else:
            self._restore_action_ui()
