"""Native PySide6 moderator console for Optimist Prime.

The console is an operator-facing presentation layer around existing bot
services. It provides a local, read-only view of captured activity, diagnostics,
logs, settings, and controls for an already-running local worker.
File/API work is performed in worker objects on ``QThread`` instances; the GUI
thread only updates widgets and Qt models.

The local worker runs bot cycles independently of the console. This window
never turns on a feature, changes Safe Mode, or performs a Reddit write merely
because it is open.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QIODevice,
    QLockFile,
    QModelIndex,
    QObject,
    QSettings,
    QSignalBlocker,
    QSortFilterProxyModel,
    QThread,
    QTimer,
    QSaveFile,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from bot_logging import configure_logging, current_log_path
from comment_feed import (
    CommentFeedService,
    create_read_only_reddit,
)
from env_file import (
    clear_inherited_runtime_environment,
    load_env_file,
    load_local_env,
    save_env_file,
)
from settings_gui import _merge_save_payload
from settings_registry import SETTING_FIELDS, SettingField
from runtime_activity import activity_path, read_activity_snapshot
from fast_test_mode import FAST_INTERVAL_SECONDS, get_fast_test_mode, start_fast_test_mode, stop_fast_test_mode


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
ICON_PATH = REPO_ROOT / "moderator_bot_icon.ico"
DEFAULT_SUBREDDIT = ""
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_MAX_ITEMS = 100
CONSOLE_RUNTIME_DIR = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "OptimistPrime"
CONSOLE_HEARTBEAT_PATH = CONSOLE_RUNTIME_DIR / "moderator_console_heartbeat.json"
CONSOLE_CONTROL_SERVER_NAME = "OptimistPrimeModeratorConsoleControl"
CONSOLE_LOCK_PATH = CONSOLE_RUNTIME_DIR / "moderator_console.lock"
CONSOLE_ACTIONS = (
    {
        "id": "activity.refresh",
        "label": "Refresh captured activity",
        "description": "Reload items already captured by the normal bot cycle; no Reddit request is made.",
        "side_effect": "read-only",
    },
    {
        "id": "activity.pause",
        "label": "Pause activity updates",
        "description": "Pause local snapshot checks while keeping captured activity visible.",
        "side_effect": "read-only",
    },
    {
        "id": "activity.resume",
        "label": "Resume activity updates",
        "description": "Resume local snapshot checks; no Reddit request is made.",
        "side_effect": "read-only",
    },
    {
        "id": "diagnostics.refresh",
        "label": "Refresh diagnostics",
        "description": "Read the local health and worker summary.",
        "side_effect": "read-only",
    },
    {
        "id": "logs.refresh",
        "label": "Refresh logs",
        "description": "Read the recent redacted operational log tail.",
        "side_effect": "read-only",
    },
    {
        "id": "window.maximize",
        "label": "Maximize console",
        "description": "Maximize the operator console within the monitor work area.",
        "side_effect": "local-ui",
    },
    {
        "id": "window.restore",
        "label": "Restore console",
        "description": "Restore the operator console to its saved window bounds.",
        "side_effect": "local-ui",
    },
    {
        "id": "window.exit",
        "label": "Exit console",
        "description": "Stop the console workers and exit the native operator console.",
        "side_effect": "local-ui",
    },
    {
        "id": "settings.toggle_advanced",
        "label": "Show or hide advanced settings",
        "description": "Reveal or hide uncommon settings without changing their values.",
        "side_effect": "local-ui",
        "parameters": ["visible"],
    },
    {
        "id": "tray.show",
        "label": "Show console",
        "description": "Restore and focus the console from its system-tray state.",
        "side_effect": "local-ui",
    },
    {
        "id": "tray.refresh_activity",
        "label": "Refresh captured activity from tray",
        "description": "Reload the local captured-activity snapshot from the tray menu.",
        "side_effect": "read-only",
    },
    {
        "id": "tray.exit",
        "label": "Exit from tray",
        "description": "Stop the console workers and exit from the tray menu.",
        "side_effect": "local-ui",
    },
    {
        "id": "worker.fast_test",
        "label": "Toggle temporary fast testing",
        "description": "Ask an already-running worker to wait 3 seconds after each cycle for up to 2 minutes; Reddit and AI budgets remain in force.",
        "side_effect": "local-runtime",
        "parameters": ["active"],
    },
)
SECRET_ENV_PREFIXES = (
    "REDDIT_",
    "DEEPSEEK_",
    "OPENROUTER_",
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "GLM_",
    "ZHIPU_",
    "GROQ_",
    "MISTRAL_",
    "TOGETHER_",
    "XAI_",
    "LLM_",
)


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _safe_error(exc: BaseException, *, max_chars: int = 320) -> str:
    """Return a useful UI message without exposing local credentials."""
    message = f"{type(exc).__name__}: {_safe_text(exc).strip()}".strip()
    for key, value in os.environ.items():
        if value and len(value) >= 4 and key.startswith(SECRET_ENV_PREFIXES):
            message = message.replace(value, "[redacted]")
    message = re.sub(
        r"(?i)(client_secret|password|refresh_token|api_key|token)=([^&\s]+)",
        r"\1=[redacted]",
        message,
    )
    if len(message) > max_chars:
        message = message[: max_chars - 1].rstrip() + "…"
    return message or type(exc).__name__


def _format_time(value: Any) -> str:
    raw = _safe_text(value).strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError):
        return _safe_text(value)


def _one_line(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", _safe_text(value)).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _redact_log_text(text: str, *, extra_values: Iterable[Any] = ()) -> str:
    result = text
    values = list(extra_values)
    for key, value in os.environ.items():
        if value and len(value) >= 4 and key.startswith(SECRET_ENV_PREFIXES):
            values.append(value)
    for value in values:
        value = _safe_text(value)
        if len(value) >= 4:
            result = result.replace(value, "[redacted]")
    return result


def default_settings_values() -> dict[str, str]:
    """Return display-safe defaults without reading the operator’s .env."""
    values: dict[str, str] = {}
    for field in SETTING_FIELDS:
        if field.field_type == "bool":
            values[field.key] = "true" if field.default.strip().lower() in {"1", "true", "yes", "on"} else "false"
        else:
            values[field.key] = field.default
    return values


def load_settings_for_ui(env_path: Path = ENV_PATH) -> tuple[dict[str, str], dict[str, bool]]:
    """Read editable settings for the UI without returning password values."""
    stored = load_env_file(env_path)
    values = default_settings_values()
    configured: dict[str, bool] = {}
    for field in SETTING_FIELDS:
        raw = stored.get(field.key, field.default)
        if field.field_type == "password":
            values[field.key] = ""
            configured[field.key] = bool(stored.get(field.key, "").strip())
        elif field.field_type == "bool":
            values[field.key] = "true" if raw.strip().lower() in {"1", "true", "yes", "on"} else "false"
        else:
            values[field.key] = raw
    return values, configured


def persist_settings_for_ui(payload: dict[str, str], env_path: Path = ENV_PATH) -> int:
    """Persist settings using the shared password-preserving merge rules."""
    merged = _merge_save_payload(payload, env_path=env_path)
    save_env_file(env_path, merged)
    return len(payload)


class CommentFeedModel(QAbstractTableModel):
    """GUI-thread model for bounded activity snapshots from bot cycles."""

    HEADERS = ("Time", "Author", "Post", "Score", "State", "Error", "Comment")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return dict(row)
        if role == Qt.ItemDataRole.ToolTipRole:
            return _safe_text(row.get("body")) or "[empty comment]"
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        column = index.column()
        if column == 0:
            return _format_time(row.get("createdAt") or row.get("createdUtc"))
        if column == 1:
            author = _safe_text(row.get("author"), "[unknown]")
            return author if author.startswith("[") else f"u/{author}"
        if column == 2:
            return _one_line(row.get("linkTitle"), 90) or "[comment]"
        if column == 3:
            score = row.get("score")
            return "" if score is None else _safe_text(score)
        if column == 4:
            return _safe_text(row.get("processingState") or row.get("state"), "unprocessed")
        if column == 5:
            return _one_line(row.get("error") or row.get("lastError") or row.get("errorMessage"), 120)
        if column == 6:
            return _one_line(row.get("body"))
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    @Slot(dict)
    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Replace the bounded view atomically on the GUI thread."""
        rows = snapshot.get("comments") if isinstance(snapshot, dict) else []
        cleaned = [dict(row) for row in rows or [] if isinstance(row, dict)]
        self.beginResetModel()
        self._rows = cleaned
        self.endResetModel()

    def comment_for_index(self, index: QModelIndex) -> dict[str, Any] | None:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        return dict(self._rows[index.row()])


class CommentFilterProxyModel(QSortFilterProxyModel):
    """Filter comments by the cheap local fields already in each snapshot."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._query = ""
        self._content_type = "All content"
        self._time_range = "Any time"
        self._processing_state = "All states"
        self._error_filter = "All items"
        self.setDynamicSortFilter(True)

    @Slot(str)
    def set_query(self, value: str) -> None:
        self._query = value.strip().casefold()
        self._invalidate_rows_filter()

    @Slot(str)
    def set_content_type(self, value: str) -> None:
        self._content_type = value
        self._invalidate_rows_filter()

    @Slot(str)
    def set_time_range(self, value: str) -> None:
        self._time_range = value
        self._invalidate_rows_filter()

    @Slot(str)
    def set_processing_state(self, value: str) -> None:
        self._processing_state = value
        self._invalidate_rows_filter()

    @Slot(str)
    def set_error_filter(self, value: str) -> None:
        self._error_filter = value
        self._invalidate_rows_filter()

    def _invalidate_rows_filter(self) -> None:
        # Qt 6.10's scoped filter-change API avoids the deprecated broad
        # invalidateFilter() call while keeping all model work on the GUI
        # thread.
        try:
            self.beginFilterChange()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        except AttributeError:  # pragma: no cover - compatibility fallback
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        source = self.sourceModel()
        if source is None:
            return False
        index = source.index(source_row, 0, source_parent)
        row = source.data(index, Qt.ItemDataRole.UserRole) or {}
        if self._content_type != "All content":
            expected = self._content_type.casefold()
            actual = _safe_text(row.get("contentType"), "comment").casefold()
            if expected not in actual:
                return False
        if self._time_range != "Any time":
            seconds = {"Last hour": 3600, "Last 6 hours": 21_600, "Last 24 hours": 86_400}.get(self._time_range)
            if seconds is not None:
                try:
                    created = float(row.get("createdUtc") or 0)
                except (TypeError, ValueError):
                    created = 0
                if created <= 0 or created < time.time() - seconds:
                    return False
        if self._processing_state != "All states":
            actual_state = _safe_text(row.get("processingState") or row.get("state"), "")
            if not actual_state and (row.get("error") or row.get("lastError") or row.get("errorMessage")):
                actual_state = "error"
            actual_state = actual_state or "unprocessed"
            if actual_state.casefold() != self._processing_state.casefold():
                return False
        has_error = bool(row.get("error") or row.get("lastError") or row.get("errorMessage"))
        if self._error_filter == "Errors only" and not has_error:
            return False
        if self._error_filter == "No errors" and has_error:
            return False
        if not self._query:
            return True
        haystack = " ".join(
            _safe_text(row.get(key))
            for key in (
                "author",
                "body",
                "linkTitle",
                "permalink",
                "subreddit",
                "id",
                "processingState",
                "state",
                "error",
                "lastError",
                "errorMessage",
            )
        ).casefold()
        return self._query in haystack

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        left_row = self.sourceModel().data(left, Qt.ItemDataRole.UserRole) or {}
        right_row = self.sourceModel().data(right, Qt.ItemDataRole.UserRole) or {}
        column = left.column()
        if column == 0:
            left_value = float(left_row.get("createdUtc") or 0)
            right_value = float(right_row.get("createdUtc") or 0)
        elif column == 3:
            left_value = float(left_row.get("score") or 0)
            right_value = float(right_row.get("score") or 0)
        elif column == 4:
            left_value = _safe_text(left_row.get("processingState") or left_row.get("state"), "unprocessed").casefold()
            right_value = _safe_text(right_row.get("processingState") or right_row.get("state"), "unprocessed").casefold()
        elif column == 5:
            left_value = _safe_text(left_row.get("error") or left_row.get("lastError") or left_row.get("errorMessage")).casefold()
            right_value = _safe_text(right_row.get("error") or right_row.get("lastError") or right_row.get("errorMessage")).casefold()
        else:
            fields = ("createdAt", "author", "linkTitle", "score", "processingState", "error", "body")
            field = fields[column]
            left_value = _safe_text(left_row.get(field)).casefold()
            right_value = _safe_text(right_row.get(field)).casefold()
        return left_value < right_value


class ActionTableModel(QAbstractTableModel):
    """Small structured model for operator actions and their results."""

    HEADERS = ("Time", "Action", "Result")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, str]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        return (row.get("time", ""), row.get("action", ""), row.get("result", ""))[index.column()]

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable if index.isValid() else Qt.ItemFlag.NoItemFlags

    @Slot(str, str)
    def append_action(self, action: str, result: str) -> None:
        self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
        self._rows.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action, "result": _one_line(result, 300)})
        self.endInsertRows()
        if len(self._rows) > 200:
            self.beginRemoveRows(QModelIndex(), 0, len(self._rows) - 201)
            del self._rows[:-200]
            self.endRemoveRows()


class RetryTableModel(QAbstractTableModel):
    """Structured view of retry/error observations surfaced by the console."""

    HEADERS = ("Time", "Source", "Status", "Message")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, str]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self._rows[index.row()]
        return (
            row.get("time", ""),
            row.get("source", ""),
            row.get("status", ""),
            row.get("message", ""),
        )[index.column()]

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable if index.isValid() else Qt.ItemFlag.NoItemFlags

    @Slot(str, str, str)
    def append_retry(self, source: str, status: str, message: str) -> None:
        self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
        self._rows.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": _one_line(source, 80),
            "status": _one_line(status, 60),
            "message": _one_line(message, 300),
        })
        self.endInsertRows()
        if len(self._rows) > 200:
            self.beginRemoveRows(QModelIndex(), 0, len(self._rows) - 201)
            del self._rows[:-200]
            self.endRemoveRows()


class FeedWorker(QObject):
    """Owns authentication and blocking feed reads inside a Qt worker thread."""

    snapshot_ready = Signal(dict)
    progress = Signal(str)
    failure = Signal(str)
    identity_ready = Signal(str)
    state_changed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        service_factory: Callable[[], CommentFeedService],
        poll_seconds: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service_factory = service_factory
        self._poll_seconds = max(1.0, float(poll_seconds))
        self._service: CommentFeedService | None = None
        self._timer: QTimer | None = None
        self._auth_retry_timer: QTimer | None = None
        self._refreshing = False
        self._stopping = False
        self._paused = False

    @Slot()
    def start(self) -> None:
        if self._stopping:
            return
        self.progress.emit("Connecting to Reddit in the background…")
        self.state_changed.emit("starting")
        self._authenticate()

    def _authenticate(self) -> None:
        if self._stopping or self._service is not None:
            return
        try:
            self._service = self._service_factory()
            account = _safe_text(getattr(self._service, "account_name", ""))
            self.identity_ready.emit(account or "authenticated account not reported")
            self.snapshot_ready.emit(self._service.snapshot())
            self._timer = QTimer(self)
            self._timer.setInterval(max(1000, int(self._poll_seconds * 1000)))
            self._timer.timeout.connect(self.refresh_now)
            self._timer.start()
            self.refresh_now()
        except Exception as exc:
            message = _safe_error(exc)
            self.state_changed.emit("error")
            self.failure.emit(message)
            self.progress.emit("Reddit connection failed; the console will retry automatically.")
            if self._auth_retry_timer is None and not self._stopping:
                self._auth_retry_timer = QTimer(self)
                self._auth_retry_timer.setInterval(60_000)
                self._auth_retry_timer.timeout.connect(self._authenticate)
                self._auth_retry_timer.start()

    @Slot()
    def refresh_now(self) -> None:
        if self._stopping or self._paused or self._service is None or self._refreshing:
            return
        self._refreshing = True
        self.progress.emit("Reading the newest comments from Reddit…")
        try:
            ok = self._service.refresh_now()
            snapshot = self._service.snapshot()
            self.snapshot_ready.emit(snapshot)
            status = snapshot.get("status", {}) if isinstance(snapshot, dict) else {}
            if ok:
                self.state_changed.emit("connected")
                self.progress.emit("Feed updated; Reddit writes are disabled for this view.")
            else:
                message = _safe_text(status.get("lastError"), "Reddit read failed; cached comments retained.")
                self.state_changed.emit("error")
                self.failure.emit(message)
        except Exception as exc:
            message = _safe_error(exc)
            self.state_changed.emit("error")
            self.failure.emit(message)
        finally:
            self._refreshing = False

    @Slot()
    def pause(self) -> None:
        if self._service is None or self._stopping:
            return
        self._paused = True
        if self._timer is not None:
            self._timer.stop()
        self._service.pause()
        self.snapshot_ready.emit(self._service.snapshot())
        self.state_changed.emit("paused")
        self.progress.emit("Automatic feed updates paused; cached comments remain visible.")

    @Slot()
    def resume(self) -> None:
        if self._service is None or self._stopping:
            return
        self._paused = False
        self._service.resume()
        if self._timer is not None:
            self._timer.start()
        self.snapshot_ready.emit(self._service.snapshot())
        self.refresh_now()

    @Slot()
    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._timer is not None:
            self._timer.stop()
        if self._auth_retry_timer is not None:
            self._auth_retry_timer.stop()
        if self._service is not None:
            self._service.stop()
            self.snapshot_ready.emit(self._service.snapshot())
        self.state_changed.emit("stopped")
        self.finished.emit()


class FeedController(QObject):
    """Own the QThread lifecycle and expose queued operations to the window."""

    snapshot_ready = Signal(dict)
    progress = Signal(str)
    failure = Signal(str)
    identity_ready = Signal(str)
    state_changed = Signal(str)
    finished = Signal()
    refresh_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(
        self,
        service_factory: Callable[[], CommentFeedService],
        poll_seconds: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._service_factory = service_factory
        self._poll_seconds = poll_seconds
        self._thread: QThread | None = None
        self._worker: FeedWorker | None = None
        self._stopped = False

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._stopped = False
        self._thread = QThread(self)
        self._thread.setObjectName("optimist-prime-feed-thread")
        self._worker = FeedWorker(self._service_factory, self._poll_seconds)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.snapshot_ready.connect(self.snapshot_ready)
        self._worker.progress.connect(self.progress)
        self._worker.failure.connect(self.failure)
        self._worker.identity_ready.connect(self.identity_ready)
        self._worker.state_changed.connect(self.state_changed)
        self._worker.finished.connect(self.finished)
        # QThread is owned by the GUI thread.  A queued quit signal would wait
        # for that thread's event loop while shutdown is waiting on this
        # worker, so invoke the thread-safe quit directly from the worker.
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self.refresh_requested.connect(self._worker.refresh_now)
        self.pause_requested.connect(self._worker.pause)
        self.resume_requested.connect(self._worker.resume)
        self.stop_requested.connect(self._worker.stop)
        self._thread.start()

    def stop(self, wait_ms: int = 75_000) -> bool:
        if self._thread is None or not self._thread.isRunning():
            return True
        self._stopped = True
        self.stop_requested.emit()
        self._thread.requestInterruption()
        return self._thread.wait(wait_ms)

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())


class LocalActivityWorker(QObject):
    """Read the bot's local captured-activity snapshot without using Reddit."""

    snapshot_ready = Signal(dict)
    progress = Signal(str)
    failure = Signal(str)
    identity_ready = Signal(str)
    state_changed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        poll_seconds: float,
        subreddit: str,
        env_path: Path = ENV_PATH,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._poll_seconds = max(1.0, float(poll_seconds))
        self._subreddit = subreddit
        self._env_path = Path(env_path)
        self._timer: QTimer | None = None
        self._stopping = False
        self._paused = False
        self._refreshing = False

    def _snapshot(self) -> dict[str, Any]:
        # The console deliberately does not load the whole operator environment
        # into its process.  Resolve only the non-secret activity path so the
        # viewer follows the same outside-the-repository location as the
        # scheduled worker.
        configured_path = os.environ.get("BOT_RUNTIME_ACTIVITY_FILE", "").strip()
        if not configured_path:
            configured_path = load_env_file(self._env_path).get("BOT_RUNTIME_ACTIVITY_FILE", "").strip()
        snapshot = read_activity_snapshot(activity_path(configured_path or None))
        items = [dict(item) for item in snapshot.get("items", []) if isinstance(item, dict)]
        status = {
            "state": "paused" if self._paused else "connected",
            "message": "Showing items captured by the bot; no Reddit request was made.",
            "subreddit": snapshot.get("subreddit", self._subreddit),
            "accountName": "local captured activity",
            "cachedCount": len(items),
            "updatedAt": snapshot.get("updatedAt"),
            "source": "bot-cycle-snapshot",
        }
        # The existing Qt model consumes the comments key.  Keep items as the
        # canonical persisted name while presenting posts and comments through
        # the same read-only model.
        return {**snapshot, "items": items, "comments": items, "status": status}

    @Slot()
    def start(self) -> None:
        if self._stopping:
            return
        self.state_changed.emit("starting")
        self.identity_ready.emit("local captured activity")
        self.refresh_now()
        self._timer = QTimer(self)
        self._timer.setInterval(max(1000, int(self._poll_seconds * 1000)))
        self._timer.timeout.connect(self.refresh_now)
        self._timer.start()

    @Slot()
    def refresh_now(self) -> None:
        if self._stopping or self._paused or self._refreshing:
            return
        self._refreshing = True
        try:
            snapshot = self._snapshot()
            self.snapshot_ready.emit(snapshot)
            self.state_changed.emit("connected")
            self.progress.emit("Local activity updated; no Reddit request was made.")
        except Exception as exc:
            message = _safe_error(exc)
            self.state_changed.emit("error")
            self.failure.emit(message)
        finally:
            self._refreshing = False

    @Slot()
    def pause(self) -> None:
        if self._stopping:
            return
        self._paused = True
        if self._timer is not None:
            self._timer.stop()
        self.snapshot_ready.emit(self._snapshot())
        self.state_changed.emit("paused")
        self.progress.emit("Local activity updates paused; captured items remain visible.")

    @Slot()
    def resume(self) -> None:
        if self._stopping:
            return
        self._paused = False
        if self._timer is not None:
            self._timer.start()
        self.refresh_now()

    @Slot()
    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        if self._timer is not None:
            self._timer.stop()
        try:
            self.snapshot_ready.emit(self._snapshot())
        except Exception:
            pass
        self.state_changed.emit("stopped")
        self.finished.emit()


class LocalActivityController(QObject):
    """Own the thread that watches the local activity snapshot."""

    snapshot_ready = Signal(dict)
    progress = Signal(str)
    failure = Signal(str)
    identity_ready = Signal(str)
    state_changed = Signal(str)
    finished = Signal()
    refresh_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()

    def __init__(
        self,
        poll_seconds: float,
        subreddit: str,
        parent: QObject | None = None,
        env_path: Path = ENV_PATH,
    ) -> None:
        super().__init__(parent)
        self._poll_seconds = poll_seconds
        self._subreddit = subreddit
        self._env_path = Path(env_path)
        self._thread: QThread | None = None
        self._worker: LocalActivityWorker | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._thread.setObjectName("optimist-prime-local-activity-thread")
        self._worker = LocalActivityWorker(self._poll_seconds, self._subreddit, self._env_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.snapshot_ready.connect(self.snapshot_ready)
        self._worker.progress.connect(self.progress)
        self._worker.failure.connect(self.failure)
        self._worker.identity_ready.connect(self.identity_ready)
        self._worker.state_changed.connect(self.state_changed)
        self._worker.finished.connect(self.finished)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self.refresh_requested.connect(self._worker.refresh_now)
        self.pause_requested.connect(self._worker.pause)
        self.resume_requested.connect(self._worker.resume)
        self.stop_requested.connect(self._worker.stop)
        self._thread.start()

    def stop(self, wait_ms: int = 75_000) -> bool:
        if self._thread is None or not self._thread.isRunning():
            return True
        self.stop_requested.emit()
        self._thread.requestInterruption()
        return self._thread.wait(wait_ms)

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())


class TaskWorker(QObject):
    """Run one diagnostic, log, or worker operation away from the GUI thread."""

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._operation = operation

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self._operation())
        except Exception as exc:
            self.failed.emit(_safe_error(exc))
        finally:
            self.finished.emit()


class TaskController(QObject):
    """One-shot QThread wrapper with no overlapping operation."""

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._operation = operation
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None

    def start(self) -> bool:
        if self._thread is not None and self._thread.isRunning():
            return False
        self._thread = QThread(self)
        self._worker = TaskWorker(self._operation)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self.succeeded)
        self._worker.failed.connect(self.failed)
        self._worker.finished.connect(self.finished)
        self._worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        return True

    def wait(self, wait_ms: int = 75_000) -> bool:
        return True if self._thread is None else self._thread.wait(wait_ms)

    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.isRunning())


def read_log_tail(max_lines: int = 180, env_path: Path = ENV_PATH) -> str:
    # Resolve the configured log directory from the local .env in this worker
    # instead of relying on whichever process environment happened to start
    # the GUI.
    stored = load_env_file(env_path)
    configured = stored.get("BOT_LOG_DIR", "").strip()
    secret_values = [
        value
        for key, value in stored.items()
        if key.startswith(SECRET_ENV_PREFIXES) and value
    ]
    if configured:
        log_directory = Path(configured)
        if not log_directory.is_absolute():
            log_directory = REPO_ROOT / log_directory
        path = log_directory / "optimist_prime.jsonl"
    else:
        path = current_log_path()
    if not path.exists():
        return f"No structured log has been created yet.\nExpected path: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return _redact_log_text("\n".join(lines[-max_lines:]), extra_values=secret_values) or "The structured log is empty."


def read_owner_diagnostics(env_path: Path = ENV_PATH) -> dict[str, Any]:
    """Read operator diagnostics in an isolated process, never the GUI process."""
    command = [
        sys.executable,
        str(REPO_ROOT / "diagnostics.py"),
        "--pretty",
        "--environment-path",
        str(env_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = completed.stdout.strip()
    if not output:
        detail = (completed.stderr or "diagnostics returned no output").strip()
        raise RuntimeError(detail[:500])
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("diagnostics returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("diagnostics returned an invalid result")
    return payload


class SettingsPanel(QWidget):
    """Native settings editor that reuses the existing secret-safe .env merge."""

    persisted = Signal(str)
    persist_requested = Signal(object, str)

    def __init__(self, env_path: Path = ENV_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._env_path = Path(env_path)
        self._editors: dict[str, QWidget] = {}
        self._fields: dict[str, SettingField] = {field.key: field for field in SETTING_FIELDS}
        self._advanced_visible = False
        self._pending: dict[str, str] = {}
        self._field_rows: dict[str, list[QWidget]] = {}
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(350)
        self._save_timer.timeout.connect(self._persist_pending)
        # Build from safe defaults first.  The actual .env read is requested
        # by ModeratorConsole through a QThread worker, so startup file I/O
        # cannot stall the GUI thread.
        self._values = default_settings_values()
        self._configured: dict[str, bool] = {}
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        intro = QLabel(
            "Changes save automatically to the local .env file. "
            "Password fields are never displayed; leaving one blank preserves the stored value. "
            "Settings affect the next bot process and do not make this window perform Reddit writes."
        )
        intro.setWordWrap(True)
        intro.setObjectName("settingsIntro")
        root.addWidget(intro)

        controls = QHBoxLayout()
        self._advanced_check = QCheckBox("Show advanced settings")
        self._advanced_check.setAccessibleName("Show advanced settings")
        self._advanced_check.setToolTip("Reveal uncommon provider, moderation, flair, and runtime-limit settings.")
        self._advanced_check.toggled.connect(self._set_advanced_visible)
        controls.addWidget(self._advanced_check)
        controls.addStretch(1)
        root.addLayout(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        contents = QWidget()
        form_root = QVBoxLayout(contents)
        form_root.setContentsMargins(8, 8, 8, 8)
        sections: dict[str, QFormLayout] = {}
        groups: dict[str, QGroupBox] = {}
        for field in SETTING_FIELDS:
            if field.section not in sections:
                group = QGroupBox(field.section)
                group.setObjectName("settingsGroup")
                layout = QFormLayout(group)
                layout.setContentsMargins(16, 24, 16, 16)
                layout.setHorizontalSpacing(18)
                layout.setVerticalSpacing(10)
                form_root.addWidget(group)
                groups[field.section] = group
                sections[field.section] = layout
            editor = self._make_editor(field)
            self._editors[field.key] = editor
            label = QLabel(field.label)
            label.setWordWrap(True)
            label.setToolTip(field.help_text)
            sections[field.section].addRow(label, editor)
            row_widgets: list[QWidget] = [label, editor]
            if field.help_text:
                help_label = QLabel(field.help_text)
                help_label.setWordWrap(True)
                help_label.setObjectName("settingHelp")
                sections[field.section].addRow("", help_label)
                row_widgets.append(help_label)
            self._field_rows[field.key] = row_widgets
        form_root.addStretch(1)
        scroll.setWidget(contents)
        root.addWidget(scroll, 1)
        self._groups = groups
        self._refresh_visibility()

    def _make_editor(self, field: SettingField) -> QWidget:
        current = self._values.get(field.key, field.default)
        if field.field_type == "bool":
            editor = QCheckBox()
            editor.setChecked(current == "true")
            editor.stateChanged.connect(lambda state, key=field.key: self._queue_value(key, "true" if state else "false"))
        elif field.field_type == "choice":
            editor = QComboBox()
            choices = list(field.choices)
            if current and current not in choices:
                choices.insert(0, current)
            editor.addItems(choices)
            editor.setCurrentText(current)
            editor.currentTextChanged.connect(lambda value, key=field.key: self._queue_value(key, value))
        elif field.field_type == "text":
            editor = QPlainTextEdit()
            editor.setPlainText(current)
            editor.setMinimumHeight(70)
            editor.textChanged.connect(lambda key=field.key, widget=editor: self._queue_value(key, widget.toPlainText()))
        elif field.field_type == "percent":
            editor = QWidget()
            row = QHBoxLayout(editor)
            row.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName(field.key + "Slider")
            slider.setRange(
                field.minimum if field.minimum is not None else 0,
                field.maximum if field.maximum is not None else 100,
            )
            slider.setSingleStep(field.step if field.step is not None else 1)
            try:
                initial = int(current)
            except (TypeError, ValueError):
                initial = int(field.default or 0)
            initial = min(slider.maximum(), max(slider.minimum(), initial))
            slider.setValue(initial)
            value_label = QLabel(f"{initial}%")
            value_label.setObjectName(field.key + "Value")
            value_label.setMinimumWidth(48)
            slider.valueChanged.connect(
                lambda value, label=value_label, key=field.key: (
                    label.setText(f"{value}%"), self._queue_value(key, str(value))
                )
            )
            row.addWidget(slider, 1)
            row.addWidget(value_label)
        else:
            editor = QLineEdit()
            if field.field_type == "password":
                editor.setEchoMode(QLineEdit.EchoMode.Password)
                editor.setPlaceholderText("Stored locally; leave blank to keep it")
                if self._configured.get(field.key):
                    editor.setToolTip("A value is already stored locally. It will not be shown here.")
            else:
                editor.setText(current)
            editor.textChanged.connect(lambda value, key=field.key: self._queue_value(key, value))
        editor.setAccessibleName(field.label)
        editor.setToolTip(field.help_text or field.label)
        if field.advanced:
            editor.setProperty("advancedSetting", True)
        return editor

    @Slot(bool)
    def _set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = visible
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        for field in SETTING_FIELDS:
            visible = self._advanced_visible or not field.advanced
            for widget in self._field_rows.get(field.key, []):
                widget.setVisible(visible)
        for group in getattr(self, "_groups", {}).values():
            group.setVisible(any(not widget.isHidden() for widget in group.findChildren(QLabel)))

    @Slot(str, str)
    def _queue_value(self, key: str, value: str) -> None:
        field = self._fields.get(key)
        if field is None:
            return
        self._pending[key] = _safe_text(value)
        self._save_timer.start()

    @Slot()
    def _persist_pending(self) -> None:
        if not self._pending:
            return
        pending = dict(self._pending)
        self._pending.clear()
        # The parent owns the worker and performs the actual file operation.
        self.persist_requested.emit(pending, str(self._env_path))

    @Slot(object)
    def apply_loaded_values(self, loaded: object) -> None:
        """Apply worker-loaded values without triggering a save."""
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            return
        values, configured = loaded
        if not isinstance(values, dict) or not isinstance(configured, dict):
            return
        self._values = {str(key): _safe_text(value) for key, value in values.items()}
        self._configured = {str(key): bool(value) for key, value in configured.items()}
        for field in SETTING_FIELDS:
            editor = self._editors.get(field.key)
            if editor is None:
                continue
            blocker = QSignalBlocker(editor)
            try:
                value = self._values.get(field.key, field.default)
                if field.field_type == "bool":
                    editor.setChecked(value == "true")
                elif field.field_type == "choice":
                    editor.setCurrentText(value)
                elif field.field_type == "text":
                    editor.setPlainText(value)
                elif field.field_type == "percent":
                    slider = editor.findChild(QSlider, field.key + "Slider")
                    value_label = editor.findChild(QLabel, field.key + "Value")
                    if slider is not None:
                        slider_blocker = QSignalBlocker(slider)
                        try:
                            try:
                                numeric = int(value)
                            except (TypeError, ValueError):
                                numeric = int(field.default or 0)
                            slider.setValue(min(slider.maximum(), max(slider.minimum(), numeric)))
                        finally:
                            del slider_blocker
                    if value_label is not None and slider is not None:
                        value_label.setText(f"{slider.value()}%")
                elif field.field_type == "password":
                    editor.clear()
                    if self._configured.get(field.key):
                        editor.setToolTip("A value is already stored locally. It will not be shown here.")
                else:
                    editor.setText(value)
            finally:
                del blocker
        self._pending.clear()


class ModeratorConsole(QMainWindow):
    """Main native operator console."""

    def __init__(
        self,
        *,
        service_factory: Callable[[], CommentFeedService] | None = None,
        subreddit: str = DEFAULT_SUBREDDIT,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        max_items: int = DEFAULT_MAX_ITEMS,
        enable_tray: bool = True,
        start_feed: bool = False,
        start_activity: bool = True,
        start_in_tray: bool = False,
        recovery_notice: bool = False,
        schedule_initial_refresh: bool = True,
        env_path: Path = ENV_PATH,
    ) -> None:
        super().__init__()
        self._env_path = Path(env_path)
        self._service_factory = service_factory
        self._subreddit = subreddit
        self._poll_seconds = poll_seconds
        self._max_items = max_items
        self._tasks: list[TaskController] = []
        self._exiting = False
        self._heartbeat_timer: QTimer | None = None
        self._fast_test_timer: QTimer | None = None
        self._settings = QSettings("OptimistPrime", "ModeratorConsole")
        self._feed_controller: FeedController | None = None
        self._activity_controller: LocalActivityController | None = None
        self._tray: QSystemTrayIcon | None = None
        self._worker_dock: QDockWidget | None = None
        self._diagnostics_dock: QDockWidget | None = None

        self.setWindowTitle("Optimist Prime Moderator Console")
        self.setWindowIcon(QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon())
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Starting the local console…")
        self._build_ui()
        self._restore_ui_state()
        self._setup_tray(enable_tray)
        self._write_heartbeat()
        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.setInterval(10_000)
        self._heartbeat_timer.timeout.connect(self._write_heartbeat)
        self._heartbeat_timer.start()
        self._fast_test_timer = QTimer(self)
        self._fast_test_timer.setInterval(1000)
        self._fast_test_timer.timeout.connect(self._refresh_fast_test_status)
        self._fast_test_timer.start()
        self._refresh_fast_test_status()
        self._load_settings_in_background()
        # ``service_factory`` and ``start_feed`` remain accepted for callers
        # written against the earlier console, but the current console never
        # starts a second Reddit connection.  Activity is read from the normal
        # bot-cycle snapshot below.
        if start_activity:
            self._activity_controller = LocalActivityController(
                poll_seconds,
                subreddit,
                self,
                env_path=self._env_path,
            )
            self._connect_activity()
            self._activity_controller.start()
        elif self._feed_controller is None:
            self._set_status("Captured activity viewer is disabled; local operator tools remain available.", "warning")
        if schedule_initial_refresh:
            QTimer.singleShot(250, self.refresh_diagnostics)
            QTimer.singleShot(350, self.refresh_logs)
        if recovery_notice:
            QTimer.singleShot(1500, self._show_recovery_notice)

    def _write_heartbeat(self) -> None:
        CONSOLE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "systemTrayAvailable": QSystemTrayIcon.isSystemTrayAvailable(),
            "trayIconRegistered": bool(self._tray is not None and self._tray.isVisible()),
            # Keep the original key for watchdogs and older diagnostics.
            "trayAvailable": bool(self._tray is not None and self._tray.isVisible()),
            "windowVisible": self.isVisible(),
        }
        output = QSaveFile(str(CONSOLE_HEARTBEAT_PATH))
        if not output.open(QIODevice.OpenModeFlag.WriteOnly):
            return
        output.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        output.commit()

    def _show_recovery_notice(self) -> None:
        if self._tray is not None:
            self._tray.showMessage(
                "Optimist Prime console restarted",
                "The console was reopened by its watchdog. The bot worker may have continued while it was unavailable.",
                QSystemTrayIcon.MessageIcon.Information,
                10_000,
            )

    def _build_ui(self) -> None:
        self._feed_model = CommentFeedModel(self)
        self._feed_proxy = CommentFilterProxyModel(self)
        self._feed_proxy.setSourceModel(self._feed_model)
        self._action_model = ActionTableModel(self)
        self._retry_model = RetryTableModel(self)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter by author, post, text, or link")
        self._filter_edit.setAccessibleName("Filter comments")
        self._filter_edit.setToolTip("Search comments and posts already captured by the normal bot cycle.")
        self._filter_edit.textChanged.connect(self._feed_proxy.set_query)
        self._content_combo = QComboBox()
        self._content_combo.addItems(["All content", "Comment", "Post"])
        self._content_combo.setAccessibleName("Content type filter")
        self._content_combo.setToolTip("Limit the captured activity by content type.")
        self._content_combo.currentTextChanged.connect(self._feed_proxy.set_content_type)
        self._time_combo = QComboBox()
        self._time_combo.addItems(["Any time", "Last hour", "Last 6 hours", "Last 24 hours"])
        self._time_combo.setAccessibleName("Time filter")
        self._time_combo.setToolTip("Limit the captured activity by age.")
        self._time_combo.currentTextChanged.connect(self._feed_proxy.set_time_range)
        self._state_combo = QComboBox()
        self._state_combo.addItems(["All states", "Unprocessed", "Queued", "Processed", "Error"])
        self._state_combo.setAccessibleName("Processing state filter")
        self._state_combo.setToolTip("Limit captured activity by processing state when available.")
        self._state_combo.currentTextChanged.connect(self._feed_proxy.set_processing_state)
        self._error_combo = QComboBox()
        self._error_combo.addItems(["All items", "Errors only", "No errors"])
        self._error_combo.setAccessibleName("Error filter")
        self._error_combo.setToolTip("Show only captured rows with or without an attached processing error.")
        self._error_combo.currentTextChanged.connect(self._feed_proxy.set_error_filter)

        self._refresh_button = self._button("Refresh captured activity", "Reload items already captured by the bot; this makes no Reddit request.", self.refresh_activity)
        self._pause_button = self._button("Pause", "Pause local activity snapshot checks while keeping captured items visible.", self.pause_activity)
        self._resume_button = self._button("Resume", "Resume local activity snapshot checks.", self.resume_activity)
        self._diagnostics_button = self._button("Refresh diagnostics", "Read local health and worker evidence in a background thread.", self.refresh_diagnostics)
        self._logs_button = self._button("Refresh logs", "Read the latest local operational log lines in a worker thread.", self.refresh_logs)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        filter_row.addWidget(self._filter_edit, 1)
        filter_row.addWidget(self._content_combo)
        filter_row.addWidget(self._time_combo)
        filter_row.addWidget(self._state_combo)
        filter_row.addWidget(self._error_combo)
        filter_row.addWidget(self._refresh_button)
        filter_row.addWidget(self._pause_button)
        filter_row.addWidget(self._resume_button)

        self._status_label = QLabel("Loading captured activity…")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setAccessibleName("Captured activity indicator")
        self._identity_label = QLabel("Source: local bot-cycle snapshot")
        self._identity_label.setObjectName("identityLabel")

        status_row = QHBoxLayout()
        status_row.addWidget(self._status_label, 1)
        status_row.addWidget(self._identity_label)
        status_row.addWidget(self._progress)

        self._feed_table = QTableView()
        self._feed_table.setModel(self._feed_proxy)
        self._feed_table.setSortingEnabled(True)
        self._feed_table.setAlternatingRowColors(True)
        self._feed_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._feed_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._feed_table.setAccessibleName("Recently captured Reddit activity")
        self._feed_table.setToolTip("Read-only comments and posts already captured by the normal bot cycle.")
        self._feed_table.horizontalHeader().setStretchLastSection(True)
        self._feed_table.horizontalHeader().setMinimumSectionSize(90)
        self._feed_table.verticalHeader().setDefaultSectionSize(32)
        self._feed_table.setColumnWidth(0, 155)
        self._feed_table.setColumnWidth(1, 150)
        self._feed_table.setColumnWidth(2, 220)
        self._feed_table.setColumnWidth(3, 70)
        self._feed_table.setColumnWidth(4, 105)
        self._feed_table.setColumnWidth(5, 160)
        self._feed_table.selectionModel().currentChanged.connect(self._show_selected_comment)

        self._details = QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setAccessibleName("Selected comment details")
        self._details.setPlaceholderText("Select a comment to see its full read-only details.")

        feed_splitter = QSplitter(Qt.Orientation.Horizontal)
        feed_splitter.addWidget(self._feed_table)
        feed_splitter.addWidget(self._details)
        feed_splitter.setStretchFactor(0, 3)
        feed_splitter.setStretchFactor(1, 2)

        feed_panel = QWidget()
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(16, 16, 16, 16)
        feed_layout.addLayout(filter_row)
        feed_layout.addLayout(status_row)
        feed_layout.addWidget(feed_splitter, 1)

        self._settings_panel = SettingsPanel(self._env_path)
        self._settings_panel.persisted.connect(lambda message: self._set_status(message, "info"))
        self._settings_panel.persist_requested.connect(self._persist_settings)
        self._settings_panel._advanced_check.toggled.connect(
            lambda visible: self._settings.setValue("settings/showAdvanced", bool(visible))
        )
        self._logs = QPlainTextEdit()
        self._logs.setReadOnly(True)
        self._logs.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._logs.setAccessibleName("Operational logs")
        self._diagnostics = QPlainTextEdit()
        self._diagnostics.setReadOnly(True)
        self._diagnostics.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._diagnostics.setAccessibleName("Diagnostics")

        self._actions_table = QTableView()
        self._actions_table.setModel(self._action_model)
        self._actions_table.setAlternatingRowColors(True)
        self._actions_table.horizontalHeader().setStretchLastSection(True)
        self._retries_table = QTableView()
        self._retries_table.setModel(self._retry_model)
        self._retries_table.setAlternatingRowColors(True)
        self._retries_table.horizontalHeader().setStretchLastSection(True)

        tabs = QTabWidget()
        tabs.setAccessibleName("Console panels")
        tabs.addTab(self._settings_panel, "Settings")
        tabs.addTab(self._logs, "Logs")
        tables_panel = QWidget()
        tables_layout = QVBoxLayout(tables_panel)
        tables_layout.setContentsMargins(8, 8, 8, 8)
        tables_layout.addWidget(QLabel("Recent local operator actions"))
        tables_layout.addWidget(self._actions_table, 1)
        tables_layout.addWidget(QLabel("Retry and failure observations"))
        tables_layout.addWidget(self._retries_table, 1)
        tabs.addTab(tables_panel, "Actions and retries")

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(feed_panel)
        main_splitter.addWidget(tabs)
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 2)
        self.setCentralWidget(main_splitter)

        self._build_worker_dock()
        self._build_diagnostics_dock()
        self._build_menu()
        self._apply_style()

    def _button(self, label: str, tooltip: str, handler: Callable[[], None]) -> QPushButton:
        button = QPushButton(label)
        button.setAccessibleName(label)
        button.setToolTip(tooltip)
        button.setStatusTip(tooltip)
        button.clicked.connect(handler)
        return button

    def _build_worker_dock(self) -> None:
        dock = QDockWidget("Worker controls", self)
        dock.setObjectName("workerDock")
        self._worker_dock = dock
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        note = QLabel(
            "These controls affect a worker that you started separately. "
            "The console does not install or manage background tasks."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        fast_test_group = QGroupBox("Temporary testing speed")
        fast_test_layout = QVBoxLayout(fast_test_group)
        self._fast_test_status = QLabel()
        self._fast_test_status.setWordWrap(True)
        self._fast_test_status.setAccessibleName("Temporary testing status")
        self._fast_test_button = self._button(
            "Start 2-minute fast testing",
            "Temporarily ask the existing worker to begin the next cycle now, then wait 3 seconds after each cycle for up to 2 minutes. All normal feature, overlap, and Reddit request limits remain active.",
            self._toggle_fast_test_mode,
        )
        fast_test_layout.addWidget(self._fast_test_status)
        fast_test_layout.addWidget(self._fast_test_button)
        layout.addWidget(fast_test_group)
        layout.addStretch(1)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_diagnostics_dock(self) -> None:
        dock = QDockWidget("Diagnostics", self)
        dock.setObjectName("diagnosticsDock")
        self._diagnostics_dock = dock
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        row = QHBoxLayout()
        row.addWidget(QLabel("Local health and recent worker evidence"), 1)
        row.addWidget(self._diagnostics_button)
        layout.addLayout(row)
        layout.addWidget(self._diagnostics, 1)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_menu(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        view_menu.setAccessibleName("View menu")
        for title, shortcut, delta in (("Increase text size", "Ctrl+=", 0.1), ("Decrease text size", "Ctrl+-", -0.1)):
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.setToolTip(title)
            action.triggered.connect(lambda checked=False, delta=delta: self._adjust_font_scale(delta))
            view_menu.addAction(action)
        view_menu.addSeparator()
        maximize_action = QAction("Maximize console", self)
        maximize_action.setShortcut(QKeySequence("Ctrl+Shift+M"))
        maximize_action.setToolTip("Maximize the console within the monitor work area.")
        maximize_action.triggered.connect(lambda: self.invoke_action("window.maximize"))
        view_menu.addAction(maximize_action)
        restore_action = QAction("Restore console", self)
        restore_action.setToolTip("Restore the console to its saved window bounds.")
        restore_action.triggered.connect(lambda: self.invoke_action("window.restore"))
        view_menu.addAction(restore_action)
        view_menu.addSeparator()
        view_menu.addAction(self._worker_dock.toggleViewAction())
        view_menu.addAction(self._diagnostics_dock.toggleViewAction())
        view_menu.addSeparator()
        refresh_action = QAction("Refresh captured activity", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self.refresh_activity)
        view_menu.addAction(refresh_action)
        exit_action = QAction("Exit console", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(lambda: self.invoke_action("window.exit"))
        view_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About this console", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QGroupBox { margin-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLabel#statusLabel { padding: 6px 10px; border-radius: 4px; background: palette(alternate-base); }
            QLabel#identityLabel { padding: 6px 10px; }
            QLabel#settingHelp { color: palette(mid); padding-bottom: 3px; }
            QTableView { selection-background-color: palette(highlight); selection-color: palette(highlighted-text); }
            QPlainTextEdit { padding: 8px; }
            QProgressBar { min-width: 72px; max-width: 120px; min-height: 10px; max-height: 10px; }
            """
        )

    def _restore_ui_state(self) -> None:
        geometry = self._settings.value("geometry")
        state = self._settings.value("windowState")
        restored_geometry = bool(geometry and self.restoreGeometry(geometry))
        if restored_geometry:
            available_areas = [screen.availableGeometry() for screen in QApplication.screens()]
            restored_geometry = any(self.frameGeometry().intersects(area) for area in available_areas)
        if not restored_geometry:
            self.resize(1500, 950)
            self.showMaximized()
        elif _safe_text(self._settings.value("windowMaximized", "false")).casefold() in {"1", "true", "yes", "on"}:
            self.showMaximized()
        if state:
            self.restoreState(state)
        self._action_model.append_action(
            "Window",
            f"Startup state: {'maximized' if self.isMaximized() else 'restored'}; "
            f"displays available: {len(QApplication.screens())}",
        )
        self._settings_panel._advanced_check.setChecked(
            _safe_text(self._settings.value("settings/showAdvanced", "false")).casefold() in {"1", "true", "yes", "on"}
        )
        try:
            saved_scale = float(self._settings.value("fontScale", 1.0))
        except (TypeError, ValueError):
            saved_scale = 1.0
        self._font_scale = max(0.9, min(1.6, saved_scale))
        self._apply_font_scale()

    def _apply_font_scale(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        base = app.property("optimistBaseFont")
        if not isinstance(base, QFont):
            base = QFont(app.font())
            app.setProperty("optimistBaseFont", QFont(base))
        font = QFont(base)
        font.setPointSizeF(max(8.0, base.pointSizeF() * self._font_scale))
        app.setFont(font)
        self._settings.setValue("fontScale", self._font_scale)

    def _adjust_font_scale(self, delta: float) -> None:
        self._font_scale = max(0.9, min(1.6, round(self._font_scale + delta, 2)))
        self._apply_font_scale()
        self._set_status(f"Interface text scale: {int(self._font_scale * 100)}%.", "info")

    def _setup_tray(self, enabled: bool) -> None:
        if not enabled or not QSystemTrayIcon.isSystemTrayAvailable():
            return
        tray = QSystemTrayIcon(self.windowIcon(), self)
        tray.setToolTip("Optimist Prime Moderator Console")
        menu = QMenu()
        show_action = menu.addAction("Show console")
        show_action.triggered.connect(lambda: self.invoke_action("tray.show"))
        refresh_action = menu.addAction("Refresh captured activity")
        refresh_action.triggered.connect(lambda: self.invoke_action("tray.refresh_activity"))
        menu.addSeparator()
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(lambda: self.invoke_action("tray.exit"))
        tray.setContextMenu(menu)
        tray.activated.connect(self._on_tray_activated)
        tray.show()
        self._tray = tray

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self._restore_from_tray()

    def _connect_feed(self) -> None:
        assert self._feed_controller is not None
        self._feed_controller.snapshot_ready.connect(self._on_snapshot)
        self._feed_controller.progress.connect(lambda message: self._set_status(message, "info"))
        self._feed_controller.failure.connect(self._on_feed_failure)
        self._feed_controller.identity_ready.connect(lambda account: self._identity_label.setText(f"Account: u/{account}"))
        self._feed_controller.state_changed.connect(self._on_feed_state)

    def _connect_activity(self) -> None:
        assert self._activity_controller is not None
        self._activity_controller.snapshot_ready.connect(self._on_snapshot)
        self._activity_controller.progress.connect(lambda message: self._set_status(message, "info"))
        self._activity_controller.failure.connect(self._on_activity_failure)
        self._activity_controller.identity_ready.connect(
            lambda source: self._identity_label.setText(f"Source: {source}")
        )
        self._activity_controller.state_changed.connect(self._on_activity_state)

    @Slot(dict)
    def _on_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._feed_model.update_snapshot(snapshot)
        status = snapshot.get("status", {}) if isinstance(snapshot, dict) else {}
        state = _safe_text(status.get("state"), "unknown")
        message = _safe_text(status.get("message"), "Feed status updated.")
        community = _safe_text(status.get("subreddit") or self._subreddit)
        scope_label = f"r/{community}" if community else "Captured activity"
        self._set_status(
            f"{scope_label} — {message} ({status.get('cachedCount', 0)} cached)",
            "error" if state == "error" else "info",
        )
        if state == "error":
            self._retry_model.append_retry("Reddit feed", "retryable", _safe_text(status.get("lastError"), message))

    @Slot(str)
    def _on_feed_failure(self, message: str) -> None:
        self._set_status(f"Reddit feed issue: {message}", "error")
        self._retry_model.append_retry("Reddit feed", "retryable", message)

    @Slot(str)
    def _on_activity_failure(self, message: str) -> None:
        self._set_status(f"Local activity snapshot issue: {message}", "error")
        self._retry_model.append_retry("Local activity", "retryable", message)

    @Slot(str)
    def _on_feed_state(self, state: str) -> None:
        if state in {"starting", "connected"}:
            self._progress.setRange(0, 0 if state == "starting" else 1)
            if state == "connected":
                self._progress.setValue(1)
        elif state in {"paused", "stopped", "error"}:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)

    @Slot(str)
    def _on_activity_state(self, state: str) -> None:
        if state == "starting":
            self._progress.setRange(0, 0)
        elif state == "connected":
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
        elif state in {"paused", "stopped", "error"}:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)

    def _set_status(self, message: str, level: str = "info") -> None:
        self._status_label.setText(message)
        self.statusBar().showMessage(message)
        self._status_label.setProperty("statusLevel", level)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)

    def maximize_console(self) -> None:
        self.showMaximized()
        self._settings.setValue("windowMaximized", True)
        self._set_status("Console maximized within the monitor work area.", "info")

    def restore_console(self) -> None:
        self.showNormal()
        self._settings.setValue("windowMaximized", False)
        self._set_status("Console restored to its saved window bounds.", "info")

    def available_actions(self) -> list[dict[str, Any]]:
        """Return the local, machine-readable action contract for this console."""
        actions: list[dict[str, Any]] = []
        activity_available = self._activity_controller is not None and self._activity_controller.is_running()
        for definition in CONSOLE_ACTIONS:
            action = dict(definition)
            action_id = _safe_text(action.get("id"))
            enabled = True
            reason = ""
            if action_id.startswith("activity.") and not activity_available:
                enabled = False
                reason = "The captured-activity viewer is not running."
            if action_id == "tray.show" and self._tray is None:
                enabled = False
                reason = "The system tray is not available for this console instance."
            if action_id == "tray.refresh_activity" and (self._tray is None or not activity_available):
                enabled = False
                reason = "The system tray and captured-activity viewer must both be available."
            if action_id == "window.maximize" and self.isMaximized():
                enabled = False
                reason = "The console is already maximized."
            if action_id == "window.restore" and not self.isMaximized():
                enabled = False
                reason = "The console is already restored."
            action["enabled"] = enabled
            action["reason"] = reason
            parameters = list(action.get("parameters", []))
            if action.get("requires_confirmation"):
                parameters.append("confirmed")
            action["parameters"] = parameters
            actions.append(action)
        return actions

    def invoke_action(self, action_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a visible-console action through the same handlers as the UI."""
        parameters = parameters or {}
        definition = next((item for item in CONSOLE_ACTIONS if item["id"] == action_id), None)
        correlation_id = uuid.uuid4().hex[:12]
        if definition is None:
            return {"status": "rejected", "actionId": action_id, "reason": "Unknown action.", "correlationId": correlation_id}
        if action_id == "worker.fast_test" and not isinstance(parameters.get("active"), bool):
            return {
                "status": "rejected",
                "actionId": action_id,
                "reason": "Choose whether temporary fast testing should be on or off.",
                "correlationId": correlation_id,
            }
        available = next(item for item in self.available_actions() if item["id"] == action_id)
        if not available["enabled"]:
            return {
                "status": "rejected",
                "actionId": action_id,
                "reason": available["reason"],
                "correlationId": correlation_id,
            }
        if definition.get("requires_confirmation") and parameters.get("confirmed") is not True:
            return {
                "status": "rejected",
                "actionId": action_id,
                "reason": "Explicit confirmation is required for this local destructive action.",
                "correlationId": correlation_id,
            }
        handlers: dict[str, Callable[[], None]] = {
            "activity.refresh": self.refresh_activity,
            "activity.pause": self.pause_activity,
            "activity.resume": self.resume_activity,
            "diagnostics.refresh": self.refresh_diagnostics,
            "logs.refresh": self.refresh_logs,
            "window.maximize": self.maximize_console,
            "window.restore": self.restore_console,
            "window.exit": self._exit_application,
            "tray.show": self._restore_from_tray,
            "tray.refresh_activity": self.refresh_activity,
            "tray.exit": self._exit_application,
        }
        if action_id == "settings.toggle_advanced":
            visible = parameters.get("visible")
            handler = lambda: self._settings_panel._advanced_check.setChecked(
                not self._settings_panel._advanced_check.isChecked() if visible is None else bool(visible)
            )
        elif action_id == "worker.fast_test":
            handler = lambda: self._set_fast_test_mode(bool(parameters.get("active")))
        elif action_id != "settings.toggle_advanced":
            handler = handlers.get(action_id)
        if handler is None:
            return {"status": "rejected", "actionId": action_id, "reason": "No handler is registered.", "correlationId": correlation_id}
        try:
            handler()
        except Exception as exc:
            message = _safe_error(exc)
            self._action_model.append_action(f"Action {action_id}", f"FAILED — {message}")
            return {"status": "failed", "actionId": action_id, "message": message, "correlationId": correlation_id}
        message = f"Action requested: {definition['label']}."
        self._action_model.append_action(f"Action {action_id}", message)
        return {"status": "requested", "actionId": action_id, "message": message, "correlationId": correlation_id}

    def refresh_activity(self) -> None:
        if self._activity_controller is None or not self._activity_controller.is_running():
            self._set_status("The captured-activity viewer is not running. Start the console again.", "warning")
            return
        self._progress.setRange(0, 0)
        self._set_status("Reloading captured activity; no Reddit request will be made…", "info")
        self._activity_controller.refresh_requested.emit()

    def pause_activity(self) -> None:
        if self._activity_controller is not None:
            self._activity_controller.pause_requested.emit()

    def resume_activity(self) -> None:
        if self._activity_controller is not None:
            self._activity_controller.resume_requested.emit()

    # Compatibility aliases for integrations written against the old viewer.
    refresh_feed = refresh_activity
    pause_feed = pause_activity
    resume_feed = resume_activity

    def _run_task(self, operation: Callable[[], Any], on_success: Callable[[Any], None], label: str) -> None:
        task = TaskController(operation, self)
        self._tasks.append(task)
        task.succeeded.connect(on_success)
        task.failed.connect(lambda message, label=label: self._task_failed(label, message))
        task.finished.connect(lambda task=task: self._task_finished(task))
        if not task.start():
            self._task_failed(label, "An identical background operation is already running.")
            self._tasks.remove(task)
            task.deleteLater()

    def _task_finished(self, task: TaskController) -> None:
        if task in self._tasks:
            self._tasks.remove(task)
        task.deleteLater()

    def _task_failed(self, label: str, message: str) -> None:
        self._set_status(f"{label} failed: {message}", "error")
        self._action_model.append_action(label, f"FAILED — {message}")
        self._retry_model.append_retry(label, "retryable", message)

    def refresh_diagnostics(self) -> None:
        self._set_status("Reading diagnostics in the background…", "info")
        self._run_task(
            lambda: read_owner_diagnostics(self._env_path),
            self._show_diagnostics,
            "Diagnostics",
        )

    def _load_settings_in_background(self) -> None:
        self._run_task(
            lambda: load_settings_for_ui(self._env_path),
            self._settings_panel.apply_loaded_values,
            "Settings load",
        )

    @Slot(object, str)
    def _persist_settings(self, payload: object, path: str) -> None:
        if not isinstance(payload, dict):
            return
        pending = {str(key): _safe_text(value) for key, value in payload.items()}
        self._run_task(
            lambda: persist_settings_for_ui(pending, Path(path)),
            lambda count: self._settings_saved(int(count)),
            "Settings",
        )

    def _settings_saved(self, count: int) -> None:
        self._settings_panel.persisted.emit(f"Saved {count} setting(s) locally.")

    def _show_diagnostics(self, result: Any) -> None:
        text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        self._diagnostics.setPlainText(_redact_log_text(text))
        self._action_model.append_action("Diagnostics", "Updated local health summary")
        self._set_status("Diagnostics updated. No credential values are displayed.", "info")

    def refresh_logs(self) -> None:
        self._run_task(lambda: read_log_tail(env_path=self._env_path), self._show_logs, "Logs")

    def _show_logs(self, text: Any) -> None:
        self._logs.setPlainText(_safe_text(text))
        self._action_model.append_action("Logs", "Updated recent operational log view")

    def _toggle_fast_test_mode(self) -> None:
        current = get_fast_test_mode()
        self.invoke_action("worker.fast_test", {"active": not current.get("active", False)})

    def _set_fast_test_mode(self, active: bool) -> None:
        result = start_fast_test_mode() if active else stop_fast_test_mode()
        self._refresh_fast_test_status()
        if result.get("active"):
            self._set_status(
                "Fast testing is on for up to 2 minutes; the next cycle will start as soon as the worker sees the request.",
                "warning",
            )
        else:
            self._set_status("Fast testing is off; the worker will return to its normal adaptive timing.", "info")

    def _refresh_fast_test_status(self) -> None:
        if not hasattr(self, "_fast_test_status"):
            return
        status = get_fast_test_mode()
        if status.get("active"):
            remaining = int(status.get("remaining_seconds", 0))
            minutes, seconds = divmod(remaining, 60)
            self._fast_test_status.setText(
                f"Fast testing is active: the worker waits {FAST_INTERVAL_SECONDS} seconds after each cycle, "
                "without overlapping runs. "
                f"It turns itself off in {minutes}:{seconds:02d}."
            )
            self._fast_test_button.setText("Stop fast testing")
        else:
            self._fast_test_status.setText(
                "Fast testing is off. It can run for up to 2 minutes; then the bot returns to its usual timing."
            )
            self._fast_test_button.setText("Start 2-minute fast testing")

    def _show_selected_comment(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            self._details.clear()
            return
        source_index = self._feed_proxy.mapToSource(current)
        row = self._feed_model.comment_for_index(source_index)
        if row is None:
            self._details.clear()
            return
        details = {
            "author": row.get("author"),
            "subreddit": row.get("subreddit"),
            "post": row.get("linkTitle"),
            "created": row.get("createdAt"),
            "score": row.get("score"),
            "processingState": row.get("processingState") or row.get("state") or "unprocessed",
            "error": row.get("error") or row.get("lastError") or row.get("errorMessage"),
            "permalink": row.get("permalink"),
            "bodyTruncated": row.get("bodyTruncated", False),
            "body": row.get("body", ""),
        }
        self._details.setPlainText(json.dumps(details, indent=2, ensure_ascii=False, default=str))

    def _restore_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "Optimist Prime Moderator Console",
            "A local PySide6 / Qt Widgets operator console for a Reddit bot.\n\n"
            "The activity view is read-only and shows items already captured by the bot cycle. "
            "It does not make a second Reddit request or perform moderation actions.",
        )

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._tray is not None and not self._exiting:
            self.hide()
            self.statusBar().showMessage("Console hidden in the system tray. Use the tray menu to restore or exit.")
            event.accept()
            return
        self._shutdown()
        event.accept()

    def _exit_application(self) -> None:
        self._exiting = True
        self.close()
        # A console started hidden in the tray may not emit
        # lastWindowClosed when its already-hidden main window is closed.
        # Quit explicitly so the watchdog can acquire the single-instance lock.
        QApplication.quit()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self._tray is not None
            and self.isMinimized()
        ):
            QTimer.singleShot(0, self.hide)
            self.statusBar().showMessage("Console minimized to the system tray. Use the tray icon to restore it.")
        super().changeEvent(event)

    def _shutdown(self) -> None:
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        self._settings.setValue("windowMaximized", self.isMaximized())
        if self._fast_test_timer is not None:
            self._fast_test_timer.stop()
        for task in list(self._tasks):
            task.wait()
        if self._feed_controller is not None:
            self._set_status("Stopping background feed work…", "info")
            self._feed_controller.stop()
        if self._activity_controller is not None:
            self._set_status("Stopping local activity work…", "info")
            self._activity_controller.stop()
        if self._tray is not None:
            self._tray.hide()
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.stop()
        try:
            record = json.loads(CONSOLE_HEARTBEAT_PATH.read_text(encoding="utf-8"))
            if record.get("pid") == os.getpid():
                CONSOLE_HEARTBEAT_PATH.unlink(missing_ok=True)
        except (OSError, ValueError, AttributeError):
            pass


def _build_service_factory(subreddit: str, poll_seconds: float, max_items: int) -> Callable[[], CommentFeedService]:
    def factory() -> CommentFeedService:
        # Authentication and local configuration loading happen on the feed
        # worker thread, never on the GUI thread.
        clear_inherited_runtime_environment()
        load_local_env(ENV_PATH, override=True)
        try:
            configure_logging(also_stderr=False)
        except Exception:
            pass
        expected_username = os.environ.get("BOT_EXPECTED_USERNAME", "").strip() or None
        reddit, account_name = create_read_only_reddit(expected_username=expected_username)
        return CommentFeedService(
            reddit,
            subreddit,
            account_name=account_name,
            poll_seconds=poll_seconds,
            max_items=max_items,
        )

    return factory


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Optimist Prime PySide6 moderator console")
    parser.add_argument("--subreddit", default=DEFAULT_SUBREDDIT, help="Optional subreddit name for the activity display")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS, help="Feed refresh interval")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS, help="Maximum cached comments")
    parser.add_argument("--no-tray", action="store_true", help="Do not use a system-tray icon")
    parser.add_argument("--start-in-tray", action="store_true", help="Start hidden with the tray icon available")
    parser.add_argument("--recovery-notice", action="store_true", help="Show a tray notice after watchdog recovery")
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Ask an already-running console to show its window, then exit",
    )
    parser.add_argument(
        "--enable-feed",
        action="store_true",
        help="Compatibility option; the console now shows only cached activity and makes no Reddit request",
    )
    parser.add_argument(
        "--no-feed",
        action="store_true",
        help="Disable the local cached-activity viewer (compatibility option)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _request_existing_console_show(server_name: str = CONSOLE_CONTROL_SERVER_NAME) -> bool:
    """Send a show request to the current console without touching the worker."""
    client = QLocalSocket()
    client.connectToServer(server_name)
    if not client.waitForConnected(1200):
        return False
    client.disconnectFromServer()
    return True


def _install_console_control_server(
    app: QApplication,
    window: ModeratorConsole,
    server_name: str = CONSOLE_CONTROL_SERVER_NAME,
) -> QLocalServer:
    """Install a local show-window command for the single running console."""
    server = QLocalServer(app)
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    if not server.listen(server_name):
        # The instance lock is already held, so a listener with this name can
        # only be left behind by a console that has exited unexpectedly.
        server.removeServer(server_name)
        if not server.listen(server_name):
            raise RuntimeError("Could not open the console's local control channel")

    def accept_connections() -> None:
        while server.hasPendingConnections():
            socket = server.nextPendingConnection()
            window._restore_from_tray()
            socket.disconnectFromServer()
            socket.disconnected.connect(socket.deleteLater)

    server.newConnection.connect(accept_connections)
    return server


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clear_inherited_runtime_environment()

    app = QApplication([sys.argv[0]])
    app.setApplicationName("Optimist Prime Moderator Console")
    app.setOrganizationName("OptimistPrime")
    CONSOLE_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(str(CONSOLE_LOCK_PATH))
    instance_lock.setStaleLockTime(30_000)
    if not instance_lock.tryLock(0):
        if args.show_existing:
            return 0 if _request_existing_console_show() else 2
        return 0
    if args.show_existing:
        # The request is specifically for an existing window; do not create a
        # new one when no console is currently holding the instance lock.
        return 2
    window = ModeratorConsole(
        subreddit=args.subreddit,
        poll_seconds=args.poll_seconds,
        max_items=args.max_items,
        enable_tray=not args.no_tray,
        start_activity=not args.no_feed,
        start_in_tray=args.start_in_tray,
        recovery_notice=args.recovery_notice,
    )
    window._instance_lock = instance_lock
    window._control_server = _install_console_control_server(app, window)
    if args.start_in_tray and window._tray is not None:
        window.hide()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
