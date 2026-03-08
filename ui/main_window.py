"""
メインウィンドウ

AnnotFlow アプリケーションのメインウィンドウ。
タブ形式で各パネル（インポート・品質チェック・分割・エクスポート）を提供する。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMainWindow, QTabWidget
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ（Phase 1 で実装）。"""

    MIN_WIDTH = 900
    MIN_HEIGHT = 600

    def __init__(self) -> None:
        super().__init__()
        self._load_stylesheet()
        self.setWindowTitle("AnnotFlow")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

    def _load_stylesheet(self) -> None:
        """assets/style.qss を読み込んでアプリ全体に適用する。"""
        qss_path = Path(__file__).parent.parent / "assets" / "style.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
