"""
メインウィンドウ

AnnotFlow アプリケーションのメインウィンドウ。
QTabWidget でインポート・品質チェック・分割・エクスポートの4タブを提供し、
各パネル間のデータ受け渡しを担う。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
)

from core.dataset import Dataset
from ui.import_panel import ImportPanel
from ui.stats_panel import StatsPanel
from ui.split_panel import SplitPanel
from ui.export_panel import ExportPanel


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ。"""

    MIN_WIDTH = 900
    MIN_HEIGHT = 600

    def __init__(self) -> None:
        super().__init__()
        self._dataset: Optional[Dataset] = None
        self._setup_ui()
        self._load_stylesheet()
        self.setWindowTitle("AnnotFlow")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)

    def _setup_ui(self) -> None:
        """UIを構築する。"""
        # タブウィジェット
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # 各パネルを作成
        self._import_panel = ImportPanel()
        self._stats_panel = StatsPanel()
        self._split_panel = SplitPanel()
        self._export_panel = ExportPanel()

        # タブに追加
        self._tabs.addTab(self._import_panel, "インポート")
        self._tabs.addTab(self._stats_panel, "品質チェック")
        self._tabs.addTab(self._split_panel, "分割")
        self._tabs.addTab(self._export_panel, "エクスポート")

        # ステータスバー
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("準備完了")

        # パネル間シグナル接続
        self._import_panel.dataset_imported.connect(self._on_dataset_imported)
        self._import_panel.status_updated.connect(self._status_bar.showMessage)
        self._stats_panel.status_updated.connect(self._status_bar.showMessage)
        self._split_panel.status_updated.connect(self._status_bar.showMessage)
        self._split_panel.dataset_split.connect(self._on_dataset_split)
        self._export_panel.status_updated.connect(self._status_bar.showMessage)

    def _load_stylesheet(self) -> None:
        """assets/style.qss を読み込んでアプリ全体に適用する。"""
        qss_path = Path(__file__).parent.parent / "assets" / "style.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _on_dataset_imported(self, dataset: Dataset) -> None:
        """インポート完了時にデータセットを各パネルに配布する。"""
        self._dataset = dataset
        self._stats_panel.set_dataset(dataset)
        self._split_panel.set_dataset(dataset)
        # 分割前はエクスポート不可
        self._export_panel.set_dataset(None)
        self._status_bar.showMessage(
            f"インポート完了: {len(dataset.records)} 件 ／ "
            f"クラス: {len(dataset.class_map.id_to_name)} 種"
        )

    def _on_dataset_split(self, dataset: Dataset) -> None:
        """分割完了時にエクスポートパネルに反映する。"""
        self._dataset = dataset
        self._export_panel.set_dataset(dataset)
        self._status_bar.showMessage(
            "分割完了。エクスポートタブからエクスポートできます。"
        )
