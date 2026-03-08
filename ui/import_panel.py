"""
インポートパネル

アノテーションデータのインポートUIを提供する。
ImportWorker を使ってバックグラウンドでファイルI/Oを実行する。

対応フォーマット：
  - COCO JSON (.json)
  - YOLO (data.yaml または classes.txt)
  - Label Studio JSON (.json)
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dataset import Dataset
from workers.import_worker import ImportWorker


# フォーマット表示名 → ImportWorker 内部キーのマッピング
_FORMAT_MAP: dict[str, str] = {
    "COCO JSON": "coco",
    "YOLO (data.yaml / classes.txt)": "yolo",
    "Label Studio JSON": "label_studio",
}


class ImportPanel(QWidget):
    """インポートパネル。"""

    # インポート完了時に Dataset を MainWindow へ渡すシグナル
    dataset_imported = Signal(object)
    # ステータスバー更新シグナル
    status_updated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[ImportWorker] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """UIを構築する。"""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        # ── フォーマット選択 ──
        fmt_group = QGroupBox("フォーマット")
        fmt_layout = QVBoxLayout()
        fmt_layout.setContentsMargins(16, 16, 16, 16)
        fmt_layout.setSpacing(8)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(list(_FORMAT_MAP.keys()))
        self._fmt_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        fmt_layout.addWidget(self._fmt_combo)
        fmt_group.setLayout(fmt_layout)
        root_layout.addWidget(fmt_group)

        # ── ファイル選択 ──
        file_group = QGroupBox("ファイル選択")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(16, 16, 16, 16)
        file_layout.setSpacing(10)

        # アノテーションファイル行
        self._ann_label = QLabel("アノテーションファイル:")
        self._ann_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        ann_row = QHBoxLayout()
        ann_row.setSpacing(8)
        self._ann_path_edit = QLineEdit()
        self._ann_path_edit.setPlaceholderText("アノテーションファイルを選択...")
        self._ann_path_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._ann_browse_btn = QPushButton("参照...")
        self._ann_browse_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._ann_browse_btn.clicked.connect(self._browse_annotation)
        ann_row.addWidget(self._ann_path_edit)
        ann_row.addWidget(self._ann_browse_btn)

        # 画像フォルダ行
        img_label = QLabel("画像フォルダ:")
        img_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        img_row = QHBoxLayout()
        img_row.setSpacing(8)
        self._img_folder_edit = QLineEdit()
        self._img_folder_edit.setPlaceholderText(
            "画像が格納されているフォルダを選択..."
        )
        self._img_folder_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        img_browse_btn = QPushButton("参照...")
        img_browse_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        img_browse_btn.clicked.connect(self._browse_image_folder)
        img_row.addWidget(self._img_folder_edit)
        img_row.addWidget(img_browse_btn)

        file_layout.addWidget(self._ann_label)
        file_layout.addLayout(ann_row)
        file_layout.addWidget(img_label)
        file_layout.addLayout(img_row)
        file_group.setLayout(file_layout)
        root_layout.addWidget(file_group)

        # ── 実行ボタン行 ──
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self._import_btn = QPushButton("インポート実行")
        self._import_btn.setProperty("role", "primary")
        self._import_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._import_btn.clicked.connect(self._start_import)
        self._cancel_btn = QPushButton("キャンセル")
        self._cancel_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_import)
        run_row.addWidget(self._import_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch()
        root_layout.addLayout(run_row)

        # ── プログレスバー ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._progress_bar.setVisible(False)
        root_layout.addWidget(self._progress_bar)

        # ── 結果サマリー ──
        result_group = QGroupBox("インポート結果")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(10)

        self._result_label = QLabel("インポートを実行すると結果が表示されます。")
        self._result_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        result_layout.addWidget(self._result_label)

        # スキップされたレコードのテーブル
        self._skip_table = QTableWidget(0, 2)
        self._skip_table.setHorizontalHeaderLabels(["ファイル名", "スキップ理由"])
        self._skip_table.horizontalHeader().setStretchLastSection(True)
        self._skip_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._skip_table.setVisible(False)
        result_layout.addWidget(self._skip_table)

        result_group.setLayout(result_layout)
        root_layout.addWidget(result_group)

    # ------------------------------------------------------------------
    # イベントハンドラ
    # ------------------------------------------------------------------

    def _on_format_changed(self, _index: int) -> None:
        """フォーマット変更時にラベルを切り替える。"""
        fmt = self._fmt_combo.currentText()
        if "YOLO" in fmt:
            self._ann_label.setText("アノテーションファイル (data.yaml / classes.txt):")
        else:
            self._ann_label.setText("アノテーションファイル (.json):")

    def _browse_annotation(self) -> None:
        """アノテーションファイルを選択する。"""
        fmt = self._fmt_combo.currentText()
        if "YOLO" in fmt:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "アノテーションファイルを選択",
                "",
                "YAML/TXT Files (*.yaml *.yml *.txt);;All Files (*)",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "アノテーションファイルを選択",
                "",
                "JSON Files (*.json);;All Files (*)",
            )
        if path:
            self._ann_path_edit.setText(path)

    def _browse_image_folder(self) -> None:
        """画像フォルダを選択する。"""
        path = QFileDialog.getExistingDirectory(self, "画像フォルダを選択")
        if path:
            self._img_folder_edit.setText(path)

    def _start_import(self) -> None:
        """インポートを開始する。"""
        ann_path = self._ann_path_edit.text().strip()
        img_folder = self._img_folder_edit.text().strip()
        fmt_key = _FORMAT_MAP[self._fmt_combo.currentText()]

        if not ann_path:
            QMessageBox.warning(
                self, "入力エラー", "アノテーションファイルを選択してください。"
            )
            return
        if not img_folder:
            QMessageBox.warning(
                self, "入力エラー", "画像フォルダを選択してください。"
            )
            return

        # UI 状態をロック
        self._import_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._skip_table.setVisible(False)
        self._skip_table.setRowCount(0)
        self._result_label.setText("インポート中...")

        # Worker 起動
        self._worker = ImportWorker(ann_path, img_folder, fmt_key)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.status_message.connect(self.status_updated)
        self._worker.finished.connect(self._on_import_finished)
        self._worker.error.connect(self._on_import_error)
        self._worker.skipped.connect(self._on_skipped)
        self._worker.start()

    def _cancel_import(self) -> None:
        """インポートをキャンセルする。"""
        if self._worker:
            self._worker.cancel()
        self._reset_ui()
        self._result_label.setText("キャンセルされました。")

    def _on_import_finished(self, dataset: Dataset) -> None:
        """インポート完了時の処理。"""
        self._reset_ui()
        class_names = ", ".join(
            dataset.class_map.id_to_name[i]
            for i in sorted(dataset.class_map.id_to_name.keys())
        )
        msg = (
            f"インポート完了！\n"
            f"  画像数: {len(dataset.records)} 件\n"
            f"  クラス数: {len(dataset.class_map.id_to_name)} 種 ({class_names})"
        )
        self._result_label.setText(msg)
        self.dataset_imported.emit(dataset)

    def _on_import_error(self, message: str) -> None:
        """インポートエラー時の処理。"""
        self._reset_ui()
        self._result_label.setText("エラーが発生しました。詳細はダイアログを確認してください。")
        QMessageBox.critical(
            self, "インポートエラー", f"インポートに失敗しました。\n\n{message}"
        )

    def _on_skipped(self, skipped_records: list[dict]) -> None:
        """スキップされたレコードをテーブルに表示する。"""
        if not skipped_records:
            return
        self._skip_table.setRowCount(len(skipped_records))
        for row, record in enumerate(skipped_records):
            self._skip_table.setItem(
                row, 0, QTableWidgetItem(record.get("filename", ""))
            )
            self._skip_table.setItem(
                row, 1, QTableWidgetItem(record.get("reason", ""))
            )
        self._skip_table.setVisible(True)

    def _reset_ui(self) -> None:
        """UI 状態をリセットする。"""
        self._import_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setVisible(False)
