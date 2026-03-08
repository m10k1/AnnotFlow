"""
エクスポートパネル

YOLO / COCO フォーマットでのエクスポートUIを提供する。
ExportWorker を使ってバックグラウンドでファイルI/Oを実行する。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.dataset import Dataset
from workers.export_worker import ExportWorker


class ExportPanel(QWidget):
    """エクスポートパネル。"""

    status_updated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._worker: Optional[ExportWorker] = None
        self._output_dir: str = ""
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """UIを構築する。"""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        # 未分割時の案内
        self._no_data_label = QLabel(
            "先に分割タブでデータセットを分割してください。"
        )
        self._no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_data_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout.addWidget(self._no_data_label)

        # コンテンツエリア
        self._content = QWidget()
        self._content.setVisible(False)
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout.addWidget(self._content)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # ── フォーマット選択（ラジオボタン）──
        fmt_group = QGroupBox("エクスポートフォーマット")
        fmt_layout = QVBoxLayout()
        fmt_layout.setContentsMargins(16, 16, 16, 16)
        fmt_layout.setSpacing(8)
        self._fmt_button_group = QButtonGroup(self)
        self._yolo_radio = QRadioButton("YOLO")
        self._yolo_radio.setChecked(True)
        self._coco_radio = QRadioButton("COCO JSON")
        self._fmt_button_group.addButton(self._yolo_radio)
        self._fmt_button_group.addButton(self._coco_radio)
        fmt_layout.addWidget(self._yolo_radio)
        fmt_layout.addWidget(self._coco_radio)
        fmt_group.setLayout(fmt_layout)
        content_layout.addWidget(fmt_group)

        # ── 出力先 ──
        output_group = QGroupBox("出力先フォルダ")
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(16, 16, 16, 16)
        output_layout.setSpacing(10)
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("出力先フォルダを選択...")
        self._output_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        out_browse_btn = QPushButton("参照...")
        out_browse_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        out_browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self._output_edit)
        out_row.addWidget(out_browse_btn)
        output_layout.addLayout(out_row)
        output_group.setLayout(output_layout)
        content_layout.addWidget(output_group)

        # ── オプション ──
        option_group = QGroupBox("オプション")
        option_layout = QVBoxLayout()
        option_layout.setContentsMargins(16, 16, 16, 16)
        option_layout.setSpacing(10)
        self._copy_images_check = QCheckBox("画像ファイルも出力フォルダにコピーする")
        self._copy_images_check.setChecked(True)
        option_layout.addWidget(self._copy_images_check)
        option_group.setLayout(option_layout)
        content_layout.addWidget(option_group)

        # ── 実行ボタン ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._export_btn = QPushButton("エクスポート実行")
        self._export_btn.setProperty("role", "primary")
        self._export_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._export_btn.clicked.connect(self._start_export)
        self._cancel_btn = QPushButton("キャンセル")
        self._cancel_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_export)
        btn_row.addWidget(self._export_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        # ── プログレスバー ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._progress_bar.setVisible(False)
        content_layout.addWidget(self._progress_bar)

        # ── 完了後：フォルダを開くボタン ──
        self._open_folder_btn = QPushButton("出力フォルダをエクスプローラーで開く")
        self._open_folder_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._open_folder_btn.setVisible(False)
        self._open_folder_btn.clicked.connect(self._open_output_folder)
        content_layout.addWidget(self._open_folder_btn)

        content_layout.addStretch()

    # ------------------------------------------------------------------
    # データ更新・イベント
    # ------------------------------------------------------------------

    def set_dataset(self, dataset: Optional[Dataset]) -> None:
        """データセットを受け取る（None の場合は未分割状態を表示）。"""
        self._dataset = dataset
        if dataset is not None:
            self._no_data_label.setVisible(False)
            self._content.setVisible(True)
            self._open_folder_btn.setVisible(False)
        else:
            self._no_data_label.setVisible(True)
            self._content.setVisible(False)

    def _browse_output(self) -> None:
        """出力先フォルダを選択する。"""
        path = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択")
        if path:
            self._output_edit.setText(path)

    def _start_export(self) -> None:
        """エクスポート Worker を起動する。"""
        output_dir = self._output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "入力エラー", "出力先フォルダを選択してください。"
            )
            return
        if self._dataset is None:
            return

        self._output_dir = output_dir
        fmt_key = "yolo" if self._yolo_radio.isChecked() else "coco"
        copy_images = self._copy_images_check.isChecked()

        self._export_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._open_folder_btn.setVisible(False)

        self._worker = ExportWorker(
            self._dataset, output_dir, fmt_key, copy_images
        )
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.status_message.connect(self.status_updated)
        self._worker.finished.connect(self._on_export_finished)
        self._worker.error.connect(self._on_export_error)
        self._worker.start()

    def _cancel_export(self) -> None:
        """エクスポートをキャンセルする。"""
        if self._worker:
            self._worker.cancel()
        self._reset_ui()

    def _on_export_finished(self, _: object) -> None:
        """エクスポート完了時の処理。"""
        self._reset_ui()
        self._open_folder_btn.setVisible(True)
        QMessageBox.information(
            self,
            "完了",
            f"エクスポートが完了しました。\n\n出力先: {self._output_dir}",
        )

    def _on_export_error(self, message: str) -> None:
        """エクスポートエラー時の処理。"""
        self._reset_ui()
        QMessageBox.critical(
            self,
            "エクスポートエラー",
            f"エクスポートに失敗しました。\n\n{message}",
        )

    def _reset_ui(self) -> None:
        """UI 状態をリセットする。"""
        self._export_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setVisible(False)

    def _open_output_folder(self) -> None:
        """出力フォルダをファイルマネージャーで開く（Windows: os.startfile）。"""
        if not self._output_dir:
            return
        if sys.platform == "win32":
            # os.startfile は Windows 専用で、エクスプローラーで確実にフォルダを開く
            os.startfile(self._output_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", self._output_dir])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", self._output_dir])
