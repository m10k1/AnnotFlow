"""
分割パネル

train/val/test 分割の設定UIを提供する。
方式A（ランダム分割）と方式B（マルチラベル層化分割）を選択できる。
分割はメインスレッドで直接実行する（splitter.py は純粋な CPU 計算で短時間）。
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dataset import Dataset
from core.splitter import split_random, split_stratified


class SplitPanel(QWidget):
    """分割パネル。"""

    status_updated = Signal(str)
    # 分割済み Dataset を MainWindow へ渡すシグナル
    dataset_split = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """UIを構築する。"""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        self._no_data_label = QLabel(
            "先にインポートタブでデータセットを読み込んでください。"
        )
        self._no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_data_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout.addWidget(self._no_data_label)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout.addWidget(self._content)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # ── 分割方式 ──
        method_group = QGroupBox("分割方式")
        method_layout = QVBoxLayout()
        method_layout.setContentsMargins(16, 16, 16, 16)
        method_layout.setSpacing(10)
        self._radio_random = QRadioButton("方式A：ランダム分割")
        self._radio_random.setChecked(True)
        self._radio_stratified = QRadioButton(
            "方式B：クラス層化分割（MultilabelStratifiedShuffleSplit）"
        )
        self._btn_group = QButtonGroup()
        self._btn_group.addButton(self._radio_random, 0)
        self._btn_group.addButton(self._radio_stratified, 1)
        method_layout.addWidget(self._radio_random)
        method_layout.addWidget(self._radio_stratified)
        method_group.setLayout(method_layout)
        content_layout.addWidget(method_group)

        # ── 比率設定 ──
        ratio_group = QGroupBox("分割比率")
        ratio_layout = QVBoxLayout()
        ratio_layout.setContentsMargins(16, 16, 16, 16)
        ratio_layout.setSpacing(10)

        for label_text, attr, default in [
            ("Train:", "_train_spin", 0.7),
            ("Val:  ", "_val_spin", 0.2),
            ("Test: ", "_test_spin", 0.1),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(QLabel(label_text))
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(default)
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            spin.valueChanged.connect(self._update_ratio_label)
            row.addWidget(spin)
            setattr(self, attr, spin)
            ratio_layout.addLayout(row)

        self._ratio_sum_label = QLabel("合計: 1.00")
        self._ratio_sum_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        ratio_layout.addWidget(self._ratio_sum_label)
        ratio_group.setLayout(ratio_layout)
        content_layout.addWidget(ratio_group)

        # ── シード値 ──
        seed_group = QGroupBox("ランダムシード")
        seed_layout = QHBoxLayout()
        seed_layout.setContentsMargins(16, 16, 16, 16)
        seed_layout.setSpacing(10)
        seed_layout.addWidget(QLabel("シード値:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        seed_layout.addWidget(self._seed_spin)
        seed_group.setLayout(seed_layout)
        content_layout.addWidget(seed_group)

        # ── 実行ボタン ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._split_btn = QPushButton("分割実行")
        self._split_btn.setProperty("role", "primary")
        self._split_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._split_btn.clicked.connect(self._execute_split)
        btn_row.addWidget(self._split_btn)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        # ── 結果サマリー ──
        result_group = QGroupBox("分割結果")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(10)

        self._result_label = QLabel("")
        self._result_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        result_layout.addWidget(self._result_label)

        self._result_table = QTableWidget(0, 2)
        self._result_table.setHorizontalHeaderLabels(["Split", "件数"])
        self._result_table.horizontalHeader().setStretchLastSection(True)
        self._result_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        result_layout.addWidget(self._result_table)

        result_group.setLayout(result_layout)
        content_layout.addWidget(result_group)

    # ------------------------------------------------------------------
    # データ更新・イベント
    # ------------------------------------------------------------------

    def set_dataset(self, dataset: Dataset) -> None:
        """データセットを受け取り、表示を切り替える。"""
        self._dataset = dataset
        self._no_data_label.setVisible(False)
        self._content.setVisible(True)
        self._result_label.setText("")
        self._result_table.setRowCount(0)

    def _update_ratio_label(self) -> None:
        """合計比率を更新して警告表示する。"""
        total = (
            self._train_spin.value()
            + self._val_spin.value()
            + self._test_spin.value()
        )
        self._ratio_sum_label.setText(f"合計: {total:.2f}")
        if abs(total - 1.0) < 0.005:
            self._ratio_sum_label.setProperty("state", "success")
        else:
            self._ratio_sum_label.setProperty("state", "error")
        self._ratio_sum_label.style().unpolish(self._ratio_sum_label)
        self._ratio_sum_label.style().polish(self._ratio_sum_label)

    def _execute_split(self) -> None:
        """分割を実行する。"""
        if self._dataset is None:
            return

        train_r = self._train_spin.value()
        val_r = self._val_spin.value()
        test_r = self._test_spin.value()
        seed = self._seed_spin.value()

        total = train_r + val_r + test_r
        if abs(total - 1.0) > 0.005:
            QMessageBox.warning(
                self,
                "入力エラー",
                f"比率の合計が 1.0 ではありません（現在: {total:.2f}）。\n"
                "合計が 1.0 になるよう調整してください。",
            )
            return

        try:
            if self._radio_stratified.isChecked():
                result = split_stratified(self._dataset, train_r, val_r, test_r, seed)
                method = "クラス層化分割"
            else:
                result = split_random(self._dataset, train_r, val_r, test_r, seed)
                method = "ランダム分割"

            counts = {"train": 0, "val": 0, "test": 0}
            for r in result.records:
                if r.split in counts:
                    counts[r.split] += 1

            self._result_label.setText(f"{method} 完了")
            self._result_table.setRowCount(3)
            for row, (split_name, count) in enumerate(counts.items()):
                self._result_table.setItem(
                    row, 0, QTableWidgetItem(split_name)
                )
                self._result_table.setItem(
                    row, 1, QTableWidgetItem(str(count))
                )

            self.status_updated.emit(f"{method} 完了: {len(result.records)} 件")
            self.dataset_split.emit(result)

        except Exception as exc:
            QMessageBox.critical(
                self, "分割エラー", f"分割中にエラーが発生しました。\n{exc}"
            )
