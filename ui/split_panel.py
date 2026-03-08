"""
分割パネル

train/val/test 分割の設定UIを提供する。
方式A（ランダム分割）と方式B（マルチラベル層化分割）を選択できる。
Phase 3: CLIPクラスタリングボタン・スライダー・プログレスバーを追加。
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
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.dataset import Dataset
from core.splitter import split_random, split_random_with_clusters, split_stratified


class SplitPanel(QWidget):
    """分割パネル。"""

    status_updated = Signal(str)
    # 分割済み Dataset を MainWindow へ渡すシグナル
    dataset_split = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._clustered_dataset: Optional[Dataset] = None  # クラスタリング済みデータセット
        self._clip_worker = None  # ClipWorker のインスタンス（実行中に保持）
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

        # ── CLIPクラスタリング ──
        content_layout.addWidget(self._build_clip_group())

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
        self._radio_cluster = QRadioButton(
            "方式C：CLIPクラスタベース比例配分（データリーク防止対応）"
        )
        self._radio_cluster.setEnabled(False)  # クラスタリング完了後に有効化
        self._btn_group = QButtonGroup()
        self._btn_group.addButton(self._radio_random, 0)
        self._btn_group.addButton(self._radio_stratified, 1)
        self._btn_group.addButton(self._radio_cluster, 2)
        method_layout.addWidget(self._radio_random)
        method_layout.addWidget(self._radio_stratified)
        method_layout.addWidget(self._radio_cluster)
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

    def _build_clip_group(self) -> QGroupBox:
        """CLIPクラスタリングセクションを構築して返す。"""
        clip_group = QGroupBox("CLIPクラスタリング（Phase 3）")
        clip_layout = QVBoxLayout()
        clip_layout.setContentsMargins(16, 16, 16, 16)
        clip_layout.setSpacing(10)

        # クラスタ数スライダー
        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        slider_row.addWidget(QLabel("クラスタ数:"))
        self._cluster_slider = QSlider(Qt.Orientation.Horizontal)
        self._cluster_slider.setRange(2, 100)
        self._cluster_slider.setValue(10)
        self._cluster_slider.setTickInterval(10)
        self._cluster_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._cluster_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._cluster_slider.valueChanged.connect(self._on_cluster_slider_changed)
        slider_row.addWidget(self._cluster_slider)
        self._cluster_count_label = QLabel("10")
        self._cluster_count_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        slider_row.addWidget(self._cluster_count_label)
        clip_layout.addLayout(slider_row)

        # 実行ボタン行
        clip_btn_row = QHBoxLayout()
        clip_btn_row.setSpacing(8)
        self._clip_run_btn = QPushButton("CLIPクラスタリングを実行")
        self._clip_run_btn.setProperty("role", "primary")
        self._clip_run_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._clip_run_btn.clicked.connect(self._run_clip_clustering)
        clip_btn_row.addWidget(self._clip_run_btn)

        self._clip_cancel_btn = QPushButton("キャンセル")
        self._clip_cancel_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._clip_cancel_btn.setVisible(False)
        self._clip_cancel_btn.clicked.connect(self._cancel_clip_clustering)
        clip_btn_row.addWidget(self._clip_cancel_btn)

        self._clip_umap_btn = QPushButton("UMAP可視化")
        self._clip_umap_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._clip_umap_btn.setEnabled(False)
        self._clip_umap_btn.clicked.connect(self._show_umap)
        clip_btn_row.addWidget(self._clip_umap_btn)
        clip_btn_row.addStretch()
        clip_layout.addLayout(clip_btn_row)

        # プログレスバー
        self._clip_progress_bar = QProgressBar()
        self._clip_progress_bar.setRange(0, 100)
        self._clip_progress_bar.setValue(0)
        self._clip_progress_bar.setVisible(False)
        self._clip_progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        clip_layout.addWidget(self._clip_progress_bar)

        # ステータスラベル
        self._clip_status_label = QLabel("")
        self._clip_status_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        clip_layout.addWidget(self._clip_status_label)

        clip_group.setLayout(clip_layout)
        return clip_group

    # ------------------------------------------------------------------
    # データ更新・イベント
    # ------------------------------------------------------------------

    def set_dataset(self, dataset: Dataset) -> None:
        """データセットを受け取り、表示を切り替える。"""
        self._dataset = dataset
        self._clustered_dataset = None
        self._no_data_label.setVisible(False)
        self._content.setVisible(True)
        self._result_label.setText("")
        self._result_table.setRowCount(0)
        self._radio_cluster.setEnabled(False)
        self._clip_umap_btn.setEnabled(False)
        self._clip_status_label.setText("")

        # クラスタ数のデフォルトを画像数の10%に設定（仕様書通り）
        n = len(dataset.records)
        default_clusters = max(2, n // 10)
        self._cluster_slider.setMaximum(max(2, n))
        self._cluster_slider.setValue(min(default_clusters, max(2, n)))

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

    def _on_cluster_slider_changed(self, value: int) -> None:
        """スライダー値が変わったときにラベルを更新する。"""
        self._cluster_count_label.setText(str(value))

    # ------------------------------------------------------------------
    # CLIPクラスタリング
    # ------------------------------------------------------------------

    def _run_clip_clustering(self) -> None:
        """CLIPクラスタリングを実行する（ClipWorker をバックグラウンドで起動）。"""
        if self._dataset is None:
            return

        from workers.clip_worker import ClipWorker

        n_clusters = self._cluster_slider.value()
        seed = self._seed_spin.value()

        # UI を実行中の状態に切り替える
        self._clip_run_btn.setEnabled(False)
        self._clip_cancel_btn.setVisible(True)
        self._clip_progress_bar.setVisible(True)
        self._clip_progress_bar.setValue(0)
        self._clip_status_label.setText("クラスタリングを準備中...")
        self._clip_umap_btn.setEnabled(False)

        self._clip_worker = ClipWorker(
            dataset=self._dataset,
            n_clusters=n_clusters,
            seed=seed,
        )
        self._clip_worker.progress.connect(self._clip_progress_bar.setValue)
        self._clip_worker.status_message.connect(self._clip_status_label.setText)
        self._clip_worker.finished.connect(self._on_clip_finished)
        self._clip_worker.error.connect(self._on_clip_error)
        self._clip_worker.start()

    def _cancel_clip_clustering(self) -> None:
        """CLIPクラスタリングをキャンセルする。"""
        if self._clip_worker is not None:
            self._clip_worker.cancel()
        self._clip_status_label.setText("キャンセルしました。")
        self._reset_clip_ui()

    def _on_clip_finished(self, dataset: Dataset) -> None:
        """CLIPクラスタリング完了時の処理。"""
        self._clustered_dataset = dataset
        self._dataset = dataset  # クラスタ情報を含むデータセットで更新

        n_clusters_actual = len({r.cluster_id for r in dataset.records})
        self._clip_status_label.setText(
            f"クラスタリング完了: {n_clusters_actual} クラスタ / {len(dataset.records)} 画像"
        )
        self.status_updated.emit(
            f"CLIPクラスタリング完了: {n_clusters_actual} クラスタ"
        )

        # 方式Cのラジオボタンを有効化してデフォルト選択する
        self._radio_cluster.setEnabled(True)
        self._radio_cluster.setChecked(True)

        # UMAP可視化ボタンを有効化する
        self._clip_umap_btn.setEnabled(True)

        self._reset_clip_ui()

    def _on_clip_error(self, message: str) -> None:
        """CLIPクラスタリングエラー時の処理。"""
        QMessageBox.critical(
            self,
            "CLIPクラスタリングエラー",
            f"クラスタリング中にエラーが発生しました。\n{message}",
        )
        self._clip_status_label.setText("エラーが発生しました。")
        self._reset_clip_ui()

    def _reset_clip_ui(self) -> None:
        """CLIPクラスタリングUIを待機状態に戻す。"""
        self._clip_run_btn.setEnabled(True)
        self._clip_cancel_btn.setVisible(False)
        self._clip_progress_bar.setVisible(False)
        self._clip_worker = None

    def _show_umap(self) -> None:
        """UMAP可視化ダイアログを表示する。"""
        if self._clustered_dataset is None:
            return

        # embedding が設定されていることを確認する
        records_with_embedding = [
            r for r in self._clustered_dataset.records
            if r.embedding is not None and r.cluster_id is not None
        ]
        if not records_with_embedding:
            QMessageBox.warning(
                self,
                "UMAP可視化",
                "埋め込みベクトルが設定されていません。"
                "先にCLIPクラスタリングを実行してください。",
            )
            return

        try:
            from ui.umap_dialog import UmapDialog
            dialog = UmapDialog(self._clustered_dataset, parent=self)
            dialog.exec()
        except ImportError as exc:
            QMessageBox.warning(
                self,
                "UMAP可視化",
                f"UMAP可視化に必要なライブラリが見つかりません。\n{exc}\n\n"
                "次のコマンドでインストールしてください:\n  uv add umap-learn matplotlib",
            )

    # ------------------------------------------------------------------
    # 分割実行
    # ------------------------------------------------------------------

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
            if self._radio_cluster.isChecked():
                # 方式C：CLIPクラスタベース比例配分
                if self._clustered_dataset is None:
                    QMessageBox.warning(
                        self,
                        "クラスタリング未実行",
                        "方式Cを使用するには先にCLIPクラスタリングを実行してください。",
                    )
                    return
                result = split_random_with_clusters(
                    self._clustered_dataset, train_r, val_r, test_r, seed
                )
                method = "CLIPクラスタベース比例配分"
            elif self._radio_stratified.isChecked():
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
