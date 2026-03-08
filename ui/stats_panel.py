"""
品質チェック・統計パネル

以下の機能をタブ形式で提供する。
  1. クラスバランスの棒グラフ（matplotlib 埋め込み）
  2. 重複画像の検出（DuplicateCheckWorker）
  3. 空ラベル（アノテーションなし）画像の検出

PySide6 の UI クラスのみを使用し、core/ モジュールは結果表示のみに使用する。
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.dataset import Dataset
from core.quality_checker import (
    compute_annotation_count_per_image,
    compute_class_distribution,
    find_empty_annotations,
)
from workers.duplicate_worker import DuplicateCheckWorker

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except Exception:
    _MPL_OK = False


class StatsPanel(QWidget):
    """品質チェック・統計パネル。"""

    status_updated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._dup_worker: Optional[DuplicateCheckWorker] = None
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

        self._content_tabs = QTabWidget()
        self._content_tabs.setVisible(False)
        self._content_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout.addWidget(self._content_tabs)

        self._build_class_tab()
        self._build_dup_tab()
        self._build_empty_tab()
        self._build_annotation_count_tab()

    def _build_class_tab(self) -> None:
        """クラスバランスタブを構築する。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        if _MPL_OK:
            self._figure = Figure(facecolor="#1e1e2e")
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self._canvas)
        else:
            lbl = QLabel("matplotlib が利用できないため、グラフを表示できません。")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            self._figure = None
            self._canvas = None

        self._content_tabs.addTab(tab, "クラスバランス")

    def _build_dup_tab(self) -> None:
        """重複画像タブを構築する。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._dup_check_btn = QPushButton("重複チェック実行")
        self._dup_check_btn.setProperty("role", "primary")
        self._dup_check_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._dup_check_btn.clicked.connect(self._start_dup_check)
        self._dup_cancel_btn = QPushButton("キャンセル")
        self._dup_cancel_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._dup_cancel_btn.setEnabled(False)
        self._dup_cancel_btn.clicked.connect(self._cancel_dup_check)
        btn_row.addWidget(self._dup_check_btn)
        btn_row.addWidget(self._dup_cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._dup_progress = QProgressBar()
        self._dup_progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._dup_progress.setVisible(False)
        layout.addWidget(self._dup_progress)

        self._dup_result_label = QLabel(
            "「重複チェック実行」ボタンを押してください。"
        )
        self._dup_result_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._dup_result_label)

        self._dup_list = QListWidget()
        self._dup_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._dup_list)

        self._content_tabs.addTab(tab, "重複画像")

    def _build_empty_tab(self) -> None:
        """空ラベルタブを構築する。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._empty_result_label = QLabel("")
        self._empty_result_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._empty_result_label)

        self._empty_list = QListWidget()
        self._empty_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._empty_list)

        self._content_tabs.addTab(tab, "空ラベル画像")

    def _build_annotation_count_tab(self) -> None:
        """アノテーション数ヒストグラムタブを構築する。"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        if _MPL_OK:
            self._hist_figure = Figure(facecolor="#1e1e2e")
            self._hist_canvas = FigureCanvas(self._hist_figure)
            self._hist_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(self._hist_canvas)
        else:
            lbl = QLabel("matplotlib が利用できないため、グラフを表示できません。")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            self._hist_figure = None
            self._hist_canvas = None

        # 外れ値リスト（極端に多い・少ない画像）
        self._outlier_label = QLabel("")
        self._outlier_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._outlier_label)

        self._outlier_list = QListWidget()
        self._outlier_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._outlier_list)

        self._content_tabs.addTab(tab, "アノテーション数")

    # ------------------------------------------------------------------
    # データ更新
    # ------------------------------------------------------------------

    def set_dataset(self, dataset: Dataset) -> None:
        """データセットを受け取り、各タブの統計を更新する。"""
        self._dataset = dataset
        self._no_data_label.setVisible(False)
        self._content_tabs.setVisible(True)
        self._update_class_distribution()
        self._update_empty_annotations()
        self._update_annotation_count_histogram()

    def _update_class_distribution(self) -> None:
        """クラスバランスの棒グラフを更新する。"""
        if self._dataset is None or not _MPL_OK:
            return

        dist = compute_class_distribution(self._dataset)
        self._figure.clear()
        ax = self._figure.add_subplot(111, facecolor="#181825")

        if not dist:
            ax.text(
                0.5, 0.5, "アノテーションがありません",
                ha="center", va="center",
                transform=ax.transAxes,
                color="#a6adc8",
            )
            self._canvas.draw()
            return

        names = list(dist.keys())
        counts = list(dist.values())
        bars = ax.bar(names, counts, color="#89b4fa", width=0.6)
        ax.set_xlabel("クラス名", color="#cdd6f4", labelpad=8)
        ax.set_ylabel("アノテーション数", color="#cdd6f4", labelpad=8)
        ax.set_title("クラスバランス", color="#cdd6f4", pad=12)
        ax.tick_params(colors="#cdd6f4", which="both")
        for spine in ax.spines.values():
            spine.set_color("#45475a")

        max_count = max(counts)
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.01,
                str(count),
                ha="center",
                va="bottom",
                color="#cdd6f4",
                fontsize=10,
            )

        self._figure.tight_layout()
        self._canvas.draw()

    def _update_empty_annotations(self) -> None:
        """空ラベル画像リストを更新する。"""
        if self._dataset is None:
            return

        empty_ids = find_empty_annotations(self._dataset)
        self._empty_list.clear()
        id_to_record = {r.image_id: r for r in self._dataset.records}

        if empty_ids:
            self._empty_result_label.setText(
                f"アノテーションなしの画像: {len(empty_ids)} 件"
            )
            self._empty_result_label.setProperty("state", "warning")
            for image_id in empty_ids:
                record = id_to_record.get(image_id)
                display = record.original_filename if record else image_id
                self._empty_list.addItem(display)
        else:
            self._empty_result_label.setText("空ラベル画像はありません。")
            self._empty_result_label.setProperty("state", "success")

        self._refresh_label_style(self._empty_result_label)

    def _update_annotation_count_histogram(self) -> None:
        """アノテーション数ヒストグラムと外れ値リストを更新する。"""
        if self._dataset is None or not _MPL_OK:
            return

        count_map = compute_annotation_count_per_image(self._dataset)
        if not count_map:
            return

        counts = list(count_map.values())

        # ヒストグラムを描画する
        self._hist_figure.clear()
        ax = self._hist_figure.add_subplot(111, facecolor="#181825")

        ax.hist(counts, bins=max(1, min(30, len(counts) // 2 + 1)),
                color="#89b4fa", edgecolor="#313244")
        ax.set_xlabel("アノテーション数", color="#cdd6f4", labelpad=8)
        ax.set_ylabel("画像数", color="#cdd6f4", labelpad=8)
        ax.set_title("画像ごとのアノテーション数分布", color="#cdd6f4", pad=12)
        ax.tick_params(colors="#cdd6f4", which="both")
        for spine in ax.spines.values():
            spine.set_color("#45475a")

        self._hist_figure.tight_layout()
        self._hist_canvas.draw()

        # 外れ値（IQR 法）を検出してリストアップする
        if len(counts) >= 4:
            import statistics
            q1 = statistics.quantiles(counts, n=4)[0]
            q3 = statistics.quantiles(counts, n=4)[2]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_ids = [
                img_id for img_id, cnt in count_map.items()
                if cnt < lower or cnt > upper
            ]
        else:
            outlier_ids = []

        self._outlier_list.clear()
        id_to_record = {r.image_id: r for r in self._dataset.records}

        if outlier_ids:
            self._outlier_label.setText(
                f"外れ値（アノテーション数が極端な画像）: {len(outlier_ids)} 件"
            )
            self._outlier_label.setProperty("state", "warning")
            for img_id in outlier_ids:
                record = id_to_record.get(img_id)
                filename = record.original_filename if record else img_id
                cnt = count_map.get(img_id, 0)
                self._outlier_list.addItem(f"{filename}  ({cnt} 件)")
        else:
            self._outlier_label.setText("外れ値は検出されませんでした。")
            self._outlier_label.setProperty("state", "success")

        self._refresh_label_style(self._outlier_label)

    # ------------------------------------------------------------------
    # 重複チェック Worker
    # ------------------------------------------------------------------

    def _start_dup_check(self) -> None:
        """重複チェック Worker を起動する。"""
        if self._dataset is None:
            return

        self._dup_check_btn.setEnabled(False)
        self._dup_cancel_btn.setEnabled(True)
        self._dup_progress.setValue(0)
        self._dup_progress.setVisible(True)
        self._dup_list.clear()
        self._dup_result_label.setText("検出中...")
        self._dup_result_label.setProperty("state", "")
        self._refresh_label_style(self._dup_result_label)

        self._dup_worker = DuplicateCheckWorker(self._dataset)
        self._dup_worker.progress.connect(self._dup_progress.setValue)
        self._dup_worker.status_message.connect(self.status_updated)
        self._dup_worker.finished.connect(self._on_dup_finished)
        self._dup_worker.error.connect(self._on_dup_error)
        self._dup_worker.start()

    def _cancel_dup_check(self) -> None:
        """重複チェックをキャンセルする。"""
        if self._dup_worker:
            self._dup_worker.cancel()
        self._reset_dup_ui()
        self._dup_result_label.setText("キャンセルされました。")

    def _on_dup_finished(self, groups: list[list[str]]) -> None:
        """重複チェック完了時の処理。"""
        self._reset_dup_ui()
        self._dup_list.clear()

        if not groups:
            self._dup_result_label.setText("重複画像は検出されませんでした。")
            self._dup_result_label.setProperty("state", "success")
        else:
            total_imgs = sum(len(g) for g in groups)
            self._dup_result_label.setText(
                f"重複グループ: {len(groups)} 件（合計 {total_imgs} 枚）"
            )
            self._dup_result_label.setProperty("state", "warning")
            id_to_record = {r.image_id: r for r in self._dataset.records}
            for i, group in enumerate(groups, 1):
                self._dup_list.addItem(f"\u2500\u2500 グループ {i} \u2500\u2500")
                for image_id in group:
                    record = id_to_record.get(image_id)
                    name = record.original_filename if record else image_id
                    self._dup_list.addItem(f"  {name}")

        self._refresh_label_style(self._dup_result_label)

    def _on_dup_error(self, message: str) -> None:
        """重複チェックエラー時の処理。"""
        self._reset_dup_ui()
        QMessageBox.critical(
            self, "エラー", f"重複チェック中にエラーが発生しました。\n{message}"
        )

    def _reset_dup_ui(self) -> None:
        """重複チェック UI をリセットする。"""
        self._dup_check_btn.setEnabled(True)
        self._dup_cancel_btn.setEnabled(False)
        self._dup_progress.setVisible(False)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_label_style(label: QLabel) -> None:
        """setProperty 後に QSS を再適用する。"""
        label.style().unpolish(label)
        label.style().polish(label)
