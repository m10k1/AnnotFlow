"""
UMAPダイアログ

CLIPクラスタリング結果を2D散布図（UMAP）で可視化するダイアログ。
matplotlib を PySide6 に埋め込んで表示する。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from core.dataset import Dataset


class UmapDialog(QDialog):
    """
    CLIPクラスタリング結果をUMAP 2D散布図で表示するダイアログ。

    依存ライブラリ（実行時にインポート）:
    - umap-learn: UMAP次元削減
    - matplotlib: グラフ描画
    """

    def __init__(self, dataset: Dataset, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("UMAP可視化 - CLIPクラスタリング結果")
        self.resize(800, 600)
        self._dataset = dataset
        self._setup_ui()

    def _setup_ui(self) -> None:
        """UIを構築する。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # タイトルラベル
        n_records = len(self._dataset.records)
        cluster_ids = {r.cluster_id for r in self._dataset.records if r.cluster_id is not None}
        title_label = QLabel(
            f"CLIPクラスタリング結果: {n_records} 画像 / {len(cluster_ids)} クラスタ"
        )
        title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(title_label)

        # matplotlibキャンバスを埋め込む
        try:
            canvas = self._build_umap_canvas()
            layout.addWidget(canvas)
        except Exception as exc:
            error_label = QLabel(f"UMAP可視化に失敗しました:\n{exc}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout.addWidget(error_label)

        # 閉じるボタン
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        close_btn = QPushButton("閉じる")
        close_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _build_umap_canvas(self):
        """
        UMAP次元削減を実行して matplotlib キャンバスを返す。

        Returns:
            FigureCanvasQTAgg: PySide6 に埋め込み可能な matplotlib キャンバス

        Raises:
            ImportError: umap-learn または matplotlib がインストールされていない場合
            ValueError: 埋め込みベクトルが設定されていない場合
        """
        import umap  # type: ignore
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        # embedding が設定されているレコードを収集する
        records_with_emb = [
            r for r in self._dataset.records
            if r.embedding is not None and r.cluster_id is not None
        ]
        if not records_with_emb:
            raise ValueError("埋め込みベクトルが設定されていません。")

        embeddings = np.array([r.embedding for r in records_with_emb], dtype=np.float32)
        cluster_ids = [r.cluster_id for r in records_with_emb]  # type: ignore[misc]

        # UMAP で2次元に削減する
        n_neighbors = min(15, len(records_with_emb) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)  # shape=(n, 2)

        # matplotlib 散布図を描画する
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#1e1e2e")

        # クラスタごとに色分けして描画
        unique_clusters = sorted(set(cluster_ids))
        cmap = plt.get_cmap("tab20")
        for cid in unique_clusters:
            mask = [i for i, c in enumerate(cluster_ids) if c == cid]
            ax.scatter(
                coords_2d[mask, 0],
                coords_2d[mask, 1],
                c=[cmap(cid % 20)],
                label=f"クラスタ {cid}",
                s=15,
                alpha=0.7,
                edgecolors="none",
            )

        ax.set_title(
            f"CLIPクラスタリング UMAP可視化 ({len(unique_clusters)} クラスタ)",
            color="#cdd6f4",
            fontsize=11,
        )
        ax.set_xlabel("UMAP次元1", color="#cdd6f4")
        ax.set_ylabel("UMAP次元2", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")

        # 凡例（クラスタ数が多い場合はスキップ）
        if len(unique_clusters) <= 20:
            legend = ax.legend(
                loc="upper right",
                fontsize=7,
                framealpha=0.3,
                labelcolor="#cdd6f4",
                facecolor="#313244",
            )

        fig.tight_layout()

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return canvas
