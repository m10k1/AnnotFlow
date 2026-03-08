"""
ClipWorker（QThread）

CLIPベクトル化・クラスタリングをバックグラウンドスレッドで実行するWorkerクラス。
OOM（Out of Memory）発生時はバッチサイズを自動的に半減して再試行する。
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.dataset import Dataset


class ClipWorker(QThread):
    """
    CLIP ベクトル化・クラスタリングを非同期で実行するWorker。

    OOM 発生時はバッチサイズを自動的に半減して再試行する。
    処理完了時には各 ImageRecord.embedding / cluster_id に値を格納してから
    finished シグナルを発火する（仕様書の要求）。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時に cluster_id が設定された Dataset を渡す
        error (str): エラー時にエラーメッセージを渡す
    """

    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        dataset: Dataset,
        n_clusters: int,
        seed: int = 42,
        initial_batch_size: int = 32,
    ) -> None:
        """
        Args:
            dataset: 処理対象の Dataset
            n_clusters: K-Means のクラスタ数
            seed: 乱数シード
            initial_batch_size: 最初のバッチサイズ（OOM時に自動半減する）
        """
        super().__init__()
        self._dataset = dataset
        self._n_clusters = n_clusters
        self._seed = seed
        self._initial_batch_size = initial_batch_size
        self._cancelled = False

    def run(self) -> None:
        """
        バックグラウンドで CLIP 処理を実行する。

        処理フロー:
        1. CLIPモデルをロード
        2. 全画像をバッチでベクトル化（OOM時はバッチサイズを半減して再試行）
        3. K-Means クラスタリング
        4. 各 ImageRecord に embedding と cluster_id を格納
        5. finished シグナルで Dataset を送出

        OOM リトライロジック:
        - RuntimeError で "out of memory" が含まれる場合のみ再試行
        - バッチサイズが 1 以下になった場合は再試行せずエラーを送出
        """
        try:
            from core.clip_clusterer import ClipClusterer

            self.status_message.emit("CLIPモデルを読み込み中...")
            self.progress.emit(0)

            clusterer = ClipClusterer(n_clusters=self._n_clusters, seed=self._seed)
            clusterer.load_model()

            if self._cancelled:
                return

            file_paths = [r.file_path for r in self._dataset.records]
            batch_size = self._initial_batch_size
            embeddings = None

            # OOM が発生した場合、バッチサイズを半減して再試行する
            while batch_size >= 1:
                if self._cancelled:
                    return

                try:
                    self.status_message.emit(
                        f"画像をベクトル化中... (バッチサイズ: {batch_size})"
                    )

                    def _progress_cb(pct: int) -> None:
                        # ベクトル化フェーズは全体進捗の 10%〜80% に割り当てる
                        self.progress.emit(10 + int(pct * 0.7))

                    embeddings = clusterer.encode_images(
                        file_paths,
                        batch_size=batch_size,
                        progress_callback=_progress_cb,
                    )
                    break  # 成功したらループを抜ける

                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch_size > 1:
                        batch_size //= 2
                        self.status_message.emit(
                            f"メモリ不足が発生しました。"
                            f"バッチサイズを {batch_size} に減らして再試行します..."
                        )
                    else:
                        # OOM 以外の RuntimeError またはバッチサイズが 1 以下
                        raise

            if embeddings is None:
                raise RuntimeError(
                    "画像のベクトル化に失敗しました（メモリ不足が解消されませんでした）。"
                )

            if self._cancelled:
                return

            self.status_message.emit("K-Means クラスタリングを実行中...")
            self.progress.emit(85)

            cluster_labels = clusterer.cluster(embeddings)
            self.progress.emit(95)

            # 仕様書の要求: 完了時に全 ImageRecord.cluster_id へ値を格納してから finished を送出
            clusterer.apply_to_dataset(self._dataset, embeddings, cluster_labels)

            self.progress.emit(100)
            self.status_message.emit(
                f"CLIPクラスタリング完了: {len(set(cluster_labels.tolist()))} クラスタ "
                f"/ {len(file_paths)} 画像"
            )
            self.finished.emit(self._dataset)

        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self) -> None:
        """キャンセルフラグを立てる（強制終了はしない）。"""
        self._cancelled = True
