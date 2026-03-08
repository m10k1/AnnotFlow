"""
ClipWorker（QThread）

CLIPベクトル化・クラスタリングをバックグラウンドスレッドで実行するWorkerクラス。
Phase 3 で実装する。
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class ClipWorker(QThread):
    """
    CLIP ベクトル化・クラスタリングを非同期で実行するWorker。

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

    def __init__(self, dataset: object, n_clusters: int, seed: int = 42) -> None:
        super().__init__()
        self._dataset = dataset
        self._n_clusters = n_clusters
        self._seed = seed
        self._cancelled = False

    def run(self) -> None:
        """
        バックグラウンドで CLIP 処理を実行する（Phase 3 で実装）。

        完了時は全 ImageRecord.cluster_id に値を格納してから
        finished シグナルで Dataset を送出する。
        """
        raise NotImplementedError("Phase 3 で実装する")

    def cancel(self) -> None:
        """キャンセルフラグを立てる。"""
        self._cancelled = True
