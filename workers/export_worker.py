"""
ExportWorker（QThread）

エクスポート処理をバックグラウンドスレッドで実行するWorkerクラス。
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class ExportWorker(QThread):
    """
    エクスポート処理を非同期で実行するWorker。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時のシグナル
        error (str): エラー時にエラーメッセージを渡す
    """
    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dataset: object, output_dir: str, fmt: str, copy_images: bool = True) -> None:
        super().__init__()
        self._dataset = dataset
        self._output_dir = output_dir
        self._fmt = fmt
        self._copy_images = copy_images
        self._cancelled = False

    def run(self) -> None:
        """バックグラウンドでエクスポート処理を実行する（Phase 1 で実装）。"""
        raise NotImplementedError("Phase 1 で実装する")

    def cancel(self) -> None:
        """キャンセルフラグを立てる。"""
        self._cancelled = True
