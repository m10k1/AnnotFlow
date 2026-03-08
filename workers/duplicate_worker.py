"""
DuplicateCheckWorker（QThread）

MD5ハッシュによる重複画像検出をバックグラウンドスレッドで実行するWorkerクラス。
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class DuplicateCheckWorker(QThread):
    """
    重複画像検出を非同期で実行するWorker。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時に重複グループのリストを渡す
        error (str): エラー時にエラーメッセージを渡す
    """
    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dataset: object) -> None:
        super().__init__()
        self._dataset = dataset
        self._cancelled = False

    def run(self) -> None:
        """バックグラウンドで重複検出処理を実行する（Phase 1 で実装）。"""
        raise NotImplementedError("Phase 1 で実装する")

    def cancel(self) -> None:
        """キャンセルフラグを立てる。"""
        self._cancelled = True
