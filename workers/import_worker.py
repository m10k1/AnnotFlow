"""
ImportWorker（QThread）

インポート処理をバックグラウンドスレッドで実行するWorkerクラス。
ファイルI/O・パース・画像サイズ取得をメインスレッドをブロックせずに処理する。
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class ImportWorker(QThread):
    """
    インポート処理を非同期で実行するWorker。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時に Dataset オブジェクトを渡す
        error (str): エラー時にエラーメッセージを渡す
        skipped (list): スキップしたレコード一覧を渡す
    """
    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    skipped = Signal(list)

    def __init__(self, file_path: str, image_folder: str, fmt: str) -> None:
        super().__init__()
        self._file_path = file_path
        self._image_folder = image_folder
        self._fmt = fmt
        self._cancelled = False     # キャンセルフラグ

    def run(self) -> None:
        """バックグラウンドでインポート処理を実行する（Phase 1 で実装）。"""
        raise NotImplementedError("Phase 1 で実装する")

    def cancel(self) -> None:
        """キャンセルフラグを立てる（強制終了しない）。"""
        self._cancelled = True
