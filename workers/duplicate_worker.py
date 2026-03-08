"""
DuplicateCheckWorker（QThread）

MD5ハッシュによる重複画像検出をバックグラウンドスレッドで実行するWorkerクラス。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from PySide6.QtCore import QThread, Signal

from core.dataset import Dataset


class DuplicateCheckWorker(QThread):
    """
    重複画像検出を非同期で実行するWorker。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時に重複グループのリスト（list[list[str]]）を渡す
        error (str): エラー時にエラーメッセージを渡す
    """
    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self._dataset = dataset
        self._cancelled = False

    def run(self) -> None:
        """バックグラウンドで重複検出処理を実行する。"""
        try:
            self.status_message.emit("重複画像を検出しています...")
            total = len(self._dataset.records)
            hash_to_ids: dict[str, list[str]] = defaultdict(list)

            for i, record in enumerate(self._dataset.records):
                if self._cancelled:
                    return

                try:
                    with open(record.file_path, "rb") as f:
                        md5 = hashlib.md5(f.read()).hexdigest()
                    hash_to_ids[md5].append(record.image_id)
                except (OSError, IOError):
                    # ファイルが存在しない場合はスキップ
                    continue

                self.progress.emit(int((i + 1) / total * 100) if total else 100)

            groups = [ids for ids in hash_to_ids.values() if len(ids) > 1]
            self.finished.emit(groups)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        """キャンセルフラグを立てる（強制終了しない）。"""
        self._cancelled = True
