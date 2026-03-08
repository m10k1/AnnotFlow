"""
ImportWorker（QThread）

インポート処理をバックグラウンドスレッドで実行するWorkerクラス。
ファイルI/O・パース・画像サイズ取得をメインスレッドをブロックせずに処理する。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from parsers.coco_parser import CocoParser
from parsers.label_studio_parser import LabelStudioParser
from parsers.yolo_parser import YoloParser

_PARSERS = {
    "coco": CocoParser,
    "yolo": YoloParser,
    "label_studio": LabelStudioParser,
}


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
        self._cancelled = False

    def run(self) -> None:
        """バックグラウンドでインポート処理を実行する。"""
        try:
            parser_class = _PARSERS.get(self._fmt)
            if parser_class is None:
                self.error.emit(f"未対応のフォーマット: {self._fmt}")
                return

            self.status_message.emit(f"読み込み中: {self._file_path}")
            self.progress.emit(10)

            if self._cancelled:
                return

            parser = parser_class()
            dataset, skipped_records = parser.parse(
                Path(self._file_path),
                Path(self._image_folder),
            )

            self.progress.emit(90)

            if self._cancelled:
                return

            if skipped_records:
                self.skipped.emit(skipped_records)

            self.progress.emit(100)
            self.status_message.emit(
                f"インポート完了: {len(dataset.records)} 件"
                + (f"（スキップ: {len(skipped_records)} 件）" if skipped_records else "")
            )
            self.finished.emit(dataset)

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def cancel(self) -> None:
        """キャンセルフラグを立てる（強制終了しない）。"""
        self._cancelled = True
