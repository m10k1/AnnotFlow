"""
ExportWorker（QThread）

エクスポート処理をバックグラウンドスレッドで実行するWorkerクラス。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core.dataset import Dataset


class ExportWorker(QThread):
    """
    エクスポート処理を非同期で実行するWorker。

    Signals:
        progress (int): 進捗（0〜100）
        status_message (str): ステータスバーへのメッセージ
        finished (object): 完了時のシグナル（Noneを渡す）
        error (str): エラー時にエラーメッセージを渡す
    """
    progress = Signal(int)
    status_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        dataset: Dataset,
        output_dir: str,
        fmt: str,
        copy_images: bool = True,
    ) -> None:
        """
        Args:
            dataset: エクスポート対象のデータセット（split が設定済みであること）
            output_dir: 出力先ディレクトリパス
            fmt: エクスポートフォーマット（"yolo" / "coco"）
            copy_images: 画像ファイルもコピーするか否か
        """
        super().__init__()
        self._dataset = dataset
        self._output_dir = output_dir
        self._fmt = fmt
        self._copy_images = copy_images
        self._cancelled = False

    def run(self) -> None:
        """バックグラウンドでエクスポート処理を実行する。"""
        try:
            self.status_message.emit(f"{self._fmt.upper()} 形式でエクスポートしています...")
            self.progress.emit(0)

            output_dir = Path(self._output_dir)

            if self._fmt == "yolo":
                from exporters.yolo_exporter import YoloExporter
                exporter = YoloExporter()
                exporter.export(self._dataset, output_dir, self._copy_images)

            elif self._fmt == "coco":
                from exporters.coco_exporter import CocoExporter
                exporter = CocoExporter()
                exporter.export(self._dataset, output_dir, self._copy_images)

            else:
                self.error.emit(f"未対応のフォーマット: {self._fmt!r}")
                return

            if self._cancelled:
                return

            self.progress.emit(100)
            self.status_message.emit("エクスポートが完了しました。")
            self.finished.emit(None)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        """キャンセルフラグを立てる（強制終了しない）。"""
        self._cancelled = True
