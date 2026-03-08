"""
YOLO フォーマットのエクスポーター

split 別に train/images/, train/labels/, val/images/, val/labels/,
test/images/, test/labels/ および data.yaml を出力する。
BBox のみを対象とし、segmentation は出力しない。
"""
from __future__ import annotations

from pathlib import Path

from core.dataset import Dataset
from exporters.base_exporter import BaseExporter


class YoloExporter(BaseExporter):
    """YOLO フォーマットのエクスポーター。"""

    def export(
        self,
        dataset: Dataset,
        output_dir: Path,
        copy_images: bool = True,
    ) -> None:
        """
        Dataset を YOLO 形式でエクスポートする。

        Args:
            dataset: エクスポート対象のデータセット（split が設定済みであること）
            output_dir: 出力先ディレクトリ
            copy_images: 画像ファイルもコピーするか否か
        """
        raise NotImplementedError("Phase 1 で実装する")
