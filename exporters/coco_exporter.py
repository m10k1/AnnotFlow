"""
COCO JSON フォーマットのエクスポーター

split 別に train.json / val.json / test.json を出力する。
segmentation フィールドは保持していた値をそのまま出力する（変換・加工なし）。
"""
from __future__ import annotations

from pathlib import Path

from core.dataset import Dataset
from exporters.base_exporter import BaseExporter


class CocoExporter(BaseExporter):
    """COCO JSON フォーマットのエクスポーター。"""

    def export(
        self,
        dataset: Dataset,
        output_dir: Path,
        copy_images: bool = True,
    ) -> None:
        """
        Dataset を COCO JSON 形式でエクスポートする。

        Args:
            dataset: エクスポート対象のデータセット（split が設定済みであること）
            output_dir: 出力先ディレクトリ
            copy_images: 画像ファイルもコピーするか否か
        """
        raise NotImplementedError("Phase 2 で実装する")
