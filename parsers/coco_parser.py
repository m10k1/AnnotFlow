"""
COCO JSON フォーマットのパーサー

COCO JSON（images / annotations / categories 構造）を読み込んで
内部データモデル（Dataset）に変換する。

座標変換: COCO の [x, y, w, h]（絶対座標）→ BoundingBox [x_min, y_min, x_max, y_max]
クラスID: category_id は無視し、name を使って ClassMap に登録・0始まりで再採番
"""
from __future__ import annotations

from pathlib import Path

from core.dataset import Dataset
from parsers.base_parser import BaseParser


class CocoParser(BaseParser):
    """COCO JSON フォーマットのパーサー。"""

    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        COCO JSON を読み込んで Dataset に変換する。

        Args:
            annotation_path: COCO JSON ファイルのパス
            image_folder: 画像フォルダのパス

        Returns:
            (Dataset, skipped_records) のタプル
        """
        raise NotImplementedError("Phase 1 で実装する")
