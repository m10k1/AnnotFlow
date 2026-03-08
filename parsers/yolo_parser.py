"""
YOLO フォーマットのパーサー

YOLO形式（.txt × 画像数 + classes.txt または data.yaml）を読み込んで
内部データモデル（Dataset）に変換する。

座標変換: YOLO の相対座標 [cx, cy, w, h]（0.0〜1.0）→ BoundingBox [x_min, y_min, x_max, y_max]
クラスID: data.yaml または classes.txt の行順をそのまま 0始まりで ClassMap に登録
"""
from __future__ import annotations

from pathlib import Path

from core.dataset import Dataset
from parsers.base_parser import BaseParser


class YoloParser(BaseParser):
    """YOLO フォーマットのパーサー。"""

    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        YOLO フォーマットを読み込んで Dataset に変換する。

        Args:
            annotation_path: classes.txt または data.yaml のパス（アノテーションフォルダを指す）
            image_folder: 画像フォルダのパス

        Returns:
            (Dataset, skipped_records) のタプル
        """
        raise NotImplementedError("Phase 1 で実装する")
