"""
パーサーの抽象基底クラス

各フォーマット（COCO JSON / YOLO / Label Studio JSON）のパーサーは
このクラスを継承して実装する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.dataset import Dataset


class BaseParser(ABC):
    """
    すべてのパーサーが実装すべき抽象基底クラス。
    """

    @abstractmethod
    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        アノテーションファイルを読み込んで Dataset に変換する。

        Args:
            annotation_path: アノテーションファイルのパス
            image_folder: 画像ファイルが格納されているフォルダのパス

        Returns:
            (Dataset, skipped_records) のタプル。
            skipped_records は画像が見つからなかったレコードの情報リスト。
            例: [{"filename": "foo.jpg", "reason": "画像ファイルが見つかりません"}]
        """
        ...
