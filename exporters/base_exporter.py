"""
エクスポーターの抽象基底クラス

各フォーマット（COCO JSON / YOLO）のエクスポーターはこのクラスを継承して実装する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.dataset import Dataset


class BaseExporter(ABC):
    """すべてのエクスポーターが実装すべき抽象基底クラス。"""

    @abstractmethod
    def export(
        self,
        dataset: Dataset,
        output_dir: Path,
        copy_images: bool = True,
    ) -> None:
        """
        Dataset を指定フォーマットでエクスポートする。

        Args:
            dataset: エクスポート対象のデータセット
            output_dir: 出力先ディレクトリ
            copy_images: 画像ファイルもコピーするか否か
        """
        ...
