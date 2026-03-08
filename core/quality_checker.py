"""
品質チェックロジック

- 重複画像の検出（MD5ハッシュ）
- 空ラベル（アノテーションなし画像）の検出
- クラスバランス統計の計算

PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

from core.dataset import Dataset


def find_duplicates(dataset: Dataset) -> list[list[str]]:
    """
    MD5ハッシュを使って完全一致する重複画像を検出する。

    Args:
        dataset: チェック対象のデータセット

    Returns:
        重複グループのリスト（各グループは重複している image_id のリスト）
    """
    raise NotImplementedError("Phase 1 で実装する")


def find_empty_annotations(dataset: Dataset) -> list[str]:
    """
    アノテーション数がゼロの画像の image_id リストを返す。

    Args:
        dataset: チェック対象のデータセット

    Returns:
        アノテーションが0件の ImageRecord の image_id リスト
    """
    raise NotImplementedError("Phase 1 で実装する")


def compute_class_distribution(dataset: Dataset) -> dict[str, int]:
    """
    クラスごとのアノテーション数を集計して返す。

    Args:
        dataset: 集計対象のデータセット

    Returns:
        {class_name: count} の辞書
    """
    raise NotImplementedError("Phase 1 で実装する")
