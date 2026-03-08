"""
CLIPベクトル化・クラスタリングロジック（Phase 3）

openai/clip-vit-base-patch32 モデルを使って画像をベクトル化し、
K-Meansクラスタリングを実行する。

PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

from core.dataset import Dataset


def compute_embeddings(dataset: Dataset, batch_size: int = 32) -> list[list[float]]:
    """
    全画像を CLIP でベクトル化して埋め込みベクトルのリストを返す。

    Args:
        dataset: ベクトル化対象のデータセット
        batch_size: バッチサイズ（メモリ不足時は自動的に半減する）

    Returns:
        各画像の埋め込みベクトルのリスト（dataset.records と同じ順序）
    """
    raise NotImplementedError("Phase 3 で実装する")


def cluster_embeddings(embeddings: list[list[float]], n_clusters: int, seed: int = 42) -> list[int]:
    """
    埋め込みベクトルに K-Means クラスタリングを適用してクラスタIDのリストを返す。

    Args:
        embeddings: 埋め込みベクトルのリスト
        n_clusters: クラスタ数
        seed: 乱数シード

    Returns:
        各画像のクラスタID（embeddings と同じ順序）
    """
    raise NotImplementedError("Phase 3 で実装する")
