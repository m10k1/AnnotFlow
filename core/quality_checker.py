"""
品質チェックロジック

- 重複画像の検出（MD5ハッシュ）
- 空ラベル（アノテーションなし画像）の検出
- クラスバランス統計の計算

PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from core.dataset import Dataset


def find_duplicates(dataset: Dataset) -> list[list[str]]:
    """
    MD5ハッシュを使って完全一致する重複画像を検出する。

    存在しないファイルパスのレコードはスキップして処理を続行する。

    Args:
        dataset: チェック対象のデータセット

    Returns:
        重複グループのリスト（各グループは重複している image_id のリスト）。
        重複がなければ空のリストを返す。
    """
    hash_to_ids: dict[str, list[str]] = defaultdict(list)

    for record in dataset.records:
        try:
            with open(record.file_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            hash_to_ids[md5].append(record.image_id)
        except (OSError, IOError):
            # ファイルが存在しない場合はスキップ
            continue

    # 2件以上の場合のみ重複グループとして返す
    return [ids for ids in hash_to_ids.values() if len(ids) > 1]


def find_empty_annotations(dataset: Dataset) -> list[str]:
    """
    アノテーション数がゼロの画像の image_id リストを返す。

    Args:
        dataset: チェック対象のデータセット

    Returns:
        アノテーションが0件の ImageRecord の image_id リスト
    """
    return [
        record.image_id
        for record in dataset.records
        if not record.annotations
    ]


def compute_annotation_count_per_image(dataset: Dataset) -> dict[str, int]:
    """
    画像ごとのアノテーション数を集計して返す。

    Args:
        dataset: 集計対象のデータセット

    Returns:
        {image_id: annotation_count} の辞書。レコードがなければ空の辞書を返す。
    """
    return {record.image_id: len(record.annotations) for record in dataset.records}


def compute_class_distribution(dataset: Dataset) -> dict[str, int]:
    """
    クラスごとのアノテーション数を集計して返す。

    Args:
        dataset: 集計対象のデータセット

    Returns:
        {class_name: count} の辞書。アノテーションがなければ空の辞書を返す。
    """
    distribution: dict[str, int] = {}

    for record in dataset.records:
        for ann in record.annotations:
            distribution[ann.class_name] = distribution.get(ann.class_name, 0) + 1

    return distribution
