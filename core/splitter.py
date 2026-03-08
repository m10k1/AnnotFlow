"""
train/val/test 分割ロジック

方式A：ランダム分割
方式B：マルチラベル層化分割（iterative-stratification を使用）
方式C：CLIPクラスタベース比例配分（データリーク防止対応）

PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

import random
from copy import copy
from dataclasses import replace

import numpy as np

from core.dataset import Annotation, Dataset, ImageRecord


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _assign_splits(
    records: list[ImageRecord],
    split_labels: list[str],
) -> list[ImageRecord]:
    """
    split_labels[i] を records[i].split に設定した新しいリストを返す。
    元のレコードは変更しない（浅いコピー）。
    """
    result = []
    for record, label in zip(records, split_labels):
        new_record = ImageRecord(
            image_id=record.image_id,
            original_filename=record.original_filename,
            file_path=record.file_path,
            width=record.width,
            height=record.height,
            annotations=record.annotations,
            split=label,
            embedding=record.embedding,
            cluster_id=record.cluster_id,
        )
        result.append(new_record)
    return result


def _build_multilabel_matrix(
    records: list[ImageRecord],
    num_classes: int,
) -> np.ndarray:
    """
    仕様書の build_multilabel_matrix 実装。
    各画像をマルチホットベクトル（クラス数次元の binary vector）に変換する。
    """
    matrix = np.zeros((len(records), num_classes), dtype=int)
    for i, record in enumerate(records):
        for ann in record.annotations:
            if 0 <= ann.class_id < num_classes:
                matrix[i, ann.class_id] = 1
    return matrix


# ---------------------------------------------------------------------------
# 方式A：ランダム分割
# ---------------------------------------------------------------------------

def split_random(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    方式A：ランダム分割。
    各 ImageRecord の split フィールドを "train" / "val" / "test" に設定して返す。

    Args:
        dataset: 分割対象のデータセット
        train_ratio: 訓練セットの比率（0.0〜1.0）
        val_ratio: 検証セットの比率（0.0〜1.0）
        test_ratio: テストセットの比率（0.0〜1.0）
        seed: 乱数シード（再現性のため）

    Returns:
        split が設定された Dataset（元のデータセットは変更しない）
    """
    records = list(dataset.records)
    rng = random.Random(seed)
    # インデックスをシャッフルして元レコードのIDを追跡する
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n = len(records)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # 端数を test に回す
    n_test = n - n_train - n_val

    split_labels: list[str] = [""] * n
    for rank, orig_idx in enumerate(indices):
        if rank < n_train:
            split_labels[orig_idx] = "train"
        elif rank < n_train + n_val:
            split_labels[orig_idx] = "val"
        else:
            if n_test > 0:
                split_labels[orig_idx] = "test"
            else:
                # test_ratio=0 の場合、あまりを val に回す
                split_labels[orig_idx] = "val"

    new_records = _assign_splits(records, split_labels)

    return Dataset(
        records=new_records,
        class_map=dataset.class_map,
        source_format=dataset.source_format,
    )


# ---------------------------------------------------------------------------
# 方式B：マルチラベル層化分割
# ---------------------------------------------------------------------------

def split_stratified(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    方式B：マルチラベル層化分割（MultilabelStratifiedShuffleSplit 使用）。
    iterative-stratification が使えない場合はランダム分割にフォールバックする。

    Args:
        dataset: 分割対象のデータセット
        train_ratio: 訓練セットの比率
        val_ratio: 検証セットの比率
        test_ratio: テストセットの比率
        seed: 乱数シード

    Returns:
        split が設定された Dataset
    """
    records = list(dataset.records)
    n = len(records)
    num_classes = len(dataset.class_map.id_to_name)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        X = np.arange(n)
        y = _build_multilabel_matrix(records, num_classes)

        split_labels: list[str] = ["train"] * n

        # test_ratio > 0 の場合: まず train+val と test に分割
        if test_ratio > 0:
            # train+val と test の分割
            test_split_ratio = test_ratio / (train_ratio + val_ratio + test_ratio)
            msss_test = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=test_split_ratio, random_state=seed
            )
            trainval_idx, test_idx = next(msss_test.split(X, y))

            for idx in test_idx:
                split_labels[idx] = "test"

            # 残り（trainval）を train と val に分割
            if len(trainval_idx) > 0 and val_ratio > 0:
                val_in_trainval = val_ratio / (train_ratio + val_ratio)
                X_tv = trainval_idx
                y_tv = y[trainval_idx]

                msss_val = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=val_in_trainval, random_state=seed
                )
                train_local_idx, val_local_idx = next(msss_val.split(X_tv, y_tv))

                for idx in trainval_idx[val_local_idx]:
                    split_labels[idx] = "val"
                # train_local_idx はそのまま "train" のまま
        else:
            # test_ratio=0: train と val のみ
            if val_ratio > 0:
                val_split_ratio = val_ratio / (train_ratio + val_ratio)
                msss_val = MultilabelStratifiedShuffleSplit(
                    n_splits=1, test_size=val_split_ratio, random_state=seed
                )
                train_idx, val_idx = next(msss_val.split(X, y))
                for idx in val_idx:
                    split_labels[idx] = "val"

    except ImportError:
        # iterative-stratification がインポートできない場合はランダム分割にフォールバック
        return split_random(dataset, train_ratio, val_ratio, test_ratio, seed=seed)

    new_records = _assign_splits(records, split_labels)

    return Dataset(
        records=new_records,
        class_map=dataset.class_map,
        source_format=dataset.source_format,
    )


# ---------------------------------------------------------------------------
# 方式C：CLIPクラスタベース比例配分（データリーク防止）
# ---------------------------------------------------------------------------

def split_random_with_clusters(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    方式C：CLIPクラスタベース比例配分分割（データリーク防止対応版）。

    仕様書「データリーク防止ルール」を厳守する。

    ✅ 正しい実装:
        各クラスタ内で random.shuffle してから、
        train / val / test に比例配分（例: 8:1:1）する。
        → 結果として各 split に全クラスタの代表画像が含まれる。

    ❌ 禁止事項:
        「同一クラスタを特定の split に固める」実装は絶対に禁止。

    cluster_id が None のレコードが 1 件でも含まれている場合は
    split_random() にフォールバックする。

    Args:
        dataset: 分割対象のデータセット（ImageRecord.cluster_id が設定済みであること）
        train_ratio: 訓練セットの比率（0.0〜1.0）
        val_ratio: 検証セットの比率（0.0〜1.0）
        test_ratio: テストセットの比率（0.0〜1.0）
        seed: 乱数シード（再現性のため）

    Returns:
        split が設定された Dataset（元のデータセットは変更しない）
    """
    records = list(dataset.records)

    # cluster_id が未設定のレコードがある場合はランダム分割にフォールバック
    if any(r.cluster_id is None for r in records):
        return split_random(dataset, train_ratio, val_ratio, test_ratio, seed=seed)

    # クラスタごとにインデックスをグルーピング
    clusters: dict[int, list[int]] = {}
    for i, record in enumerate(records):
        cid = record.cluster_id  # type: ignore[assignment]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(i)

    rng = random.Random(seed)
    split_labels: list[str] = [""] * len(records)

    # 各クラスタ内でシャッフルしてから比例配分する
    # ← データリーク防止の核心：クラスタ単位でsplitを割り当てない
    for cluster_indices in clusters.values():
        rng.shuffle(cluster_indices)
        n = len(cluster_indices)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        n_test = n - n_train - n_val

        for rank, orig_idx in enumerate(cluster_indices):
            if rank < n_train:
                split_labels[orig_idx] = "train"
            elif rank < n_train + n_val:
                split_labels[orig_idx] = "val"
            else:
                # test_ratio=0 の場合、余りを val に回す
                split_labels[orig_idx] = "test" if n_test > 0 else "val"

    new_records = _assign_splits(records, split_labels)

    return Dataset(
        records=new_records,
        class_map=dataset.class_map,
        source_format=dataset.source_format,
    )
