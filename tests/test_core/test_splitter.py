"""
分割ロジックのテスト（Phase 1: 実装）

- split_random: 方式A ランダム分割
- split_stratified: 方式B マルチラベル層化分割
"""
from __future__ import annotations

import pytest

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from core.splitter import split_random, split_random_with_clusters, split_stratified


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_multilabel_dataset(n: int = 30) -> Dataset:
    """
    マルチラベル分割テスト用 Dataset（n件）。
    全クラス（dog=0, cat=1, car=2）が複数枚に存在する。
    """
    class_map = ClassMap.from_names(["dog", "cat", "car"])
    records = []
    for i in range(n):
        pattern = i % 3
        if pattern == 0:
            anns = [
                Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
                Annotation(1, "cat", BoundingBox(200, 50, 350, 200)),
            ]
        elif pattern == 1:
            anns = [
                Annotation(1, "cat", BoundingBox(10, 20, 100, 150)),
                Annotation(2, "car", BoundingBox(200, 50, 400, 300)),
            ]
        else:
            anns = [
                Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
                Annotation(2, "car", BoundingBox(200, 50, 400, 300)),
            ]
        records.append(ImageRecord(
            image_id=str(i).zfill(3),
            original_filename=f"image_{str(i).zfill(3)}.jpg",
            file_path=f"/tmp/images/image_{str(i).zfill(3)}.jpg",
            width=640,
            height=480,
            annotations=anns,
        ))
    return Dataset(records=records, class_map=class_map, source_format="coco")


# ---------------------------------------------------------------------------
# split_random のテスト
# ---------------------------------------------------------------------------

def test_split_ratio_sum():
    """train+val+testの件数の合計が元のデータセット件数と一致すること。"""
    dataset = _make_multilabel_dataset(30)
    result = split_random(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42)

    total = len(result.records)
    assert total == 30

    counts = {"train": 0, "val": 0, "test": 0}
    for r in result.records:
        assert r.split in ("train", "val", "test"), f"無効な split 値: {r.split}"
        counts[r.split] += 1

    assert counts["train"] + counts["val"] + counts["test"] == 30


def test_split_reproducibility():
    """同一シード値で2回分割した結果が一致すること（再現性の確認）。"""
    dataset = _make_multilabel_dataset(30)
    result1 = split_random(dataset, 0.7, 0.2, 0.1, seed=99)
    result2 = split_random(dataset, 0.7, 0.2, 0.1, seed=99)

    splits1 = {r.image_id: r.split for r in result1.records}
    splits2 = {r.image_id: r.split for r in result2.records}
    assert splits1 == splits2


def test_split_no_record_duplicated_across_splits():
    """同一image_idがtrain/val/testに重複して割り当てられないこと。"""
    dataset = _make_multilabel_dataset(30)
    result = split_random(dataset, 0.7, 0.2, 0.1, seed=42)

    # image_id は一意であることを確認
    image_ids = [r.image_id for r in result.records]
    assert len(image_ids) == len(set(image_ids)), "同一 image_id が複数存在する"


def test_split_random_assigns_all_splits():
    """ランダム分割でtrain/val/testすべてに少なくとも1件割り当てられること（30件の場合）。"""
    dataset = _make_multilabel_dataset(30)
    result = split_random(dataset, 0.7, 0.2, 0.1, seed=42)

    splits = {r.split for r in result.records}
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits


def test_split_without_test_set():
    """test_ratio=0.0 の場合、testに割り当てられる画像がないこと。"""
    dataset = _make_multilabel_dataset(20)
    result = split_random(dataset, 0.8, 0.2, 0.0, seed=42)

    test_records = [r for r in result.records if r.split == "test"]
    assert len(test_records) == 0


def test_split_ratio_approximately_correct():
    """
    分割比率が近似的に正しいこと（100件で train≈70, val≈20, test≈10）。
    ±10件の誤差を許容する。
    """
    dataset = _make_multilabel_dataset(100)
    result = split_random(dataset, 0.7, 0.2, 0.1, seed=42)

    counts = {"train": 0, "val": 0, "test": 0}
    for r in result.records:
        counts[r.split] += 1

    assert abs(counts["train"] - 70) <= 10
    assert abs(counts["val"] - 20) <= 10
    assert abs(counts["test"] - 10) <= 10


# ---------------------------------------------------------------------------
# split_stratified のテスト
# ---------------------------------------------------------------------------

def test_stratified_split_multilabel():
    """マルチラベル画像（1枚に複数クラス混在）に対して層化分割がエラーなく動作すること。"""
    dataset = _make_multilabel_dataset(30)
    # エラーなく完了することを確認（ValueError 等が出ないこと）
    result = split_stratified(dataset, 0.7, 0.2, 0.1, seed=42)

    total = len(result.records)
    assert total == 30

    for r in result.records:
        assert r.split in ("train", "val", "test")


def test_stratified_split_ratio_sum():
    """層化分割でも合計件数が元のデータセット件数と一致すること。"""
    dataset = _make_multilabel_dataset(30)
    result = split_stratified(dataset, 0.7, 0.2, 0.1, seed=42)

    assert len(result.records) == 30
    counts = {"train": 0, "val": 0, "test": 0}
    for r in result.records:
        counts[r.split] += 1
    assert sum(counts.values()) == 30


def test_stratified_split_no_record_duplicated():
    """層化分割でも同一image_idが複数のsplitに重複しないこと。"""
    dataset = _make_multilabel_dataset(30)
    result = split_stratified(dataset, 0.7, 0.2, 0.1, seed=42)

    image_ids = [r.image_id for r in result.records]
    assert len(image_ids) == len(set(image_ids))


def test_stratified_split_reproducibility():
    """同一シード値で層化分割した結果が一致すること。"""
    dataset = _make_multilabel_dataset(30)
    result1 = split_stratified(dataset, 0.7, 0.2, 0.1, seed=7)
    result2 = split_stratified(dataset, 0.7, 0.2, 0.1, seed=7)

    splits1 = {r.image_id: r.split for r in result1.records}
    splits2 = {r.image_id: r.split for r in result2.records}
    assert splits1 == splits2


# ---------------------------------------------------------------------------
# split_random_with_clusters のテスト（方式C：CLIPクラスタベース比例配分）
# ---------------------------------------------------------------------------

def _make_clustered_dataset(n_clusters: int, per_cluster: int) -> Dataset:
    """
    各クラスタに per_cluster 件ずつ含む Dataset を生成する。
    cluster_id が設定済みの ImageRecord を持つ。
    """
    class_map = ClassMap.from_names(["dog"])
    records = []
    for cid in range(n_clusters):
        for j in range(per_cluster):
            idx = cid * per_cluster + j
            records.append(ImageRecord(
                image_id=str(idx),
                original_filename=f"img_{idx:03d}.jpg",
                file_path=f"/tmp/img_{idx:03d}.jpg",
                width=640,
                height=480,
                annotations=[],
                cluster_id=cid,
            ))
    return Dataset(records=records, class_map=class_map, source_format="coco")


def test_cluster_based_split_total_count():
    """クラスタ分割後の合計件数が元と一致すること。"""
    dataset = _make_clustered_dataset(n_clusters=3, per_cluster=10)
    result = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=42)

    assert len(result.records) == 30
    for r in result.records:
        assert r.split in ("train", "val", "test"), f"無効な split 値: {r.split!r}"


def test_cluster_based_split_proportional_within_each_cluster():
    """
    データリーク防止ルール検証（正しい実装）:
    各クラスタ内で train / val / test に比例配分されること。
    クラスタが単一の split に固められていないこと。
    """
    # 3クラスタ × 15件 = 45件（各クラスタで 7:2:1 比率 → train≈10, val≈3, test≈2）
    dataset = _make_clustered_dataset(n_clusters=3, per_cluster=15)
    result = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=42)

    for cid in range(3):
        cluster_records = [r for r in result.records if r.cluster_id == cid]
        splits_in_cluster = {r.split for r in cluster_records}
        # 15件のクラスタ内に train と val の両方が存在すること（比例配分の証明）
        assert "train" in splits_in_cluster, (
            f"クラスタ {cid} に train が含まれていない（データリーク防止ルール違反の疑い）"
        )
        assert "val" in splits_in_cluster, (
            f"クラスタ {cid} に val が含まれていない（データリーク防止ルール違反の疑い）"
        )


def test_cluster_based_split_no_duplicate():
    """同一 image_id が複数の split に重複して割り当てられないこと。"""
    dataset = _make_clustered_dataset(n_clusters=4, per_cluster=10)
    result = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=42)

    image_ids = [r.image_id for r in result.records]
    assert len(image_ids) == len(set(image_ids)), "同一 image_id が複数存在する"


def test_cluster_based_split_reproducibility():
    """同一シード値で2回分割した結果が一致すること（再現性）。"""
    dataset = _make_clustered_dataset(n_clusters=3, per_cluster=10)
    result1 = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=99)
    result2 = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=99)

    splits1 = {r.image_id: r.split for r in result1.records}
    splits2 = {r.image_id: r.split for r in result2.records}
    assert splits1 == splits2


def test_cluster_based_split_fallback_when_no_cluster_id():
    """cluster_id が None のレコードがある場合、split_random にフォールバックすること。"""
    # cluster_id が None のデータセット
    class_map = ClassMap.from_names(["dog"])
    records = [
        ImageRecord(
            image_id=str(i),
            original_filename=f"img_{i}.jpg",
            file_path=f"/tmp/img_{i}.jpg",
            width=640, height=480,
            annotations=[],
            cluster_id=None,  # 未設定
        )
        for i in range(20)
    ]
    dataset = Dataset(records=records, class_map=class_map, source_format="coco")
    result = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=42)

    # フォールバックでランダム分割が行われること
    assert len(result.records) == 20
    assert all(r.split in ("train", "val", "test") for r in result.records)


def test_cluster_based_split_not_cluster_grouped():
    """
    データリーク防止ルール：クラスタが単一の split に固められていないこと。
    各クラスタ内で split が混在していること。
    """
    # 3クラスタ × 20件 = 60件（十分な件数でテスト）
    dataset = _make_clustered_dataset(n_clusters=3, per_cluster=20)
    result = split_random_with_clusters(dataset, 0.7, 0.2, 0.1, seed=42)

    for cid in range(3):
        cluster_splits = {r.split for r in result.records if r.cluster_id == cid}
        assert len(cluster_splits) > 1, (
            f"クラスタ {cid} が単一の split に固められている"
            "（データリーク防止ルール違反: 禁止事項）"
        )
