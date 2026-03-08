"""
品質チェックロジックのテスト（Phase 1: 実装）

- find_duplicates: MD5ハッシュによる重複画像検出
- find_empty_annotations: アノテーションなし画像の検出
- compute_class_distribution: クラスごとのアノテーション数集計
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from core.quality_checker import (
    compute_class_distribution,
    find_duplicates,
    find_empty_annotations,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_record(
    image_id: str,
    file_path: str,
    annotations: list[Annotation] | None = None,
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        original_filename=f"image_{image_id}.jpg",
        file_path=file_path,
        width=640,
        height=480,
        annotations=annotations or [],
    )


def _make_dataset(records: list[ImageRecord]) -> Dataset:
    return Dataset(
        records=records,
        class_map=ClassMap.from_names(["dog", "cat", "car"]),
        source_format="coco",
    )


# ---------------------------------------------------------------------------
# find_duplicates のテスト
# ---------------------------------------------------------------------------

def test_find_duplicates_returns_groups(tmp_path: Path):
    """同一MD5ハッシュの画像が重複グループとして検出されること。"""
    # 同じ内容の画像を2枚、別内容を1枚作成
    img_a = tmp_path / "image_001.jpg"
    img_b = tmp_path / "image_002.jpg"
    img_c = tmp_path / "image_003.jpg"
    Image.new("RGB", (640, 480), color=(255, 0, 0)).save(img_a)
    Image.new("RGB", (640, 480), color=(255, 0, 0)).save(img_b)  # image_001 と同一
    Image.new("RGB", (640, 480), color=(0, 255, 0)).save(img_c)  # 別画像

    records = [
        _make_record("001", str(img_a)),
        _make_record("002", str(img_b)),
        _make_record("003", str(img_c)),
    ]
    dataset = _make_dataset(records)

    groups = find_duplicates(dataset)

    # 1グループ（001と002）が検出されること
    assert len(groups) == 1
    assert set(groups[0]) == {"001", "002"}


def test_find_duplicates_no_duplicates(tmp_path: Path):
    """重複がない場合は空のリストが返されること。"""
    img_a = tmp_path / "image_001.jpg"
    img_b = tmp_path / "image_002.jpg"
    Image.new("RGB", (640, 480), color=(255, 0, 0)).save(img_a)
    Image.new("RGB", (640, 480), color=(0, 0, 255)).save(img_b)

    records = [
        _make_record("001", str(img_a)),
        _make_record("002", str(img_b)),
    ]
    dataset = _make_dataset(records)

    groups = find_duplicates(dataset)
    assert groups == []


def test_find_duplicates_three_identical(tmp_path: Path):
    """3枚が同一の場合、1グループに3件含まれること。"""
    content = Image.new("RGB", (640, 480), color=(128, 128, 128))
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        content.save(tmp_path / name)

    records = [
        _make_record("001", str(tmp_path / "a.jpg")),
        _make_record("002", str(tmp_path / "b.jpg")),
        _make_record("003", str(tmp_path / "c.jpg")),
    ]
    dataset = _make_dataset(records)

    groups = find_duplicates(dataset)
    assert len(groups) == 1
    assert set(groups[0]) == {"001", "002", "003"}


def test_find_duplicates_skips_missing_files(tmp_path: Path):
    """存在しないファイルパスのレコードはスキップしてクラッシュしないこと。"""
    img_a = tmp_path / "image_001.jpg"
    Image.new("RGB", (640, 480)).save(img_a)

    records = [
        _make_record("001", str(img_a)),
        _make_record("002", str(tmp_path / "nonexistent.jpg")),  # 存在しない
    ]
    dataset = _make_dataset(records)

    # 例外が発生せず、重複なし（001のみ）
    groups = find_duplicates(dataset)
    assert groups == []


# ---------------------------------------------------------------------------
# find_empty_annotations のテスト
# ---------------------------------------------------------------------------

def test_find_empty_annotations_detects_zero_annotation():
    """アノテーション数がゼロの画像が検出されること。"""
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[
            Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
        ]),
        _make_record("002", "/fake/002.jpg", annotations=[]),
        _make_record("003", "/fake/003.jpg", annotations=[]),
    ]
    dataset = _make_dataset(records)

    empty_ids = find_empty_annotations(dataset)

    assert "001" not in empty_ids
    assert "002" in empty_ids
    assert "003" in empty_ids


def test_find_empty_annotations_none_when_all_annotated():
    """すべての画像にアノテーションがある場合は空のリストが返されること。"""
    ann = Annotation(0, "dog", BoundingBox(10, 20, 100, 150))
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[ann]),
        _make_record("002", "/fake/002.jpg", annotations=[ann]),
    ]
    dataset = _make_dataset(records)

    empty_ids = find_empty_annotations(dataset)
    assert empty_ids == []


# ---------------------------------------------------------------------------
# compute_class_distribution のテスト
# ---------------------------------------------------------------------------

def test_compute_class_distribution():
    """クラスごとのアノテーション数が正しく集計されること。"""
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[
            Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
            Annotation(1, "cat", BoundingBox(200, 50, 350, 200)),
        ]),
        _make_record("002", "/fake/002.jpg", annotations=[
            Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
        ]),
        _make_record("003", "/fake/003.jpg", annotations=[
            Annotation(2, "car", BoundingBox(50, 100, 400, 350)),
        ]),
    ]
    dataset = _make_dataset(records)

    dist = compute_class_distribution(dataset)

    assert dist["dog"] == 2
    assert dist["cat"] == 1
    assert dist["car"] == 1


def test_compute_class_distribution_empty_dataset():
    """アノテーションが一切ない場合は空の辞書が返されること。"""
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[]),
    ]
    dataset = _make_dataset(records)

    dist = compute_class_distribution(dataset)
    assert dist == {}


# ---------------------------------------------------------------------------
# compute_annotation_count_per_image のテスト
# ---------------------------------------------------------------------------

from core.quality_checker import compute_annotation_count_per_image  # noqa: E402


def test_compute_annotation_count_per_image_basic():
    """画像ごとのアノテーション数が正しく集計されること。"""
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[
            Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
            Annotation(1, "cat", BoundingBox(200, 50, 350, 200)),
        ]),
        _make_record("002", "/fake/002.jpg", annotations=[
            Annotation(0, "dog", BoundingBox(10, 20, 100, 150)),
        ]),
        _make_record("003", "/fake/003.jpg", annotations=[]),
    ]
    dataset = _make_dataset(records)

    result = compute_annotation_count_per_image(dataset)

    assert result["001"] == 2
    assert result["002"] == 1
    assert result["003"] == 0


def test_compute_annotation_count_per_image_empty_dataset():
    """レコードが0件の場合は空の辞書が返されること。"""
    dataset = _make_dataset([])

    result = compute_annotation_count_per_image(dataset)

    assert result == {}


def test_compute_annotation_count_per_image_all_empty():
    """全レコードのアノテーション数が0の場合も正しく返されること。"""
    records = [
        _make_record("001", "/fake/001.jpg", annotations=[]),
        _make_record("002", "/fake/002.jpg", annotations=[]),
    ]
    dataset = _make_dataset(records)

    result = compute_annotation_count_per_image(dataset)

    assert result == {"001": 0, "002": 0}
