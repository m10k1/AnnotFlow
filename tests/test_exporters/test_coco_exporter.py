"""
COCO エクスポーターのテスト（Phase 2: 実装）

- split 別の JSON ファイル生成確認
- segmentation の無加工パススルー確認
- 内部絶対座標 → COCO [x, y, w, h] への変換確認
- 元ファイル名の保持確認
- category_id の 1始まり変換確認
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from exporters.coco_exporter import CocoExporter


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_record(
    image_id: str,
    original_filename: str,
    annotations: list[Annotation],
    split: str | None = None,
    width: int = 640,
    height: int = 480,
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        original_filename=original_filename,
        file_path=f"/fake/images/{original_filename}",
        width=width,
        height=height,
        annotations=annotations,
        split=split,
    )


def _make_ann(
    class_id: int,
    class_name: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    segmentation: list | None = None,
) -> Annotation:
    return Annotation(
        class_id=class_id,
        class_name=class_name,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        segmentation=segmentation,
    )


def _make_dataset(records: list[ImageRecord], class_names: list[str]) -> Dataset:
    return Dataset(
        records=records,
        class_map=ClassMap.from_names(class_names),
        source_format="coco",
    )


# ---------------------------------------------------------------------------
# test_coco_export_creates_split_files
# ---------------------------------------------------------------------------

def test_coco_export_creates_split_files(tmp_path: Path) -> None:
    """split 別に train.json / val.json / test.json が生成されること。"""
    records = [
        _make_record("001", "image_001.jpg", [_make_ann(0, "dog", 10, 20, 110, 70)], split="train"),
        _make_record("002", "image_002.jpg", [_make_ann(0, "dog", 10, 20, 110, 70)], split="val"),
        _make_record("003", "image_003.jpg", [_make_ann(0, "dog", 10, 20, 110, 70)], split="test"),
    ]
    dataset = _make_dataset(records, ["dog"])

    exporter = CocoExporter()
    exporter.export(dataset, tmp_path, copy_images=False)

    assert (tmp_path / "train.json").exists(), "train.json が生成されていない"
    assert (tmp_path / "val.json").exists(), "val.json が生成されていない"
    assert (tmp_path / "test.json").exists(), "test.json が生成されていない"

    # train.json に1件のみ含まれていること
    train_data = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert len(train_data["images"]) == 1
    assert len(train_data["annotations"]) == 1


# ---------------------------------------------------------------------------
# test_coco_export_segmentation_preserved
# ---------------------------------------------------------------------------

def test_coco_export_segmentation_preserved(tmp_path: Path) -> None:
    """エクスポートした COCO JSON に segmentation がそのまま出力されること。"""
    seg_data = [[10, 20, 100, 20, 100, 150, 10, 150]]
    records = [
        _make_record(
            "001", "image_001.jpg",
            [_make_ann(0, "dog", 10, 20, 100, 150, segmentation=seg_data)],
            split="train",
        ),
    ]
    dataset = _make_dataset(records, ["dog"])

    CocoExporter().export(dataset, tmp_path, copy_images=False)

    data = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["segmentation"] == seg_data, \
        "segmentation が変換されずにそのまま出力されていない"


# ---------------------------------------------------------------------------
# test_coco_export_bbox_absolute_to_coco_format
# ---------------------------------------------------------------------------

def test_coco_export_bbox_absolute_to_coco_format(tmp_path: Path) -> None:
    """内部の絶対座標 BBox が COCO 形式の [x, y, w, h] に変換されて出力されること。

    BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70)
    → COCO bbox = [10, 20, 100, 50]  (width=100, height=50)
    """
    records = [
        _make_record(
            "001", "image_001.jpg",
            [_make_ann(0, "dog", 10.0, 20.0, 110.0, 70.0)],
            split="train",
        ),
    ]
    dataset = _make_dataset(records, ["dog"])

    CocoExporter().export(dataset, tmp_path, copy_images=False)

    data = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    bbox = data["annotations"][0]["bbox"]
    assert bbox == [10.0, 20.0, 100.0, 50.0], f"BBox 変換が正しくない: {bbox}"


# ---------------------------------------------------------------------------
# test_coco_export_preserves_original_filename
# ---------------------------------------------------------------------------

def test_coco_export_preserves_original_filename(tmp_path: Path) -> None:
    """エクスポートされる JSON の file_name が元のファイル名（接頭辞なし）であること。"""
    records = [
        _make_record(
            "001", "my_photo.jpg",   # Label Studio 接頭辞なし、元のファイル名
            [_make_ann(0, "dog", 10, 20, 110, 70)],
            split="train",
        ),
    ]
    dataset = _make_dataset(records, ["dog"])

    CocoExporter().export(dataset, tmp_path, copy_images=False)

    data = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    file_name = data["images"][0]["file_name"]
    assert file_name == "my_photo.jpg", f"file_name が正しくない: {file_name}"


# ---------------------------------------------------------------------------
# test_coco_export_category_ids_zero_based
# ---------------------------------------------------------------------------

def test_coco_export_category_ids_zero_based(tmp_path: Path) -> None:
    """内部の 0始まり class_id が COCO 形式の 1始まり category_id に変換されること。

    内部: dog=0, cat=1
    COCO: dog の category_id=1, cat の category_id=2
    """
    records = [
        _make_record(
            "001", "image_001.jpg",
            [
                _make_ann(0, "dog", 10, 20, 110, 70),
                _make_ann(1, "cat", 200, 50, 350, 200),
            ],
            split="train",
        ),
    ]
    dataset = _make_dataset(records, ["dog", "cat"])

    CocoExporter().export(dataset, tmp_path, copy_images=False)

    data = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))

    # categories の id が 1始まりであること
    cat_ids = {c["name"]: c["id"] for c in data["categories"]}
    assert cat_ids["dog"] == 1, f"dog の category_id が 1 でない: {cat_ids['dog']}"
    assert cat_ids["cat"] == 2, f"cat の category_id が 2 でない: {cat_ids['cat']}"

    # annotations の category_id も 1始まりであること
    ann_cat_ids = {a["category_id"] for a in data["annotations"]}
    assert 1 in ann_cat_ids, "dog (category_id=1) が annotations に存在しない"
    assert 2 in ann_cat_ids, "cat (category_id=2) が annotations に存在しない"
