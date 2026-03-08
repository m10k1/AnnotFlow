"""
COCOパーサーのテスト（Phase 1: 実装）
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from parsers.coco_parser import CocoParser


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def coco_json(tmp_path: Path) -> Path:
    data = {
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"},
        ],
        "images": [
            {"id": 1, "file_name": "image_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image_002.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 90, 130], "area": 11700, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 50, 150, 150], "area": 22500, "iscrowd": 0,
             "segmentation": [[200, 50, 350, 50, 350, 200, 200, 200]]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 100, 350, 250], "area": 87500, "iscrowd": 0},
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


@pytest.fixture
def img_dir_both(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")
    Image.new("RGB", (800, 600)).save(img_dir / "image_002.jpg")
    return img_dir


@pytest.fixture
def img_dir_one_only(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")
    return img_dir


# ---------------------------------------------------------------------------
# テスト本体
# ---------------------------------------------------------------------------

def test_coco_bbox_conversion(coco_json, img_dir_both):
    """COCOの[x,y,w,h]が内部モデルの絶対座標[x_min,y_min,x_max,y_max]に正しく変換されること。
    bbox=[10, 20, 90, 130] -> x_min=10, y_min=20, x_max=100, y_max=150
    """
    parser = CocoParser()
    dataset, _ = parser.parse(coco_json, img_dir_both)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    dog_ann = next(a for a in record.annotations if a.class_name == "dog")

    assert dog_ann.bbox.x_min == pytest.approx(10.0)
    assert dog_ann.bbox.y_min == pytest.approx(20.0)
    assert dog_ann.bbox.x_max == pytest.approx(100.0)
    assert dog_ann.bbox.y_max == pytest.approx(150.0)


def test_coco_category_id_renumbered_to_zero_based(coco_json, img_dir_both):
    """COCOの1始まりcategory_idが0始まりに正規化されること。COCO: dog=1, cat=2 -> 内部: dog=0, cat=1"""
    parser = CocoParser()
    dataset, _ = parser.parse(coco_json, img_dir_both)

    assert dataset.class_map.name_to_id["dog"] == 0
    assert dataset.class_map.name_to_id["cat"] == 1

    for record in dataset.records:
        for ann in record.annotations:
            assert ann.class_id == dataset.class_map.name_to_id[ann.class_name]


def test_coco_segmentation_preserved_without_modification(coco_json, img_dir_both):
    """segmentationフィールドが変換されずそのまま保持されること。"""
    parser = CocoParser()
    dataset, _ = parser.parse(coco_json, img_dir_both)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    cat_ann = next(a for a in record.annotations if a.class_name == "cat")

    assert cat_ann.segmentation == [[200, 50, 350, 50, 350, 200, 200, 200]]


def test_coco_missing_image_skipped(coco_json, img_dir_one_only):
    """対応する画像ファイルが存在しないレコードがスキップされ、skipped_recordsに記録されること。"""
    parser = CocoParser()
    dataset, skipped = parser.parse(coco_json, img_dir_one_only)

    filenames = [r.original_filename for r in dataset.records]
    assert not any("image_002" in f for f in filenames)
    assert len(skipped) == 1
    assert "image_002" in str(skipped[0])


def test_coco_multiple_annotations_per_image(coco_json, img_dir_both):
    """1枚の画像に複数のアノテーションが正しく紐づくこと。"""
    parser = CocoParser()
    dataset, _ = parser.parse(coco_json, img_dir_both)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    assert len(record.annotations) == 2


def test_coco_class_map_zero_based(coco_json, img_dir_both):
    """COCO JSON のカテゴリが ClassMap に0始まりで登録されること。"""
    parser = CocoParser()
    dataset, _ = parser.parse(coco_json, img_dir_both)

    ids = list(dataset.class_map.id_to_name.keys())
    assert min(ids) == 0
    assert len(ids) == 2
