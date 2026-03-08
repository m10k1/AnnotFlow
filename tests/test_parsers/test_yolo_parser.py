"""
YOLOパーサーのテスト（Phase 1: 実装）
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from parsers.yolo_parser import YoloParser


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def yolo_dir(tmp_path: Path) -> Path:
    """YOLO形式のデータセットを tmp_path/yolo に作成して返す。"""
    root = tmp_path / "yolo"
    root.mkdir()
    (root / "data.yaml").write_text("names:\n  - dog\n  - cat\n  - car\nnc: 3\n", encoding="utf-8")
    labels = root / "labels"
    labels.mkdir()
    # image_001: dog(cx=0.5,cy=0.5,w=0.25,h=0.25), cat(cx=0.25,cy=0.375,w=0.15625,h=0.3125)
    (labels / "image_001.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n1 0.25 0.375 0.15625 0.3125\n", encoding="utf-8"
    )
    # image_002: car(cx=0.28125,cy=0.29167,w=0.43750,h=0.41667)
    (labels / "image_002.txt").write_text(
        "2 0.28125 0.29167 0.43750 0.41667\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def img_dir_640(tmp_path: Path) -> Path:
    """640x480 の画像ファイル 2枚を含む画像フォルダ。"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")
    Image.new("RGB", (640, 480)).save(img_dir / "image_002.jpg")
    return img_dir


@pytest.fixture
def img_dir_one_only(tmp_path: Path) -> Path:
    """image_001.jpg のみ（image_002.jpg 欠落）。"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")
    return img_dir


# ---------------------------------------------------------------------------
# テスト本体
# ---------------------------------------------------------------------------

def test_yolo_relative_to_absolute_conversion(yolo_dir, img_dir_640):
    """YOLO相対座標(cx,cy,w,h)が絶対座標に正しく変換されること。
    cx=0.5, cy=0.5, w=0.25, h=0.25, W=640, H=480
    -> x_min=240, y_min=180, x_max=400, y_max=300
    """
    parser = YoloParser()
    dataset, _ = parser.parse(yolo_dir / "data.yaml", img_dir_640)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    dog_ann = next(a for a in record.annotations if a.class_name == "dog")

    assert dog_ann.bbox.x_min == pytest.approx(240.0)
    assert dog_ann.bbox.y_min == pytest.approx(180.0)
    assert dog_ann.bbox.x_max == pytest.approx(400.0)
    assert dog_ann.bbox.y_max == pytest.approx(300.0)


def test_yolo_roundtrip(yolo_dir, img_dir_640):
    """絶対座標 -> YOLO相対座標 -> 絶対座標の変換で元の値に戻ること（丸め誤差1px以内）。"""
    parser = YoloParser()
    dataset, _ = parser.parse(yolo_dir / "data.yaml", img_dir_640)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    dog_ann = next(a for a in record.annotations if a.class_name == "dog")

    W, H = float(record.width), float(record.height)
    bbox = dog_ann.bbox

    cx = (bbox.x_min + bbox.x_max) / 2 / W
    cy = (bbox.y_min + bbox.y_max) / 2 / H
    w = (bbox.x_max - bbox.x_min) / W
    h = (bbox.y_max - bbox.y_min) / H

    x_min_rt = (cx - w / 2) * W
    y_min_rt = (cy - h / 2) * H
    x_max_rt = (cx + w / 2) * W
    y_max_rt = (cy + h / 2) * H

    assert abs(x_min_rt - bbox.x_min) <= 1.0
    assert abs(y_min_rt - bbox.y_min) <= 1.0
    assert abs(x_max_rt - bbox.x_max) <= 1.0
    assert abs(y_max_rt - bbox.y_max) <= 1.0


def test_yolo_class_id_matches_yaml_order(yolo_dir, img_dir_640):
    """data.yamlの行順通りにクラスIDが採番されること。dog=0, cat=1, car=2"""
    parser = YoloParser()
    dataset, _ = parser.parse(yolo_dir / "data.yaml", img_dir_640)

    assert dataset.class_map.name_to_id["dog"] == 0
    assert dataset.class_map.name_to_id["cat"] == 1
    assert dataset.class_map.name_to_id["car"] == 2


def test_yolo_missing_image_skipped(yolo_dir, img_dir_one_only):
    """対応する画像ファイルが存在しないレコードがスキップされること。"""
    parser = YoloParser()
    dataset, skipped = parser.parse(yolo_dir / "data.yaml", img_dir_one_only)

    filenames = [r.original_filename for r in dataset.records]
    assert not any("image_002" in f for f in filenames)
    assert len(skipped) == 1
    assert "image_002" in str(skipped[0])


def test_yolo_classes_txt_fallback(tmp_path: Path):
    """data.yaml がない場合に classes.txt からクラス情報を読み込めること。"""
    root = tmp_path / "yolo_cls"
    root.mkdir()
    (root / "classes.txt").write_text("dog\ncat\ncar\n", encoding="utf-8")
    labels = root / "labels"
    labels.mkdir()
    (labels / "image_001.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    img_dir = tmp_path / "images_cls"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")

    parser = YoloParser()
    dataset, _ = parser.parse(root / "classes.txt", img_dir)

    assert dataset.class_map.name_to_id["dog"] == 0
    assert len(dataset.records) == 1
