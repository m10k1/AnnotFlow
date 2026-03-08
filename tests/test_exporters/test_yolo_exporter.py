"""
YOLO エクスポーターのテスト（Phase 1: 実装）

- 絶対座標 → YOLO相対座標への変換精度
- 元ファイル名の保持
- data.yaml のクラス内容
- ディレクトリ構造の正確さ
- segmentation が出力されないこと
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from exporters.yolo_exporter import YoloExporter


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_dataset_with_splits(tmp_path: Path) -> Dataset:
    """
    train/val それぞれに1枚ずつの画像を持つ Dataset を作成する。
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (640, 480)).save(img_dir / "image_001.jpg")
    Image.new("RGB", (800, 600)).save(img_dir / "image_002.jpg")

    class_map = ClassMap.from_names(["dog", "cat"])

    records = [
        ImageRecord(
            image_id="001",
            original_filename="image_001.jpg",
            file_path=str(img_dir / "image_001.jpg"),
            width=640,
            height=480,
            split="train",
            annotations=[
                Annotation(
                    class_id=0,
                    class_name="dog",
                    bbox=BoundingBox(x_min=240.0, y_min=180.0, x_max=400.0, y_max=300.0),
                    segmentation=[[240, 180, 400, 180, 400, 300, 240, 300]],  # segmentation あり
                ),
                Annotation(
                    class_id=1,
                    class_name="cat",
                    bbox=BoundingBox(x_min=10.0, y_min=20.0, x_max=100.0, y_max=150.0),
                ),
            ],
        ),
        ImageRecord(
            image_id="002",
            original_filename="image_002.jpg",
            file_path=str(img_dir / "image_002.jpg"),
            width=800,
            height=600,
            split="val",
            annotations=[
                Annotation(
                    class_id=0,
                    class_name="dog",
                    bbox=BoundingBox(x_min=50.0, y_min=100.0, x_max=400.0, y_max=350.0),
                ),
            ],
        ),
    ]

    return Dataset(records=records, class_map=class_map, source_format="coco")


# ---------------------------------------------------------------------------
# テスト本体
# ---------------------------------------------------------------------------

def test_yolo_export_absolute_to_relative_conversion(tmp_path: Path):
    """エクスポート時に絶対座標がYOLO相対座標に正しく変換されること。
    bbox: x_min=240, y_min=180, x_max=400, y_max=300, W=640, H=480
    cx=(240+400)/2/640=0.5, cy=(180+300)/2/480=0.5, w=(400-240)/640=0.25, h=(300-180)/480=0.25
    """
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    label_path = output_dir / "train" / "labels" / "image_001.txt"
    assert label_path.exists(), "train/labels/image_001.txt が生成されていない"

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    # dog (class_id=0) の行を取得
    dog_line = next(l for l in lines if l.startswith("0 "))
    parts = dog_line.split()
    assert len(parts) == 5

    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    assert cx == pytest.approx(0.5, abs=1e-4)
    assert cy == pytest.approx(0.5, abs=1e-4)
    assert w == pytest.approx(0.25, abs=1e-4)
    assert h == pytest.approx(0.25, abs=1e-4)


def test_yolo_export_preserves_original_filename(tmp_path: Path):
    """エクスポートされるtxtファイル名が元の画像ファイル名（拡張子なし）と一致すること。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    # image_001.jpg → image_001.txt
    assert (output_dir / "train" / "labels" / "image_001.txt").exists()
    assert (output_dir / "val" / "labels" / "image_002.txt").exists()


def test_yolo_export_data_yaml_contains_all_classes(tmp_path: Path):
    """data.yaml にすべてのクラス名が含まれること。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    yaml_path = output_dir / "data.yaml"
    assert yaml_path.exists(), "data.yaml が生成されていない"

    content = yaml_path.read_text(encoding="utf-8")
    assert "dog" in content
    assert "cat" in content


def test_yolo_export_directory_structure(tmp_path: Path):
    """train/images/, train/labels/, val/images/, val/labels/ のディレクトリ構造が作成されること。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    # ラベルディレクトリが存在すること
    assert (output_dir / "train" / "labels").is_dir()
    assert (output_dir / "val" / "labels").is_dir()
    # copy_images=False なので images ディレクトリは作成しない（またはあっても空）


def test_yolo_export_no_segmentation(tmp_path: Path):
    """YOLOエクスポートにsegmentationが出力されないこと（BBoxのみ、各行が5要素）。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    label_path = output_dir / "train" / "labels" / "image_001.txt"
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    for line in lines:
        parts = line.split()
        assert len(parts) == 5, (
            f"YOLO ラベル行は class_id cx cy w h の5要素のみ: {line!r}"
        )


def test_yolo_export_copy_images(tmp_path: Path):
    """copy_images=True の場合、画像ファイルが出力ディレクトリにコピーされること。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=True)

    assert (output_dir / "train" / "images" / "image_001.jpg").exists()
    assert (output_dir / "val" / "images" / "image_002.jpg").exists()


def test_yolo_export_data_yaml_nc_field(tmp_path: Path):
    """data.yaml に nc（クラス数）フィールドが含まれること。"""
    dataset = _make_dataset_with_splits(tmp_path)
    output_dir = tmp_path / "output"

    exporter = YoloExporter()
    exporter.export(dataset, output_dir, copy_images=False)

    content = (output_dir / "data.yaml").read_text(encoding="utf-8")
    assert "nc:" in content
