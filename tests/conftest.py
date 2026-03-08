"""
共通フィクスチャ（conftest.py）

全テストモジュールから利用できる共通のサンプルデータやフィクスチャを定義する。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.dataset import (
    Annotation,
    BoundingBox,
    ClassMap,
    Dataset,
    ImageRecord,
)

# ---------------------------------------------------------------------------
# フィクスチャディレクトリ
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# ClassMap フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_class_map() -> ClassMap:
    """テスト用クラスマップ（dog=0, cat=1, car=2）"""
    return ClassMap.from_names(["dog", "cat", "car"])


# ---------------------------------------------------------------------------
# ImageRecord フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_record(sample_class_map: ClassMap) -> ImageRecord:
    """テスト用 ImageRecord（dog と cat が1枚に混在するマルチラベル画像）"""
    return ImageRecord(
        image_id="001",
        original_filename="image_001.jpg",
        file_path="/tmp/images/image_001.jpg",
        width=640,
        height=480,
        annotations=[
            Annotation(
                class_id=0,
                class_name="dog",
                bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=150),
            ),
            Annotation(
                class_id=1,
                class_name="cat",
                bbox=BoundingBox(x_min=200, y_min=50, x_max=350, y_max=200),
            ),
        ],
    )


@pytest.fixture
def sample_image_record_with_segmentation(sample_class_map: ClassMap) -> ImageRecord:
    """テスト用 ImageRecord（segmentation フィールドを持つ）"""
    return ImageRecord(
        image_id="002",
        original_filename="image_002.jpg",
        file_path="/tmp/images/image_002.jpg",
        width=640,
        height=480,
        annotations=[
            Annotation(
                class_id=0,
                class_name="dog",
                bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=150),
                segmentation=[[10, 20, 100, 20, 100, 150, 10, 150]],
            ),
        ],
    )


@pytest.fixture
def sample_car_record(sample_class_map: ClassMap) -> ImageRecord:
    """テスト用 ImageRecord（car のみ）"""
    return ImageRecord(
        image_id="003",
        original_filename="image_003.jpg",
        file_path="/tmp/images/image_003.jpg",
        width=800,
        height=600,
        annotations=[
            Annotation(
                class_id=2,
                class_name="car",
                bbox=BoundingBox(x_min=50, y_min=100, x_max=400, y_max=350),
            ),
        ],
    )


@pytest.fixture
def empty_record() -> ImageRecord:
    """テスト用 ImageRecord（アノテーションなし）"""
    return ImageRecord(
        image_id="004",
        original_filename="image_004.jpg",
        file_path="/tmp/images/image_004.jpg",
        width=640,
        height=480,
        annotations=[],
    )


# ---------------------------------------------------------------------------
# Dataset フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dataset(sample_image_record: ImageRecord, sample_class_map: ClassMap) -> Dataset:
    """テスト用 Dataset（10件、すべて同一の ImageRecord のコピー）"""
    records = [
        ImageRecord(
            image_id=str(i).zfill(3),
            original_filename=f"image_{str(i).zfill(3)}.jpg",
            file_path=f"/tmp/images/image_{str(i).zfill(3)}.jpg",
            width=640,
            height=480,
            annotations=[
                Annotation(
                    class_id=0,
                    class_name="dog",
                    bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=150),
                ),
                Annotation(
                    class_id=1,
                    class_name="cat",
                    bbox=BoundingBox(x_min=200, y_min=50, x_max=350, y_max=200),
                ),
            ],
        )
        for i in range(1, 11)
    ]
    return Dataset(records=records, class_map=sample_class_map, source_format="coco")


@pytest.fixture
def multilabel_dataset() -> Dataset:
    """
    マルチラベルの分割テスト用 Dataset（30件）。
    各クラスの分布に偏りがあるが、全クラスが複数の画像に存在する。
    """
    class_map = ClassMap.from_names(["dog", "cat", "car"])
    records = []
    for i in range(30):
        # ローテーションで異なるクラス組み合わせを割り当てる
        pattern = i % 3
        if pattern == 0:
            anns = [
                Annotation(class_id=0, class_name="dog",
                           bbox=BoundingBox(10, 20, 100, 150)),
                Annotation(class_id=1, class_name="cat",
                           bbox=BoundingBox(200, 50, 350, 200)),
            ]
        elif pattern == 1:
            anns = [
                Annotation(class_id=1, class_name="cat",
                           bbox=BoundingBox(10, 20, 100, 150)),
                Annotation(class_id=2, class_name="car",
                           bbox=BoundingBox(200, 50, 400, 300)),
            ]
        else:
            anns = [
                Annotation(class_id=0, class_name="dog",
                           bbox=BoundingBox(10, 20, 100, 150)),
                Annotation(class_id=2, class_name="car",
                           bbox=BoundingBox(200, 50, 400, 300)),
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
# フィクスチャファイルパス（サンプルデータ）
# ---------------------------------------------------------------------------

@pytest.fixture
def coco_sample_path() -> Path:
    """COCO JSON サンプルファイルのパス"""
    return FIXTURES_DIR / "sample_coco.json"


@pytest.fixture
def yolo_sample_dir() -> Path:
    """YOLO サンプルデータディレクトリのパス"""
    return FIXTURES_DIR / "yolo_sample"


@pytest.fixture
def label_studio_sample_path() -> Path:
    """Label Studio JSON サンプルファイルのパス"""
    return FIXTURES_DIR / "sample_label_studio.json"
