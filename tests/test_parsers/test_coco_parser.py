"""
COCOパーサーのテスト（Phase 0: スケルトン）

Phase 1 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_coco_bbox_conversion():
    """COCOの[x,y,w,h]が内部モデルの絶対座標[x_min,y_min,x_max,y_max]に正しく変換されること"""
    pytest.skip("Phase 1 で実装")


def test_coco_category_id_renumbered_to_zero_based():
    """COCOの1始まりcategory_idが0始まりに正規化されること"""
    pytest.skip("Phase 1 で実装")


def test_coco_segmentation_preserved_without_modification():
    """segmentationフィールドが変換されずそのまま保持されること"""
    pytest.skip("Phase 1 で実装")


def test_coco_missing_image_skipped():
    """対応する画像ファイルが存在しないレコードがスキップされ、skipped_recordsに記録されること"""
    pytest.skip("Phase 1 で実装")


def test_coco_multiple_annotations_per_image():
    """1枚の画像に複数のアノテーションが正しく紐づくこと"""
    pytest.skip("Phase 1 で実装")


def test_coco_class_map_zero_based():
    """COCO JSON のカテゴリが ClassMap に0始まりで登録されること"""
    pytest.skip("Phase 1 で実装")
