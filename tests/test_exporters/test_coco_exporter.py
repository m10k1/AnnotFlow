"""
COCO エクスポーターのテスト（Phase 0: スケルトン）

Phase 2 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_coco_export_creates_split_files():
    """split別にtrain.json / val.json / test.jsonが生成されること"""
    pytest.skip("Phase 2 で実装")


def test_coco_export_segmentation_preserved():
    """エクスポートした COCO JSON に segmentation がそのまま出力されること"""
    pytest.skip("Phase 2 で実装")


def test_coco_export_bbox_absolute_to_coco_format():
    """内部の絶対座標BBoxがCOCO形式の[x,y,w,h]に変換されて出力されること"""
    pytest.skip("Phase 2 で実装")


def test_coco_export_preserves_original_filename():
    """エクスポートされるJSONのfile_nameが元のファイル名であること"""
    pytest.skip("Phase 2 で実装")


def test_coco_export_category_ids_zero_based():
    """エクスポートした COCO JSON のcategory_idが1始まりに変換されること"""
    pytest.skip("Phase 2 で実装")
