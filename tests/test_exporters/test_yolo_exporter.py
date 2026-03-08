"""
YOLO エクスポーターのテスト（Phase 0: スケルトン）

Phase 1 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_yolo_export_absolute_to_relative_conversion():
    """エクスポート時に絶対座標がYOLO相対座標に正しく変換されること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_export_preserves_original_filename():
    """エクスポートされるtxtファイル名が元の画像ファイル名と一致すること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_export_data_yaml_contains_all_classes():
    """data.yaml にすべてのクラス名が含まれること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_export_directory_structure():
    """train/images/, train/labels/ 等の正しいディレクトリ構造が作成されること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_export_no_segmentation():
    """YOLOエクスポートに segmentation が出力されないこと（BBoxのみ）"""
    pytest.skip("Phase 1 で実装")
