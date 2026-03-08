"""
YOLOパーサーのテスト（Phase 0: スケルトン）

Phase 1 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_yolo_relative_to_absolute_conversion():
    """
    YOLO相対座標(cx,cy,w,h)が絶対座標に正しく変換されること。

    例: cx=0.5, cy=0.5, w=0.25, h=0.25, image_width=640, image_height=480
    → x_min=240, y_min=180, x_max=400, y_max=300
    """
    pytest.skip("Phase 1 で実装")


def test_yolo_roundtrip():
    """絶対座標 → YOLO相対座標 → 絶対座標の変換で元の値に戻ること（丸め誤差1px以内）"""
    pytest.skip("Phase 1 で実装")


def test_yolo_class_id_matches_yaml_order():
    """data.yamlの行順通りにクラスIDが採番されること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_missing_image_skipped():
    """対応する画像ファイルが存在しないレコードがスキップされること"""
    pytest.skip("Phase 1 で実装")


def test_yolo_classes_txt_fallback():
    """data.yaml がない場合に classes.txt からクラス情報を読み込めること"""
    pytest.skip("Phase 1 で実装")
