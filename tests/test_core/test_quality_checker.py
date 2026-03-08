"""
品質チェックロジックのテスト（Phase 0: スケルトン）

Phase 1 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_find_duplicates_returns_groups():
    """同一 MD5 ハッシュの画像が重複グループとして検出されること"""
    pytest.skip("Phase 1 で実装")


def test_find_duplicates_no_duplicates():
    """重複がない場合は空のリストが返されること"""
    pytest.skip("Phase 1 で実装")


def test_find_empty_annotations_detects_zero_annotation():
    """アノテーション数がゼロの画像が検出されること"""
    pytest.skip("Phase 1 で実装")


def test_find_empty_annotations_none_when_all_annotated():
    """すべての画像にアノテーションがある場合は空のリストが返されること"""
    pytest.skip("Phase 1 で実装")


def test_compute_class_distribution():
    """クラスごとのアノテーション数が正しく集計されること"""
    pytest.skip("Phase 1 で実装")
