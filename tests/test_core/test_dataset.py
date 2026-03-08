"""
内部データモデルのテスト（Phase 0: 一部実装済み）

ClassMap のユニットテストは実装済みなのでスケルトンではなく実際のテストコードを含む。
マージ処理のテストは Phase 1 で実装する。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from core.dataset import (
    ClassMap,
    ConflictResolution,
    Dataset,
    ImageRecord,
    MergeConflict,
    merge_datasets_dry_run,
    merge_datasets_resolve,
)


# ---------------------------------------------------------------------------
# ClassMap のテスト（Phase 0 で検証可能）
# ---------------------------------------------------------------------------

def test_classmap_zero_based_numbering():
    """ClassMap.from_names で 0始まりの ID が採番されること"""
    class_map = ClassMap.from_names(["dog", "cat", "car"])
    assert class_map.id_to_name == {0: "dog", 1: "cat", 2: "car"}
    assert class_map.name_to_id == {"dog": 0, "cat": 1, "car": 2}


def test_classmap_get_or_add_existing_class():
    """登録済みクラス名で get_or_add を呼ぶと既存のIDが返されること"""
    class_map = ClassMap.from_names(["dog", "cat"])
    assert class_map.get_or_add("dog") == 0
    assert class_map.get_or_add("cat") == 1


def test_classmap_get_or_add_new_class():
    """未登録クラス名で get_or_add を呼ぶと新しいIDが採番されること"""
    class_map = ClassMap.from_names(["dog", "cat"])
    new_id = class_map.get_or_add("car")
    assert new_id == 2
    assert class_map.id_to_name[2] == "car"
    assert class_map.name_to_id["car"] == 2


def test_classmap_bidirectional_consistency():
    """id_to_name と name_to_id が常に双方向で一致していること"""
    class_map = ClassMap.from_names(["dog", "cat"])
    class_map.get_or_add("car")
    for id_, name in class_map.id_to_name.items():
        assert class_map.name_to_id[name] == id_


# ---------------------------------------------------------------------------
# MVC 分離の確認（Phase 0 で検証可能）
# ---------------------------------------------------------------------------

def test_merge_dry_run_no_ui_import():
    """core/dataset.py が PySide6 をimportしていないこと（MVCの分離確認）"""
    source = pathlib.Path("core/dataset.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for imp in imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                assert "PySide6" not in alias.name, (
                    "core/dataset.py は PySide6 をimportしてはならない"
                )
        elif isinstance(imp, ast.ImportFrom):
            module = imp.module or ""
            assert "PySide6" not in module, (
                "core/dataset.py は PySide6 をimportしてはならない"
            )


# ---------------------------------------------------------------------------
# マージ処理のテスト（Phase 1 で実装）
# ---------------------------------------------------------------------------

def test_merge_dry_run_same_name_different_id():
    """パターン1：同一クラス名・異なるIDのデータセットをdry_runするとIDが統一されること"""
    pytest.skip("Phase 1 で実装")


def test_merge_dry_run_union_of_classes():
    """パターン2：片方にしか存在しないクラスがマージ後に和集合として含まれること"""
    pytest.skip("Phase 1 で実装")


def test_merge_dry_run_conflict_report():
    """dry_run 時の ID の変更が conflict_report リストに記録されること"""
    pytest.skip("Phase 1 で実装")


def test_merge_dry_run_pattern3_returns_unresolved():
    """パターン3（同ID・異クラス名）がunresolved_conflictsに格納され、自動解決されないこと"""
    pytest.skip("Phase 1 で実装")


def test_merge_resolve_applies_user_choice():
    """merge_datasets_resolve() がユーザーの選択（ConflictResolution）を正しく適用すること"""
    pytest.skip("Phase 1 で実装")
