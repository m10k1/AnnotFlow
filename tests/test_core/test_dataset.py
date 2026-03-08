"""
内部データモデルのテスト

ClassMap・データセットマージ（2段階）・MVC分離の確認。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from core.dataset import (
    Annotation,
    BoundingBox,
    ClassMap,
    ConflictResolution,
    Dataset,
    ImageRecord,
    MergeConflict,
    merge_datasets_dry_run,
    merge_datasets_resolve,
)


# ---------------------------------------------------------------------------
# ヘルパー：テスト用 Dataset を作るファクトリ
# ---------------------------------------------------------------------------

def _make_dataset(class_names: list[str], source_format: str = "coco") -> Dataset:
    """指定したクラス名だけを持つ空 Dataset を返す。"""
    return Dataset(
        records=[],
        class_map=ClassMap.from_names(class_names),
        source_format=source_format,
    )


def _make_dataset_with_custom_map(
    id_to_name: dict[int, str], source_format: str = "coco"
) -> Dataset:
    """任意の id_to_name を持つ Dataset を返す（パターン3のテスト用）。"""
    name_to_id = {v: k for k, v in id_to_name.items()}
    return Dataset(
        records=[],
        class_map=ClassMap(id_to_name=id_to_name, name_to_id=name_to_id),
        source_format=source_format,
    )


def _make_record(class_id: int, class_name: str, image_id: str = "001") -> ImageRecord:
    """テスト用 ImageRecord を返す。"""
    return ImageRecord(
        image_id=image_id,
        original_filename=f"image_{image_id}.jpg",
        file_path=f"/tmp/images/image_{image_id}.jpg",
        width=640,
        height=480,
        annotations=[
            Annotation(
                class_id=class_id,
                class_name=class_name,
                bbox=BoundingBox(10, 20, 100, 150),
            )
        ],
    )


# ---------------------------------------------------------------------------
# ClassMap のテスト
# ---------------------------------------------------------------------------

def test_classmap_zero_based_numbering():
    """ClassMap.from_names で 0始まりの ID が採番されること"""
    class_map = ClassMap.from_names(["dog", "cat", "car"])
    assert class_map.id_to_name == {0: "dog", 1: "cat", 2: "car"}
    assert class_map.name_to_id == {"dog": 0, "cat": 1, "car": 2}


def test_classmap_get_or_add_existing_class():
    """登録済みクラス名で get_or_add を呼ぶと既存の ID が返されること"""
    class_map = ClassMap.from_names(["dog", "cat"])
    assert class_map.get_or_add("dog") == 0
    assert class_map.get_or_add("cat") == 1


def test_classmap_get_or_add_new_class():
    """未登録クラス名で get_or_add を呼ぶと新しい ID が採番されること"""
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
# MVC 分離の確認
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
# マージ処理テスト（第1段階 dry_run）
# ---------------------------------------------------------------------------

def test_merge_dry_run_same_name_different_id():
    """
    パターン1：同一クラス名・異なるID。
    ds_a: {0:dog, 1:cat}、ds_b: {0:cat, 1:dog}（IDが逆転）
    → 名前を正として自動解決され、unresolved_conflicts が空であること。
    → マージ後も dog と cat が両方含まれること。
    """
    ds_a = _make_dataset(["dog", "cat"])
    # ds_b は意図的に cat=0, dog=1 とする（パターン1）
    ds_b = _make_dataset_with_custom_map({0: "cat", 1: "dog"})

    result = merge_datasets_dry_run([ds_a, ds_b])

    assert result.unresolved_conflicts == [], "パターン1は自動解決されるべき"
    names = set(result.merged_dataset.class_map.name_to_id.keys())
    assert "dog" in names
    assert "cat" in names


def test_merge_dry_run_same_name_different_id_remap_applied():
    """
    パターン1でID再採番が行われ、レコードのアノテーションIDが新しい ClassMap に
    合わせて更新されること。
    """
    # ds_a: dog=0, ds_b: dog=1（同名・異ID）
    ds_a = _make_dataset_with_custom_map({0: "dog"})
    ds_b = _make_dataset_with_custom_map({1: "dog"})

    # ds_b に dog の ImageRecord を追加（dog のID=1 でアノテーション）
    record_b = _make_record(class_id=1, class_name="dog", image_id="002")
    ds_b.records.append(record_b)

    result = merge_datasets_dry_run([ds_a, ds_b])

    # マージ後、dog の ID は 0 に統一されているはず
    merged_dog_id = result.merged_dataset.class_map.name_to_id["dog"]
    for record in result.merged_dataset.records:
        for ann in record.annotations:
            if ann.class_name == "dog":
                assert ann.class_id == merged_dog_id, (
                    f"dog のアノテーション ID が ClassMap と一致しない: {ann.class_id} != {merged_dog_id}"
                )


def test_merge_dry_run_union_of_classes():
    """
    パターン2：片方にしか存在しないクラス。
    ds_a: {dog, cat}、ds_b: {dog, car}
    → 和集合 {dog, cat, car} が含まれること。
    """
    ds_a = _make_dataset(["dog", "cat"])
    ds_b = _make_dataset(["dog", "car"])

    result = merge_datasets_dry_run([ds_a, ds_b])

    assert result.unresolved_conflicts == [], "パターン2は自動解決されるべき"
    names = set(result.merged_dataset.class_map.name_to_id.keys())
    assert names == {"dog", "cat", "car"}


def test_merge_dry_run_conflict_report():
    """
    ID 変更があった場合に conflict_report に記録されること。
    ds_a: {0:dog}, ds_b: {1:dog}（dog が異なる ID）
    → ds_b の dog が 0 に再採番され conflict_report に記録される。
    """
    ds_a = _make_dataset_with_custom_map({0: "dog"})
    ds_b = _make_dataset_with_custom_map({1: "dog"})

    result = merge_datasets_dry_run([ds_a, ds_b], dataset_labels=["A", "B"])

    # ds_b の dog は 1 → 0 に変更されるはず
    dog_changes = [
        r for r in result.conflict_report
        if r["class_name"] == "dog" and r["dataset"] == "B"
    ]
    assert len(dog_changes) >= 1
    changed = dog_changes[0]
    assert changed["old_id"] != changed["new_id"]


def test_merge_dry_run_pattern3_returns_unresolved():
    """
    パターン3：同ID・異クラス名（それぞれの名前が相手のデータセットに存在しない）。
    ds_a: {0:dog}、ds_b: {0:cat}
    → unresolved_conflicts に格納され、自動解決されないこと。
    """
    ds_a = _make_dataset_with_custom_map({0: "dog"})
    ds_b = _make_dataset_with_custom_map({0: "cat"})

    result = merge_datasets_dry_run([ds_a, ds_b])

    assert len(result.unresolved_conflicts) == 1
    conflict = result.unresolved_conflicts[0]
    assert conflict.conflicting_id == 0
    # name_from_dataset_a と name_from_dataset_b は dog と cat（順序は実装依存）
    both_names = {conflict.name_from_dataset_a, conflict.name_from_dataset_b}
    assert both_names == {"dog", "cat"}


def test_merge_dry_run_pattern1_not_misdetected_as_pattern3():
    """
    パターン1（同名・異ID）をパターン3と誤検出しないこと。
    ds_a: {0:dog, 1:cat}、ds_b: {0:cat, 1:dog}
    → 同じ ID に別の名前が存在するが、両名前とも両データセットに存在するので
      パターン3ではない（unresolved_conflicts が空）。
    """
    ds_a = _make_dataset_with_custom_map({0: "dog", 1: "cat"})
    ds_b = _make_dataset_with_custom_map({0: "cat", 1: "dog"})

    result = merge_datasets_dry_run([ds_a, ds_b])

    assert result.unresolved_conflicts == [], (
        "パターン1（名前が両方に存在する）をパターン3と誤検出してはならない"
    )


def test_merge_dry_run_records_preserved():
    """マージ後、全データセットのレコード合計数が保持されること。"""
    ds_a = _make_dataset(["dog"])
    ds_a.records = [_make_record(0, "dog", "001"), _make_record(0, "dog", "002")]
    ds_b = _make_dataset(["cat"])
    ds_b.records = [_make_record(0, "cat", "003")]

    result = merge_datasets_dry_run([ds_a, ds_b])

    assert len(result.merged_dataset.records) == 3


# ---------------------------------------------------------------------------
# マージ処理テスト（第2段階 resolve）
# ---------------------------------------------------------------------------

def test_merge_resolve_applies_user_choice():
    """
    merge_datasets_resolve() がユーザーの選択（ConflictResolution）を正しく適用すること。
    ds_a: {0:dog}、ds_b: {0:cat} → ユーザーが "dog" を採択
    → 最終 Dataset に "cat" が存在せず、"dog" のみになること。
    """
    ds_a = _make_dataset_with_custom_map({0: "dog"})
    ds_b = _make_dataset_with_custom_map({0: "cat"})

    # ds_b に cat のレコードを追加
    ds_b.records.append(_make_record(0, "cat", "002"))

    dry = merge_datasets_dry_run([ds_a, ds_b])
    assert len(dry.unresolved_conflicts) == 1

    resolution = ConflictResolution(
        conflict=dry.unresolved_conflicts[0],
        chosen_name="dog",
    )
    final = merge_datasets_resolve(dry, [resolution])

    # "cat" が "dog" に統合されていること
    assert "cat" not in final.class_map.name_to_id
    assert "dog" in final.class_map.name_to_id

    # レコードのアノテーションが "dog" になっていること
    for record in final.records:
        for ann in record.annotations:
            assert ann.class_name == "dog"
            assert ann.class_id == final.class_map.name_to_id["dog"]


def test_merge_resolve_no_conflict_returns_dry_dataset():
    """競合がない場合（パターン2）、resolve は dry_run の Dataset をそのまま返すこと。
    ds_a: {dog, cat}、ds_b: {dog, car} → 共通クラス dog がありパターン3なし。
    """
    ds_a = _make_dataset(["dog", "cat"])
    ds_b = _make_dataset(["dog", "car"])

    dry = merge_datasets_dry_run([ds_a, ds_b])
    assert dry.unresolved_conflicts == []

    final = merge_datasets_resolve(dry, [])

    assert set(final.class_map.name_to_id.keys()) == {"dog", "cat", "car"}
