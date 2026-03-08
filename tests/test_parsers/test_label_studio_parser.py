"""
Label Studio パーサーのテスト（Phase 0: スケルトン）

restore_original_filename() と find_image_file() のユニットテストは
実装済みの関数なのでスケルトンではなく実際のテストコードを含む。
その他のテストは Phase 1 で実装する。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from parsers.label_studio_parser import find_image_file, restore_original_filename


# ---------------------------------------------------------------------------
# restore_original_filename() のテスト（Phase 0 で検証可能）
# ---------------------------------------------------------------------------

def test_restore_original_filename_removes_prefix():
    """8文字16進数+ハイフンの接頭辞が除去されて元のファイル名が復元されること"""
    assert restore_original_filename("abcdef12-original_name.jpg") == "original_name.jpg"


def test_restore_original_filename_no_prefix():
    """接頭辞がない場合はそのままの値が返されること"""
    assert restore_original_filename("original_name.jpg") == "original_name.jpg"


def test_restore_original_filename_hyphen_in_name():
    """元のファイル名自体にハイフンが含まれる場合でも正しく復元されること"""
    assert restore_original_filename("abcdef12-my-photo-001.jpg") == "my-photo-001.jpg"


def test_restore_original_filename_uppercase_hex():
    """接頭辞が大文字16進数でも正しく除去されること"""
    assert restore_original_filename("ABCDEF12-image.png") == "image.png"


def test_restore_original_filename_short_prefix():
    """7文字以下の16進数+ハイフンはIDとみなさず、そのまま返すこと"""
    # 7文字の16進数はパターンに一致しないためそのまま返す
    assert restore_original_filename("abcdef1-image.jpg") == "abcdef1-image.jpg"


def test_restore_original_filename_non_hex_prefix():
    """16進数以外の文字を含む接頭辞はIDとみなさず、そのまま返すこと"""
    assert restore_original_filename("xxxxxxxx-image.jpg") == "xxxxxxxx-image.jpg"


# ---------------------------------------------------------------------------
# find_image_file() のテスト（Phase 0 で検証可能）
# ---------------------------------------------------------------------------

def test_find_image_file_by_original_name(tmp_path: Path):
    """復元した元のファイル名で画像が正しく見つかること"""
    # 元のファイル名で画像を作成
    (tmp_path / "original_name.jpg").write_bytes(b"fake image data")

    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "original_name.jpg"


def test_find_image_file_by_id_prefixed_name(tmp_path: Path):
    """元のファイル名で見つからない場合にID付きファイル名でフォールバック検索されること"""
    # ID付きファイル名で画像を作成（Label Studio がそのまま保存した場合）
    (tmp_path / "abcdef12-original_name.jpg").write_bytes(b"fake image data")

    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "abcdef12-original_name.jpg"


def test_find_image_file_not_found_returns_none(tmp_path: Path):
    """どちらの名前でも見つからない場合にNoneが返されること"""
    result = find_image_file(tmp_path, "abcdef12-nonexistent.jpg")
    assert result is None


def test_find_image_file_prefers_original_name(tmp_path: Path):
    """元のファイル名とID付きファイル名の両方が存在する場合、元のファイル名が優先されること"""
    (tmp_path / "original_name.jpg").write_bytes(b"original")
    (tmp_path / "abcdef12-original_name.jpg").write_bytes(b"id prefixed")

    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "original_name.jpg"


# ---------------------------------------------------------------------------
# LabelStudioParser のテスト（Phase 1 で実装）
# ---------------------------------------------------------------------------

def test_label_studio_percent_to_absolute_conversion():
    """Label Studioの%座標が絶対座標に正しく変換されること"""
    pytest.skip("Phase 1 で実装")


def test_label_studio_cancelled_annotation_excluded():
    """was_cancelled: true のアノテーションが除外されること"""
    pytest.skip("Phase 1 で実装")


def test_label_studio_multiple_annotators_detected():
    """複数アノテーターが存在する場合に検出され、annotator_listが返されること"""
    pytest.skip("Phase 1 で実装")


def test_label_studio_original_filename_stored_without_prefix():
    """`ImageRecord.original_filename` に接頭辞なしのファイル名が格納されること"""
    pytest.skip("Phase 1 で実装")


def test_label_studio_missing_image_skipped():
    """対応する画像ファイルが存在しないレコードがスキップされること"""
    pytest.skip("Phase 1 で実装")


def test_label_studio_single_annotator_used_as_is():
    """was_cancelled でないアノテーションが1件のみの場合はそのまま採用されること"""
    pytest.skip("Phase 1 で実装")
