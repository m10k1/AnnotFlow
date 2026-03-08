"""
Label Studio パーサーのテスト（Phase 1: 実装）

restore_original_filename() と find_image_file() は Phase 0 から実装済み。
LabelStudioParser のテストを Phase 1 で追加する。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from parsers.label_studio_parser import (
    LabelStudioParser,
    find_image_file,
    restore_original_filename,
)


# ---------------------------------------------------------------------------
# restore_original_filename() のテスト（Phase 0 から維持）
# ---------------------------------------------------------------------------

def test_restore_original_filename_removes_prefix():
    assert restore_original_filename("abcdef12-original_name.jpg") == "original_name.jpg"


def test_restore_original_filename_no_prefix():
    assert restore_original_filename("original_name.jpg") == "original_name.jpg"


def test_restore_original_filename_hyphen_in_name():
    assert restore_original_filename("abcdef12-my-photo-001.jpg") == "my-photo-001.jpg"


def test_restore_original_filename_uppercase_hex():
    assert restore_original_filename("ABCDEF12-image.png") == "image.png"


def test_restore_original_filename_short_prefix():
    assert restore_original_filename("abcdef1-image.jpg") == "abcdef1-image.jpg"


def test_restore_original_filename_non_hex_prefix():
    assert restore_original_filename("xxxxxxxx-image.jpg") == "xxxxxxxx-image.jpg"


# ---------------------------------------------------------------------------
# find_image_file() のテスト（Phase 0 から維持）
# ---------------------------------------------------------------------------

def test_find_image_file_by_original_name(tmp_path: Path):
    (tmp_path / "original_name.jpg").write_bytes(b"fake")
    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "original_name.jpg"


def test_find_image_file_by_id_prefixed_name(tmp_path: Path):
    (tmp_path / "abcdef12-original_name.jpg").write_bytes(b"fake")
    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "abcdef12-original_name.jpg"


def test_find_image_file_not_found_returns_none(tmp_path: Path):
    assert find_image_file(tmp_path, "abcdef12-nonexistent.jpg") is None


def test_find_image_file_prefers_original_name(tmp_path: Path):
    (tmp_path / "original_name.jpg").write_bytes(b"original")
    (tmp_path / "abcdef12-original_name.jpg").write_bytes(b"id prefixed")
    result = find_image_file(tmp_path, "abcdef12-original_name.jpg")
    assert result == tmp_path / "original_name.jpg"


# ---------------------------------------------------------------------------
# LabelStudioParser のテスト（Phase 1）
# ---------------------------------------------------------------------------

@pytest.fixture
def img_dir(tmp_path: Path, label_studio_sample_path: Path) -> Path:
    """サンプル JSON に対応する画像ファイルを tmp_path に作成して返す。"""
    d = tmp_path / "images"
    d.mkdir()
    Image.new("RGB", (640, 480)).save(d / "image_001.jpg")
    Image.new("RGB", (800, 600)).save(d / "image_002.jpg")
    Image.new("RGB", (640, 480)).save(d / "image_003.jpg")
    return d


def test_label_studio_percent_to_absolute_conversion(img_dir, label_studio_sample_path):
    """Label Studioの%座標が絶対座標に正しく変換されること。
    item1: x=1.5625, y=4.1667, w=14.0625, h=27.0833, W=640, H=480
    -> x_min=10.0, y_min=20.0, x_max=100.0, y_max=150.0
    """
    parser = LabelStudioParser()
    dataset, _ = parser.parse(label_studio_sample_path, img_dir)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    dog_ann = next(a for a in record.annotations if a.class_name == "dog")

    assert dog_ann.bbox.x_min == pytest.approx(10.0, abs=0.5)
    assert dog_ann.bbox.y_min == pytest.approx(20.0, abs=0.5)
    assert dog_ann.bbox.x_max == pytest.approx(100.0, abs=0.5)
    assert dog_ann.bbox.y_max == pytest.approx(150.0, abs=0.5)


def test_label_studio_cancelled_annotation_excluded(img_dir, label_studio_sample_path):
    """was_cancelled: true のアノテーションが除外されること。
    item3 (cafebabe-image_003.jpg) は was_cancelled=true なので annotations が空になること。
    """
    parser = LabelStudioParser()
    dataset, _ = parser.parse(label_studio_sample_path, img_dir)

    record_003 = next(
        (r for r in dataset.records if "image_003" in r.original_filename), None
    )
    if record_003 is not None:
        assert len(record_003.annotations) == 0
    # キャンセル済みアノテーションしかない画像はレコードに含まれないか、含まれても annotations が空


def test_label_studio_multiple_annotators_detected(label_studio_sample_path):
    """複数アノテーターが存在する場合にscan_annotators()で検出されること。"""
    parser = LabelStudioParser()
    annotators = parser.scan_annotators(label_studio_sample_path)

    emails = {a["email"] for a in annotators}
    assert "annotator_a@example.com" in emails
    assert "annotator_b@example.com" in emails


def test_label_studio_original_filename_stored_without_prefix(img_dir, label_studio_sample_path):
    """ImageRecord.original_filename に接頭辞なしのファイル名が格納されること。"""
    parser = LabelStudioParser()
    dataset, _ = parser.parse(label_studio_sample_path, img_dir)

    for record in dataset.records:
        # 接頭辞（8文字16進数+ハイフン）が含まれていないこと
        assert not record.original_filename.startswith("abcdef")
        assert not record.original_filename.startswith("deadbeef")
        assert not record.original_filename.startswith("cafebabe")


def test_label_studio_missing_image_skipped(tmp_path, label_studio_sample_path):
    """対応する画像ファイルが存在しないレコードがスキップされること。"""
    empty_dir = tmp_path / "empty_images"
    empty_dir.mkdir()

    parser = LabelStudioParser()
    dataset, skipped = parser.parse(label_studio_sample_path, empty_dir)

    assert len(dataset.records) == 0
    assert len(skipped) > 0


def test_label_studio_single_annotator_used_as_is(img_dir, label_studio_sample_path):
    """was_cancelled でないアノテーションが1件のみの画像はそのまま採用されること。
    image_001 は annotator_a のみの非キャンセルアノテーションを持つ。
    """
    parser = LabelStudioParser()
    dataset, _ = parser.parse(label_studio_sample_path, img_dir)

    record = next(r for r in dataset.records if "image_001" in r.original_filename)
    # dog と cat の2つの bbox が含まれていること
    class_names = {a.class_name for a in record.annotations}
    assert "dog" in class_names
    assert "cat" in class_names
