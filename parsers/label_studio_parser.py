"""
Label Studio JSON フォーマットのパーサー
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from parsers.base_parser import BaseParser


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def restore_original_filename(file_upload: str) -> str:
    """
    Label Studio のファイルアップロードフィールドから ID 接頭辞を除去して
    元のファイル名を復元する。

    Label Studio の付与形式: "{8文字の16進数}-{元のファイル名}"
    例: "abcdef12-original_name.jpg" → "original_name.jpg"
    """
    pattern = r'^[0-9a-f]{8}-(.+)$'
    match = re.match(pattern, file_upload, re.IGNORECASE)
    if match:
        return match.group(1)
    return file_upload


def find_image_file(image_folder: Path, file_upload: str) -> Path | None:
    """
    画像ファイルを以下の順序で検索する:
    1. 復元した元のファイル名で検索
    2. 見つからなければ file_upload の値（ID付き）で検索
    3. どちらも見つからなければ None を返す（スキップ対象）
    """
    original_name = restore_original_filename(file_upload)

    candidate = image_folder / original_name
    if candidate.exists():
        return candidate

    candidate = image_folder / file_upload
    if candidate.exists():
        return candidate

    return None


# ---------------------------------------------------------------------------
# パーサー本体
# ---------------------------------------------------------------------------

class LabelStudioParser(BaseParser):
    """Label Studio JSON フォーマットのパーサー。"""

    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        Label Studio JSON を読み込んで Dataset に変換する。

        複数アノテーターがいる場合は最初の非キャンセルアノテーションを採用する。

        Returns:
            (Dataset, skipped_records) のタプル
        """
        items: list[dict] = json.loads(annotation_path.read_text(encoding="utf-8"))

        class_map = ClassMap.from_names([])
        records: list[ImageRecord] = []
        skipped: list[dict] = []

        for item in items:
            file_upload: str = item.get("file_upload", "")
            img_path = find_image_file(image_folder, file_upload)

            if img_path is None:
                skipped.append({"file_upload": file_upload, "reason": "image_not_found"})
                continue

            original_filename = restore_original_filename(file_upload)

            # was_cancelled でない最初のアノテーションを採用
            chosen_annotation: dict | None = None
            for ann in item.get("annotations", []):
                if not ann.get("was_cancelled", False):
                    chosen_annotation = ann
                    break

            annotations: list[Annotation] = []
            if chosen_annotation is not None:
                for result in chosen_annotation.get("result", []):
                    if result.get("type") != "rectanglelabels":
                        continue
                    value = result["value"]
                    labels_list: list[str] = value.get("rectanglelabels", [])
                    if not labels_list:
                        continue
                    class_name = labels_list[0]
                    class_id = class_map.get_or_add(class_name)

                    orig_w = result.get("original_width", 1)
                    orig_h = result.get("original_height", 1)
                    x_pct = value["x"]
                    y_pct = value["y"]
                    w_pct = value["width"]
                    h_pct = value["height"]

                    x_min = x_pct / 100.0 * orig_w
                    y_min = y_pct / 100.0 * orig_h
                    x_max = (x_pct + w_pct) / 100.0 * orig_w
                    y_max = (y_pct + h_pct) / 100.0 * orig_h

                    annotations.append(Annotation(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    ))

            with Image.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

            records.append(ImageRecord(
                image_id=str(item.get("id", "")),
                original_filename=original_filename,
                file_path=str(img_path),
                width=img_w,
                height=img_h,
                annotations=annotations,
            ))

        dataset = Dataset(records=records, class_map=class_map, source_format="label_studio")
        return dataset, skipped

    def scan_annotators(self, annotation_path: Path) -> list[dict]:
        """
        アノテーションファイルを事前スキャンしてアノテーター一覧を返す。

        Returns:
            アノテーター情報のリスト: [{"id": 1, "email": "user@example.com"}, ...]
        """
        items: list[dict] = json.loads(annotation_path.read_text(encoding="utf-8"))
        seen_ids: set[int] = set()
        annotators: list[dict] = []

        for item in items:
            for ann in item.get("annotations", []):
                if ann.get("was_cancelled", False):
                    continue
                completed_by = ann.get("completed_by", {})
                uid = completed_by.get("id")
                if uid is not None and uid not in seen_ids:
                    seen_ids.add(uid)
                    annotators.append({
                        "id": uid,
                        "email": completed_by.get("email", ""),
                    })

        return annotators
