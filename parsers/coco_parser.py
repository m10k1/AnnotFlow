"""
COCO JSON フォーマットのパーサー

座標変換: COCO の [x, y, w, h]（絶対座標）→ BoundingBox [x_min, y_min, x_max, y_max]
クラスID: category_id は無視し、name を使って ClassMap に登録・0始まりで再採番
"""
from __future__ import annotations

import json
from pathlib import Path

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from parsers.base_parser import BaseParser

# 試みる画像拡張子（優先度順）
_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


def _find_image(image_folder: Path, file_name: str) -> Path | None:
    """file_name に対応する画像ファイルを image_folder から検索する。"""
    candidate = image_folder / file_name
    if candidate.exists():
        return candidate
    # 拡張子を変えて検索
    stem = Path(file_name).stem
    for ext in _IMAGE_EXTENSIONS:
        c = image_folder / (stem + ext)
        if c.exists():
            return c
    return None


class CocoParser(BaseParser):
    """COCO JSON フォーマットのパーサー。"""

    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        COCO JSON を読み込んで Dataset に変換する。

        Returns:
            (Dataset, skipped_records) のタプル
        """
        with annotation_path.open(encoding="utf-8") as f:
            coco = json.load(f)

        # カテゴリを名前順に ClassMap へ登録（COCO の category_id は無視）
        # 仕様: カテゴリ名の初出順で 0始まりに再採番する
        category_id_to_name: dict[int, str] = {
            cat["id"]: cat["name"] for cat in coco.get("categories", [])
        }
        # 名前リストを categories の並び順で作成
        ordered_names: list[str] = []
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in ordered_names:
                ordered_names.append(name)
        class_map = ClassMap.from_names(ordered_names)

        # image_id → image_info のマップ
        image_info: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}

        # image_id → annotations のマップ
        ann_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            ann_by_image.setdefault(ann["image_id"], []).append(ann)

        records: list[ImageRecord] = []
        skipped: list[dict] = []

        for img_id, img_info in image_info.items():
            file_name = img_info["file_name"]
            img_path = _find_image(image_folder, file_name)

            if img_path is None:
                skipped.append({"file_name": file_name, "reason": "image_not_found"})
                continue

            annotations: list[Annotation] = []
            for raw_ann in ann_by_image.get(img_id, []):
                cat_id = raw_ann["category_id"]
                name = category_id_to_name.get(cat_id)
                if name is None:
                    continue

                x, y, w, h = raw_ann["bbox"]
                bbox = BoundingBox(
                    x_min=float(x),
                    y_min=float(y),
                    x_max=float(x + w),
                    y_max=float(y + h),
                )
                annotations.append(Annotation(
                    class_id=class_map.name_to_id[name],
                    class_name=name,
                    bbox=bbox,
                    segmentation=raw_ann.get("segmentation") or None,
                ))

            records.append(ImageRecord(
                image_id=str(img_id),
                original_filename=file_name,
                file_path=str(img_path),
                width=img_info["width"],
                height=img_info["height"],
                annotations=annotations,
            ))

        dataset = Dataset(records=records, class_map=class_map, source_format="coco")
        return dataset, skipped
