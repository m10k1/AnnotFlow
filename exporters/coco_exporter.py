"""
COCO JSON フォーマットのエクスポーター

split 別に train.json / val.json / test.json を出力する。
- 内部の絶対座標 BBox を COCO 形式 [x, y, w, h] に変換する
- 内部の 0始まり class_id を COCO 形式の 1始まり category_id に変換する
- segmentation フィールドは保持していた値をそのまま出力する（変換・加工なし）
- original_filename を file_name に使用する
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from core.dataset import Dataset, ImageRecord
from exporters.base_exporter import BaseExporter


class CocoExporter(BaseExporter):
    """COCO JSON フォーマットのエクスポーター。"""

    def export(
        self,
        dataset: Dataset,
        output_dir: Path,
        copy_images: bool = True,
    ) -> None:
        """
        Dataset を COCO JSON 形式でエクスポートする。

        split が設定されていないレコードは "train" として扱う。
        split 別に train.json / val.json / test.json を生成する。

        Args:
            dataset: エクスポート対象のデータセット（split が設定済みであること）
            output_dir: 出力先ディレクトリ
            copy_images: 画像ファイルもコピーするか否か
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # split ごとにレコードをグループ化する（split=None は "train" 扱い）
        split_groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in dataset.records:
            split_key = record.split if record.split else "train"
            split_groups[split_key].append(record)

        # COCO カテゴリリストを構築（内部 0始まり → COCO 1始まり）
        categories = [
            {
                "id": class_id + 1,
                "name": name,
                "supercategory": "none",
            }
            for class_id, name in sorted(dataset.class_map.id_to_name.items())
        ]

        # split ごとに JSON を出力する
        for split_name, records in split_groups.items():
            coco_data = self._build_coco_json(records, categories)
            out_file = output_dir / f"{split_name}.json"
            out_file.write_text(
                json.dumps(coco_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # 画像をコピーする場合
            if copy_images:
                images_dir = output_dir / split_name / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                for record in records:
                    src = Path(record.file_path)
                    if src.exists():
                        shutil.copy2(src, images_dir / record.original_filename)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _build_coco_json(
        self,
        records: list[ImageRecord],
        categories: list[dict],
    ) -> dict:
        """
        指定 split のレコードから COCO JSON 構造を組み立てる。

        Args:
            records: 対象レコードのリスト
            categories: カテゴリリスト（1始まりのIDを持つ）

        Returns:
            COCO 形式の辞書
        """
        info = {
            "description": "AnnotFlow export",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "AnnotFlow",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        }

        images = []
        annotations = []
        ann_id = 1  # COCO の annotation id は 1始まり

        for img_id, record in enumerate(records, start=1):
            # images エントリ
            images.append({
                "id": img_id,
                "file_name": record.original_filename,
                "width": record.width,
                "height": record.height,
            })

            # annotations エントリ
            for ann in record.annotations:
                bbox = ann.bbox
                # 内部絶対座標 → COCO [x_min, y_min, width, height]
                coco_x = bbox.x_min
                coco_y = bbox.y_min
                coco_w = bbox.x_max - bbox.x_min
                coco_h = bbox.y_max - bbox.y_min

                ann_entry: dict = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann.class_id + 1,   # 0始まり → 1始まり
                    "bbox": [coco_x, coco_y, coco_w, coco_h],
                    "area": coco_w * coco_h,
                    "iscrowd": 0,
                    # segmentation は加工せずそのまま出力（None の場合は空リスト）
                    "segmentation": ann.segmentation if ann.segmentation is not None else [],
                }

                annotations.append(ann_entry)
                ann_id += 1

        return {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
