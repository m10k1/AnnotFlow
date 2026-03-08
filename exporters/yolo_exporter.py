"""
YOLO フォーマットのエクスポーター

split 別に train/images/, train/labels/, val/images/, val/labels/,
test/images/, test/labels/ および data.yaml を出力する。
BBox のみを対象とし、segmentation は出力しない。
"""
from __future__ import annotations

import shutil
from pathlib import Path

from core.dataset import Dataset, ImageRecord
from exporters.base_exporter import BaseExporter


def _absolute_to_yolo(
    x_min: float, y_min: float, x_max: float, y_max: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """
    絶対座標 BoundingBox → YOLO 相対座標 (cx, cy, w, h) に変換する。

    変換式（仕様書より）:
        cx = (x_min + x_max) / 2 / image_width
        cy = (y_min + y_max) / 2 / image_height
        w  = (x_max - x_min) / image_width
        h  = (y_max - y_min) / image_height
    """
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


class YoloExporter(BaseExporter):
    """YOLO フォーマットのエクスポーター。"""

    def export(
        self,
        dataset: Dataset,
        output_dir: Path,
        copy_images: bool = True,
    ) -> None:
        """
        Dataset を YOLO 形式でエクスポートする。

        出力ディレクトリ構造:
            output_dir/
                data.yaml
                train/
                    images/  (copy_images=True の場合)
                    labels/
                val/
                    images/  (copy_images=True の場合)
                    labels/
                test/         (test レコードが存在する場合)
                    images/
                    labels/

        Args:
            dataset: エクスポート対象のデータセット（split が設定済みであること）
            output_dir: 出力先ディレクトリ
            copy_images: 画像ファイルもコピーするか否か
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # split 別のレコードを収集
        split_records: dict[str, list[ImageRecord]] = {}
        for record in dataset.records:
            if record.split is None:
                continue
            split_records.setdefault(record.split, []).append(record)

        # 各 split のディレクトリを作成してラベルを出力
        for split_name, records in split_records.items():
            labels_dir = output_dir / split_name / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            if copy_images:
                images_dir = output_dir / split_name / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

            for record in records:
                stem = Path(record.original_filename).stem
                label_path = labels_dir / f"{stem}.txt"

                # YOLO ラベルファイルを生成（BBox のみ、segmentation は出力しない）
                lines: list[str] = []
                for ann in record.annotations:
                    bbox = ann.bbox
                    cx, cy, w, h = _absolute_to_yolo(
                        bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                        record.width, record.height,
                    )
                    lines.append(f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                label_path.write_text("\n".join(lines), encoding="utf-8")

                # 画像のコピー
                if copy_images and record.file_path:
                    src = Path(record.file_path)
                    if src.exists():
                        dst = output_dir / split_name / "images" / record.original_filename
                        shutil.copy2(src, dst)

        # data.yaml を生成
        self._write_data_yaml(output_dir, split_records, dataset)

    def _write_data_yaml(
        self,
        output_dir: Path,
        split_records: dict[str, list[ImageRecord]],
        dataset: Dataset,
    ) -> None:
        """data.yaml を生成する。"""
        # クラス名を ID 順で取得
        num_classes = len(dataset.class_map.id_to_name)
        class_names = [
            dataset.class_map.id_to_name[i] for i in range(num_classes)
        ]

        lines = [f"nc: {num_classes}", "names:"]
        for name in class_names:
            lines.append(f"  - {name}")

        # 各 split のパスを記載
        for split_name in ("train", "val", "test"):
            if split_name in split_records:
                lines.append(f"{split_name}: {split_name}/images")

        (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
