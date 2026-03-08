"""
YOLO フォーマットのパーサー

座標変換: YOLO の相対座標 [cx, cy, w, h]（0.0〜1.0）→ BoundingBox [x_min, y_min, x_max, y_max]
クラスID: data.yaml または classes.txt の行順をそのまま 0始まりで ClassMap に登録
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord
from parsers.base_parser import BaseParser

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


def _yolo_to_absolute(cx: float, cy: float, w: float, h: float,
                      img_w: int, img_h: int) -> BoundingBox:
    """YOLO 相対座標 → 絶対座標 BoundingBox に変換。"""
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _load_class_names(annotation_path: Path) -> list[str]:
    """data.yaml または classes.txt からクラス名リストを読み込む。"""
    suffix = annotation_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        # 簡易 YAML パース（PyYAML 非依存）
        names: list[str] = []
        in_names = False
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip())
                elif stripped and not stripped.startswith("#"):
                    break  # names ブロック終了
        return names
    else:
        # classes.txt: 1行1クラス名
        return [
            line.strip()
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _find_image(image_folder: Path, stem: str) -> Path | None:
    """ステム名に対応する画像ファイルを image_folder から検索する。"""
    for ext in _IMAGE_EXTENSIONS:
        c = image_folder / (stem + ext)
        if c.exists():
            return c
    return None


class YoloParser(BaseParser):
    """YOLO フォーマットのパーサー。"""

    def parse(
        self,
        annotation_path: Path,
        image_folder: Path,
    ) -> tuple[Dataset, list[dict]]:
        """
        YOLO フォーマットを読み込んで Dataset に変換する。

        Args:
            annotation_path: classes.txt または data.yaml のパス
            image_folder: 画像フォルダのパス

        Returns:
            (Dataset, skipped_records) のタプル
        """
        class_names = _load_class_names(annotation_path)
        class_map = ClassMap.from_names(class_names)

        # ラベルディレクトリは annotation_path の親の labels/ サブディレクトリ
        labels_dir = annotation_path.parent / "labels"

        records: list[ImageRecord] = []
        skipped: list[dict] = []

        for txt_path in sorted(labels_dir.glob("*.txt")):
            stem = txt_path.stem
            img_path = _find_image(image_folder, stem)

            if img_path is None:
                skipped.append({"file_name": stem, "reason": "image_not_found"})
                continue

            with Image.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

            annotations: list[Annotation] = []
            for line in txt_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bbox = _yolo_to_absolute(cx, cy, w, h, img_w, img_h)
                name = class_map.id_to_name.get(class_id, f"class_{class_id}")
                annotations.append(Annotation(
                    class_id=class_id,
                    class_name=name,
                    bbox=bbox,
                ))

            records.append(ImageRecord(
                image_id=stem,
                original_filename=img_path.name,
                file_path=str(img_path),
                width=img_w,
                height=img_h,
                annotations=annotations,
            ))

        dataset = Dataset(records=records, class_map=class_map, source_format="yolo")
        return dataset, skipped
