"""
内部データモデル（共通形式）

すべてのフォーマットはインポート後にこのモジュールの共通モデルへ変換して保持する。
PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 基本データクラス
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    バウンディングボックス。
    内部では常にピクセル単位の絶対座標（x_min, y_min, x_max, y_max）で保持する。
    """
    x_min: float  # ピクセル単位の絶対座標
    y_min: float  # ピクセル単位の絶対座標
    x_max: float  # ピクセル単位の絶対座標
    y_max: float  # ピクセル単位の絶対座標


@dataclass
class Annotation:
    """
    1つのアノテーション（バウンディングボックス＋クラス情報）。
    segmentation フィールドはCOCO形式のデータをそのまま保持するためのもので、
    変換・座標計算・バリデーションは一切行わない。
    """
    class_id: int        # 内部では0始まりに正規化済み
    class_name: str
    bbox: BoundingBox    # 変換・分割ロジックが参照するのはここだけ
    segmentation: list | None = None  # 保持するが加工しない。COCOエクスポート時にそのまま出力


@dataclass
class ImageRecord:
    """
    1枚の画像とそれに紐づくアノテーションの集合。
    """
    image_id: str
    original_filename: str      # 元ファイル名を必ず保持する（Label Studioの接頭辞を除去済み）
    file_path: str              # 実際のファイルパス
    width: int
    height: int
    annotations: list[Annotation]
    split: str | None = None          # "train" / "val" / "test"
    embedding: list | None = None     # CLIPベクトル（オプション、Phase 3で使用）
    cluster_id: int | None = None     # CLIPクラスタリング結果（オプション、Phase 3で使用）


# ---------------------------------------------------------------------------
# クラスIDとクラス名のマッピング管理
# ---------------------------------------------------------------------------

@dataclass
class ClassMap:
    """
    クラスIDとクラス名の双方向マッピング。
    内部では常に0始まりのIDを使用する。
    パーサーはクラス名のみを渡し、IDの採番は ClassMap が行う。
    """
    id_to_name: dict[int, str]   # {0: "dog", 1: "cat", 2: "car"}
    name_to_id: dict[str, int]   # {"dog": 0, "cat": 1, "car": 2}

    @classmethod
    def from_names(cls, names: list[str]) -> "ClassMap":
        """名前リストから0始まりで自動採番してClassMapを生成する。"""
        id_to_name = {i: name for i, name in enumerate(names)}
        name_to_id = {name: i for i, name in enumerate(names)}
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)

    def get_or_add(self, name: str) -> int:
        """クラス名が未登録なら追加して新しいIDを返す。登録済みならそのIDを返す。"""
        if name not in self.name_to_id:
            new_id = len(self.id_to_name)
            self.id_to_name[new_id] = name
            self.name_to_id[name] = new_id
        return self.name_to_id[name]


@dataclass
class Dataset:
    """
    複数の ImageRecord と、クラスマッピングを持つデータセット全体。
    """
    records: list[ImageRecord]
    class_map: ClassMap          # class_namesの代わりにClassMapで一元管理
    source_format: str


# ---------------------------------------------------------------------------
# マージ処理用データクラス
# ---------------------------------------------------------------------------

@dataclass
class MergeConflict:
    """
    ユーザーによる解決が必要な競合（パターン3：同ID・異クラス名 専用）。
    """
    conflicting_id: int
    name_from_dataset_a: str
    name_from_dataset_b: str
    dataset_a_label: str   # 識別用ラベル（例: "ファイルA"）
    dataset_b_label: str


@dataclass
class MergeDryRunResult:
    """
    第1段階マージの結果：自動解決済みのデータ＋未解決の競合リスト。
    """
    merged_dataset: Dataset                      # パターン1・2は自動解決済み
    unresolved_conflicts: list[MergeConflict]    # パターン3のみ格納
    conflict_report: list[dict]                  # 変更ログ（CSVエクスポート用）


@dataclass
class ConflictResolution:
    """パターン3に対するユーザーの解決指示。"""
    conflict: MergeConflict
    chosen_name: str  # ユーザーが正として選んだクラス名


# ---------------------------------------------------------------------------
# マージ関数
# ---------------------------------------------------------------------------

def merge_datasets_dry_run(
    datasets: list[Dataset],
    dataset_labels: list[str] | None = None,
) -> MergeDryRunResult:
    """
    第1段階マージ。UIは一切呼び出さない。

    - パターン1（同名・異ID）: 名前を正として自動解決
    - パターン2（片方のみ存在）: 和集合として自動解決
    - パターン3（同ID・異名）: 解決せず unresolved_conflicts に積む

    Args:
        datasets: マージ対象のDatasetリスト
        dataset_labels: 各データセットの識別用ラベル（省略時は "Dataset_0" 等）

    Returns:
        MergeDryRunResult: 自動解決済みデータと未解決競合リスト
    """
    if dataset_labels is None:
        dataset_labels = [f"Dataset_{i}" for i in range(len(datasets))]

    conflict_report: list[dict] = []
    unresolved_conflicts: list[MergeConflict] = []

    # 全データセットのクラス名の和集合を名前ベースで収集
    all_names: list[str] = []
    for ds in datasets:
        for name in ds.class_map.name_to_id:
            if name not in all_names:
                all_names.append(name)

    # 新しいClassMapを0始まりで構築
    merged_class_map = ClassMap.from_names(all_names)

    # 全レコードを収集し、class_idを新ClassMapに基づいて再採番
    merged_records: list[ImageRecord] = []
    for ds_idx, ds in enumerate(datasets):
        for record in ds.records:
            new_annotations: list[Annotation] = []
            for ann in record.annotations:
                old_id = ann.class_id
                name = ds.class_map.id_to_name.get(old_id)
                if name is None:
                    # クラスIDに対応する名前が存在しない場合はスキップ
                    continue
                new_id = merged_class_map.name_to_id[name]

                # 変更があれば conflict_report に記録
                if old_id != new_id:
                    conflict_report.append({
                        "dataset": dataset_labels[ds_idx],
                        "class_name": name,
                        "old_id": old_id,
                        "new_id": new_id,
                    })

                new_annotations.append(Annotation(
                    class_id=new_id,
                    class_name=name,
                    bbox=ann.bbox,
                    segmentation=ann.segmentation,
                ))

            # ImageRecord をコピーして新しいアノテーションを設定
            merged_records.append(ImageRecord(
                image_id=record.image_id,
                original_filename=record.original_filename,
                file_path=record.file_path,
                width=record.width,
                height=record.height,
                annotations=new_annotations,
                split=record.split,
                embedding=record.embedding,
                cluster_id=record.cluster_id,
            ))

    # パターン3の検出（同ID・異なるクラス名）
    # 各データセット間で同じIDに異なる名前が割り当てられているケースを探す
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds_a = datasets[i]
            ds_b = datasets[j]
            for id_a, name_a in ds_a.class_map.id_to_name.items():
                name_b = ds_b.class_map.id_to_name.get(id_a)
                if name_b is not None and name_b != name_a:
                    # 同ID・異なる名前（パターン3）
                    # ただし、どちらの名前も and_names に含まれているため
                    # 自動解決はせず unresolved_conflicts に積む
                    conflict = MergeConflict(
                        conflicting_id=id_a,
                        name_from_dataset_a=name_a,
                        name_from_dataset_b=name_b,
                        dataset_a_label=dataset_labels[i],
                        dataset_b_label=dataset_labels[j],
                    )
                    # 重複チェック（同じ競合を重複登録しない）
                    already_exists = any(
                        c.conflicting_id == id_a
                        and c.name_from_dataset_a == name_a
                        and c.name_from_dataset_b == name_b
                        for c in unresolved_conflicts
                    )
                    if not already_exists:
                        unresolved_conflicts.append(conflict)

    merged_dataset = Dataset(
        records=merged_records,
        class_map=merged_class_map,
        source_format="merged",
    )

    return MergeDryRunResult(
        merged_dataset=merged_dataset,
        unresolved_conflicts=unresolved_conflicts,
        conflict_report=conflict_report,
    )


def merge_datasets_resolve(
    dry_run_result: MergeDryRunResult,
    resolutions: list[ConflictResolution],
) -> Dataset:
    """
    第2段階マージ。UIは一切呼び出さない。
    ユーザーの選択（resolutions）を適用して最終的なDatasetを返す。

    Args:
        dry_run_result: 第1段階マージの結果
        resolutions: パターン3の各競合に対するユーザーの解決指示

    Returns:
        Dataset: 最終的なマージ済みデータセット
    """
    dataset = dry_run_result.merged_dataset

    # 解決指示を適用して不要なクラスを統合する
    # chosen_name で採択されなかった名前を chosen_name にリネームする
    name_remap: dict[str, str] = {}
    for resolution in resolutions:
        conflict = resolution.conflict
        # どちらか一方が chosen_name でない場合、そちらを chosen_name にリマップ
        if conflict.name_from_dataset_a != resolution.chosen_name:
            name_remap[conflict.name_from_dataset_a] = resolution.chosen_name
        if conflict.name_from_dataset_b != resolution.chosen_name:
            name_remap[conflict.name_from_dataset_b] = resolution.chosen_name

    if not name_remap:
        # リマップ不要なら dry_run の結果をそのまま返す
        return dataset

    # 新しい ClassMap を再構築
    new_names: list[str] = []
    for name in dataset.class_map.id_to_name.values():
        resolved_name = name_remap.get(name, name)
        if resolved_name not in new_names:
            new_names.append(resolved_name)

    new_class_map = ClassMap.from_names(new_names)

    # 全レコードのアノテーションを新ClassMapに基づいて更新
    updated_records: list[ImageRecord] = []
    for record in dataset.records:
        new_annotations: list[Annotation] = []
        for ann in record.annotations:
            old_name = ann.class_name
            resolved_name = name_remap.get(old_name, old_name)
            new_id = new_class_map.name_to_id[resolved_name]
            new_annotations.append(Annotation(
                class_id=new_id,
                class_name=resolved_name,
                bbox=ann.bbox,
                segmentation=ann.segmentation,
            ))
        updated_records.append(ImageRecord(
            image_id=record.image_id,
            original_filename=record.original_filename,
            file_path=record.file_path,
            width=record.width,
            height=record.height,
            annotations=new_annotations,
            split=record.split,
            embedding=record.embedding,
            cluster_id=record.cluster_id,
        ))

    return Dataset(
        records=updated_records,
        class_map=new_class_map,
        source_format=dataset.source_format,
    )
