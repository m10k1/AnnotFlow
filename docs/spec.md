# アノテーションデータ後処理ツール 仕様書（Claude Code用）

## プロジェクト概要

Label Studio（およびその他のアノテーションツール）から出力されたアノテーションデータを
受け取り、整形・品質チェック・分割・エクスポートを行うデスクトップGUIツール。

**ツール名（仮）:** `AnnotFlow`  
**対象OS:** Windows 10/11  
**UIフレームワーク:** PySide6  
**言語:** Python 3.10+

---

## 対応スコープ（重要）

### 本ツールが対象とするタスク

**対象：物体検出（Object Detection）= バウンディングボックス（BBox）のみ**

| タスク | 対応 | 備考 |
|---|---|---|
| 物体検出（BBox） | ✅ 対象 | 変換・分割・品質チェックすべて対応 |
| セグメンテーション | ⚠️ 保持のみ | データは破棄せず保持するが、変換・分割ロジックには一切関与させない |
| キーポイント検出 | ❌ 対象外 | インポート時に無視する |
| 画像分類 | ❌ 対象外 | インポート時に無視する |

### セグメンテーションデータの扱い方針

- COCO JSONをインポートする際、`segmentation` フィールドが存在する場合は **そのままの値を `Annotation.segmentation` に格納して保持する**
- 分割ロジック（train/val/test）はBBoxのクラスIDのみを参照し、`segmentation` の内容は参照しない
- エクスポート時（COCO JSON）は保持していた `segmentation` をそのまま出力する（変換・加工は行わない）
- YOLOエクスポートはBBoxのみを対象とし、`segmentation` は出力しない（YOLOのsegmentation形式には変換しない）

```python
@dataclass
class Annotation:
    class_id: int
    class_name: str
    bbox: BoundingBox           # 変換・分割ロジックが参照するのはここだけ
    segmentation: list | None = None  # 保持するが加工しない。COCOエクスポート時にそのまま出力
```

> **Claude Codeへの指示:**
> `segmentation` フィールドに対して変換・座標計算・バリデーションのロジックを
> 一切実装しないこと。パーサーで受け取った値をそのまま格納し、
> エクスポーター（COCO）でそのまま出力するだけでよい。

---

## アーキテクチャ方針

- MVCまたはMVVMパターンで設計する
- コア処理ロジック（パーサー・変換・分割・品質チェック）はUIから分離してモジュール化する
- 各フォーマットのパーサーは `parsers/` ディレクトリに個別モジュールとして実装する
- 設定はJSONファイルで永続化する（`config.json`）

---

## ディレクトリ構成

```
annotflow/
├── main.py                  # エントリポイント
├── pyproject.toml           # プロジェクト設定・依存関係（uvで管理）
├── config.json              # ユーザー設定の永続化
├── ui/
│   ├── main_window.py       # メインウィンドウ
│   ├── import_panel.py      # インポートパネル
│   ├── stats_panel.py       # 統計・品質チェックパネル
│   ├── split_panel.py       # 分割設定パネル
│   └── export_panel.py      # エクスポートパネル
├── core/
│   ├── dataset.py           # 内部データモデル（共通形式）
│   ├── splitter.py          # train/val/test分割ロジック
│   ├── quality_checker.py   # 品質チェックロジック
│   └── clip_clusterer.py    # CLIPベクトル化・クラスタリング
├── workers/
│   ├── import_worker.py     # ImportWorker（QThread）
│   ├── export_worker.py     # ExportWorker（QThread）
│   ├── duplicate_worker.py  # DuplicateCheckWorker（QThread）
│   └── clip_worker.py       # ClipWorker（QThread）
├── parsers/
│   ├── base_parser.py       # 抽象基底クラス
│   ├── label_studio_parser.py
│   ├── coco_parser.py
│   └── yolo_parser.py
├── exporters/
│   ├── base_exporter.py     # 抽象基底クラス
│   ├── coco_exporter.py
│   └── yolo_exporter.py
└── tests/
    ├── conftest.py              # 共通フィクスチャ
    ├── fixtures/                # テスト用サンプルデータ（JSON・TXT等）
    ├── test_parsers/
    │   ├── test_label_studio_parser.py
    │   ├── test_coco_parser.py
    │   └── test_yolo_parser.py
    ├── test_core/
    │   ├── test_dataset.py
    │   ├── test_splitter.py
    │   └── test_quality_checker.py
    └── test_exporters/
        ├── test_coco_exporter.py
        └── test_yolo_exporter.py
```

---

## 内部データモデル（共通形式）

すべてのフォーマットはインポート後に以下の共通モデルに変換して保持する。

#### ⚠️ 座標系の統一ルール（重要）

**内部データモデルはすべて「ピクセル単位の絶対座標」で保持する。**

各フォーマットとの変換は以下の通り：

| フォーマット | 座標形式 | 内部モデルへの変換方向 |
|---|---|---|
| COCO JSON | 絶対座標 `[x, y, width, height]` | `x_min=x, y_min=y, x_max=x+w, y_max=y+h` |
| YOLO | 相対座標 `[cx, cy, w, h]`（0.0〜1.0） | `×画像サイズ` で絶対座標に変換 |
| Label Studio JSON | 相対座標（%表記、0〜100） | `÷100 ×画像サイズ` で絶対座標に変換 |

**YOLO変換の具体式（パーサー・エクスポーター実装時に必ず使用すること）：**

```python
# YOLO → 内部モデル（絶対座標）
x_min = (cx - w / 2) * image_width
y_min = (cy - h / 2) * image_height
x_max = (cx + w / 2) * image_width
y_max = (cy + h / 2) * image_height

# 内部モデル（絶対座標）→ YOLO
cx = (x_min + x_max) / 2 / image_width
cy = (y_min + y_max) / 2 / image_height
w  = (x_max - x_min) / image_width
h  = (y_max - y_min) / image_height
```

**Label Studio変換の具体式：**

```python
# Label Studio JSON → 内部モデル（絶対座標）
# Label StudioのBBoxはx, y, width, heightがすべて%表記（0〜100）
x_min = (x_pct / 100) * image_width
y_min = (y_pct / 100) * image_height
x_max = x_min + (w_pct / 100) * image_width
y_max = y_min + (h_pct / 100) * image_height
```

> **注意:** 変換時に `image_width` / `image_height` が必須となるため、
> パーサーは必ず画像サイズを `ImageRecord` に格納してから BoundingBox の変換を行うこと。
> 画像サイズがアノテーションファイル内に記載されていない場合は、
> Pillowで実際の画像ファイルを開いてサイズを取得する。

```python
@dataclass
class BoundingBox:
    x_min: float  # ピクセル単位の絶対座標
    y_min: float  # ピクセル単位の絶対座標
    x_max: float  # ピクセル単位の絶対座標
    y_max: float  # ピクセル単位の絶対座標

@dataclass
class Annotation:
    class_id: int
    class_name: str
    bbox: BoundingBox
    segmentation: list | None = None  # COCOセグメンテーション用

@dataclass
class ImageRecord:
    image_id: str
    original_filename: str      # 元ファイル名を必ず保持する
    file_path: str              # 実際のファイルパス
    width: int
    height: int
    annotations: list[Annotation]
    split: str | None = None    # "train" / "val" / "test"
    embedding: list | None = None   # CLIPベクトル（オプション、Phase 3で使用）
    cluster_id: int | None = None   # CLIPクラスタリング結果（オプション、Phase 3で使用）

@dataclass
class Dataset:
    records: list[ImageRecord]
    class_map: "ClassMap"        # class_namesの代わりにClassMapで一元管理
    source_format: str
```

---

### クラスIDとクラス名のマッピング管理

#### ⚠️ フォーマット間のクラス定義の差異（重要）

各フォーマットはクラスIDの持ち方が根本的に異なる。

| フォーマット | クラス定義の場所 | ID体系 |
|---|---|---|
| COCO JSON | ファイル内`categories`に`id`と`name`が一緒に定義 | 1始まりが多い（0始まりとは限らない） |
| YOLO | `data.yaml`または`classes.txt`に名前のみ、IDは行番号 | 0始まり固定 |
| Label Studio JSON | `result`内の`rectanglelabels`にクラス名のみ | IDを持たない |

**内部モデルは必ず0始まりに統一する。インポート時にIDを正規化すること。**

#### ClassMapデータクラス

クラスIDとクラス名の双方向マッピングを`ClassMap`として`Dataset`が一元管理する。
パーサーはクラス名のみを渡し、IDの採番は`ClassMap`が行う設計とする。

```python
@dataclass
class ClassMap:
    id_to_name: dict[int, str]   # {0: "dog", 1: "cat", 2: "car"}
    name_to_id: dict[str, int]   # {"dog": 0, "cat": 1, "car": 2}

    @classmethod
    def from_names(cls, names: list[str]) -> "ClassMap":
        """名前リストから0始まりで自動採番してClassMapを生成する"""
        id_to_name = {i: name for i, name in enumerate(names)}
        name_to_id = {name: i for i, name in enumerate(names)}
        return cls(id_to_name, name_to_id)

    def get_or_add(self, name: str) -> int:
        """クラス名が未登録なら追加して新しいIDを返す"""
        if name not in self.name_to_id:
            new_id = len(self.id_to_name)
            self.id_to_name[new_id] = name
            self.name_to_id[name] = new_id
        return self.name_to_id[name]
```

#### インポート時の正規化ルール

| フォーマット | 正規化方法 |
|---|---|
| COCO JSON | `category_id`は無視し、`name`を使って`ClassMap`に登録・0始まりで再採番する |
| YOLO | `data.yaml`または`classes.txt`の行順をそのまま0始まりで`ClassMap`に登録する |
| Label Studio JSON | クラス名の**初出順**に0始まりで採番し、採番結果を`config.json`に保存する。次回インポート時は保存済みの順序を優先して読み込み、採番が変わらないようにする |

> **Claude Codeへの指示:**
> パーサーの返り値にクラスIDを含めないこと。
> パーサーはクラス名（文字列）のみを`Annotation`に格納し、
> `ClassMap.get_or_add(name)`でIDを取得してから`Annotation.class_id`にセットする。

---

### 複数データセットのマージ時のクラス衝突管理

複数のデータセットをインポートしてマージする際、同一クラス名・異なるID、または同一ID・異なるクラス名の衝突が発生しうる。

#### 衝突のパターンと対処方針

**パターン1：同一クラス名・異なるID（最も多い）**
```
データセットA: {0: "dog", 1: "cat"}
データセットB: {0: "cat", 1: "dog"}
→ 名前を正としてIDを再採番する（名前ベースでマージ）
```

**パターン2：片方にしか存在しないクラス**
```
データセットA: {0: "dog", 1: "cat"}
データセットB: {0: "dog", 1: "car"}
→ 和集合を取り、欠けているクラスは追加採番する
   マージ後: {0: "dog", 1: "cat", 2: "car"}
```

**パターン3：同一ID・異なるクラス名（まれだが危険）**
```
データセットA: {0: "dog"}
データセットB: {0: "cat"}
→ これはIDベースでは解決不能なため、必ずユーザーに確認を求める
```

#### マージ処理の実装ルール

**⚠️ MVCパターン厳守：`core/dataset.py` からUIを呼び出すことは絶対に禁止**

`merge_datasets()` はUIダイアログを呼び出したり、処理を一時停止したりしてはならない。
代わりに「2段階マージ」の設計とする。

**第1段階：`merge_datasets_dry_run()` ― 自動解決可能なものを処理し、未解決競合を返す**

```python
@dataclass
class MergeConflict:
    """ユーザーによる解決が必要な競合（パターン3専用）"""
    conflicting_id: int
    name_from_dataset_a: str
    name_from_dataset_b: str
    dataset_a_label: str   # 識別用ラベル（例: "ファイルA"）
    dataset_b_label: str

@dataclass
class MergeDryRunResult:
    """第1段階の結果：自動解決済みのデータ＋未解決の競合リスト"""
    merged_dataset: Dataset          # パターン1・2は自動解決済み
    unresolved_conflicts: list[MergeConflict]  # パターン3のみ格納
    conflict_report: list[dict]      # 変更ログ（CSVエクスポート用）

def merge_datasets_dry_run(
    datasets: list[Dataset]
) -> MergeDryRunResult:
    """
    第1段階マージ。
    - パターン1（同名・異ID）: 名前を正として自動解決
    - パターン2（片方のみ存在）: 和集合として自動解決
    - パターン3（同ID・異名）: 解決せずunresolved_conflictsに積む
    UIは一切呼び出さない。
    """
```

**第2段階：`merge_datasets_resolve()` ― ユーザーの選択を受けて最終マージを完了する**

```python
@dataclass
class ConflictResolution:
    """パターン3に対するユーザーの解決指示"""
    conflict: MergeConflict
    chosen_name: str  # ユーザーが正として選んだクラス名

def merge_datasets_resolve(
    dry_run_result: MergeDryRunResult,
    resolutions: list[ConflictResolution]
) -> Dataset:
    """
    第2段階マージ。
    ユーザーの選択（resolutions）を適用して最終的なDatasetを返す。
    UIは一切呼び出さない。
    """
```

**UI側（MergeWorker または import_panel.py）の責務：**

```python
# 1. 第1段階を呼び出す（Workerスレッドで実行）
dry_run_result = merge_datasets_dry_run(datasets)

# 2. 未解決競合があればメインスレッドでダイアログを表示する
if dry_run_result.unresolved_conflicts:
    resolutions = show_conflict_resolution_dialog(dry_run_result.unresolved_conflicts)
    # ↑ UIダイアログはメインスレッドでのみ呼び出す

# 3. ユーザーの選択を渡して第2段階を呼び出す（Workerスレッドで実行）
final_dataset = merge_datasets_resolve(dry_run_result, resolutions)
```

> **Claude Codeへの指示:**
> `core/dataset.py` 内のいかなる関数からも、QMessageBox・QDialog・その他
> PySide6のUIクラスを import・呼び出ししてはならない。
> コアロジックとUIの橋渡しはすべてシグナル／戻り値／データクラスで行うこと。

---

## 機能仕様

### 1. インポート機能

#### 対応フォーマット

| フォーマット | 形式 | 備考 |
|---|---|---|
| Label Studio JSON | `.json` | エクスポート形式（独自スキーマ） |
| COCO JSON | `.json` | `images` / `annotations` / `categories` 構造 |
| YOLO | `.txt` × 画像数 + `classes.txt` or `data.yaml` | クラス定義ファイルも必須 |

#### 重要要件
- **元のファイル名を保持すること**（Label Studioがリネームした場合も、元のファイル名をメタデータから復元する）
- 画像ファイルのフォルダは別途指定させる（アノテーションファイルと画像フォルダを分離して指定できるUIにする）

#### Label Studioのファイル名復元ルール

Label Studioはアップロード時にファイル名の先頭へ一意のID文字列を付与する。

```
付与後（file_uploadフィールドの値）: abcdef12-original_name.jpg
元のファイル名:                       original_name.jpg
```

**復元の具体的な手順：**

```python
import re

def restore_original_filename(file_upload: str) -> str:
    """
    Label StudioのファイルアップロードフィールドからID接頭辞を除去して
    元のファイル名を復元する。

    Label Studioの付与形式: "{8文字の16進数}-{元のファイル名}"
    例: "abcdef12-original_name.jpg" → "original_name.jpg"
    """
    # 8文字の16進数 + ハイフン で始まる場合のみID接頭辞を除去する
    pattern = r'^[0-9a-f]{8}-(.+)$'
    match = re.match(pattern, file_upload, re.IGNORECASE)
    if match:
        return match.group(1)   # 接頭辞を除いた元のファイル名を返す
    # パターンに一致しない場合はfile_uploadの値をそのまま返す（安全なフォールバック）
    return file_upload
```

**画像ファイルの探索ルール：**

復元した元のファイル名で画像フォルダを検索するが、
Label Studioが付与したID付きファイル名のまま保存されている場合も考慮して
以下の順序で探索すること。

```python
def find_image_file(image_folder: Path, file_upload: str) -> Path | None:
    """
    1. 復元した元のファイル名で検索
    2. 見つからなければfile_uploadの値（ID付き）で検索
    3. どちらも見つからなければNoneを返す（スキップ対象）
    """
    original_name = restore_original_filename(file_upload)

    # 1. 元のファイル名で検索
    candidate = image_folder / original_name
    if candidate.exists():
        return candidate

    # 2. ID付きファイル名で検索（Label Studioがそのまま保存した場合）
    candidate = image_folder / file_upload
    if candidate.exists():
        return candidate

    # 3. 見つからない場合はNone（スキップ対象としてskipped_recordsに追加する）
    return None
```

> **Claude Codeへの指示:**
> `restore_original_filename()` と `find_image_file()` は
> `parsers/label_studio_parser.py` 内にヘルパー関数として実装すること。
> `ImageRecord.original_filename` には必ず復元後のファイル名（接頭辞なし）を格納すること。

#### 画像ファイルが存在しない場合の挙動

- インポート処理中に、アノテーションに対応する画像ファイルが指定フォルダに見つからない場合、**そのレコードをスキップしてインポートを継続する**（処理を中断しない）
- スキップしたレコードはすべて「エラー一覧」としてインポート完了後にUIに表示する

**エラー一覧の表示仕様：**

| 項目 | 内容 |
|---|---|
| 表示タイミング | インポート完了後、サマリーダイアログ内にタブとして表示 |
| 表示内容 | ファイル名・アノテーションID・エラー理由（「画像ファイルが見つかりません」） |
| 操作 | 一覧をCSVとしてエクスポートできるボタンを提供する |
| 正常件数の扱い | エラーがあっても正常に読み込めたレコードはそのまま使用可能とする |

```python
# パーサー内での実装イメージ
skipped_records = []

for record in parsed_records:
    image_path = image_folder / record.original_filename
    if not image_path.exists():
        skipped_records.append({
            "filename": record.original_filename,
            "reason": "画像ファイルが見つかりません"
        })
        continue  # スキップして次のレコードへ
    # 正常処理...

# インポート完了後にskipped_recordsをUIに渡して表示する
```

> **Claude Codeへの指示:**
> 画像が見つからない場合に例外を送出してインポート全体を中断する実装は禁止。
> 必ずスキップして継続し、スキップ情報を呼び出し元に返す設計にすること。

#### 複数アノテーターへの対応（Label Studio JSON限定）

> ⚠️ **注意：Label Studioのレビュー機能（accept/reject）はEnterprise版専用**
> コミュニティ版のエクスポートJSONにはレビュー済みフラグは存在しない。
> 以下はコミュニティ版で利用可能な情報のみを使った対応方針とする。

Label Studioで複数人がアノテーションすると、1画像に対して複数の`annotations`エントリが存在する。

```json
{
  "file_upload": "image_001.jpg",
  "annotations": [
    {
      "completed_by": {"id": 1, "email": "annotator_a@example.com"},
      "was_cancelled": false,
      "result": [ ...アノテーターAのBBox... ]
    },
    {
      "completed_by": {"id": 2, "email": "annotator_b@example.com"},
      "was_cancelled": false,
      "result": [ ...アノテーターBのBBox... ]
    }
  ]
}
```

**処理ルール：**

1. `was_cancelled: true` のアノテーションは**無条件で除外**する
2. 残ったアノテーションが**1件のみ**の場合はそのまま採用する
3. 残ったアノテーションが**2件以上**の場合、インポートダイアログで以下を選択させる：

| 選択肢 | 動作 |
|---|---|
| アノテーターを指定して読み込む | ドロップダウンでアノテーター（email）を1人選択し、その人のアノテーションのみ採用 |
| 全アノテーターを個別に読み込む | アノテーターごとに別々の`Dataset`として読み込む（後でマージや比較が可能） |

4. 「アノテーターを指定」の場合、選択したアノテーター以外のエントリは破棄する
5. アノテーターの一覧はインポートファイルを事前スキャンして動的に取得する

> **Claude Codeへの指示:**
> 複数アノテーター判定はLabel Studio JSONパーサー内にのみ実装すること。
> COCO・YOLOパーサーには複数アノテーター処理を実装しない。

#### UIの操作フロー
1. フォーマットをドロップダウンで選択
2. アノテーションファイル（またはフォルダ）をファイルダイアログで選択
3. 画像フォルダを選択
4. **（Label Studio JSONのみ）** ファイルを事前スキャンし、複数アノテーターが検出された場合はアノテーター選択ダイアログを表示
5. 「インポート実行」ボタン → プログレスバー表示
6. インポート完了後、サマリー（件数・クラス一覧・採用アノテーター）を表示

---

### 2. 品質チェック・統計機能

以下をタブ形式で表示する。

#### 2-1. クラスバランス統計
- クラスごとのアノテーション数を棒グラフで表示（matplotlibをPySide6に埋め込む）
- 数値テーブルも併記（クラス名・件数・割合）
- 不均衡なクラスがある場合は警告表示

#### 2-2. 重複画像の検出
- MD5ハッシュによる完全一致検出
- 重複リストをダイアログで表示し、削除するものをユーザーが選択できる

#### 2-3. 空ラベル・未アノテーション画像の検出
- アノテーション数がゼロの画像をリストアップ
- 一覧表示し、除外するかどうか選択できる

#### 2-4. アノテーション数の可視化
- 画像ごとのアノテーション数をヒストグラムで表示
- 外れ値（アノテーション数が極端に多い・少ない画像）をハイライト

#### 2-5. CLIPクラスタリング（類似画像の分散サンプリング）
- モデル: `openai/clip-vit-base-patch32`（`transformers` + `torch` を使用）
- 全画像をCLIPでベクトル化し、K-Meansクラスタリングを実行
- クラスタ数はスライダーで指定（デフォルト: 画像数の10%）
- 各クラスタから均等にサンプリングすることで、特徴が偏らない分割が可能
- 処理はバックグラウンドスレッド（QThread）で実行し、プログレスバーを表示
- クラスタの2D可視化（UMAPまたはt-SNE）をオプションで表示

##### ⚠️ データリーク防止ルール（重要）

CLIPは事前学習済みの固定モデルであるため、ユーザーデータで追加学習しているわけではなく、古典的な意味でのデータリークは発生しない。ただし、クラスタリング結果の使い方を誤ると「疑似リーク」が起きるため、以下のルールを厳守すること。

**CLIPクラスタリングの唯一の目的：各split内に視覚的多様性を確保すること**

```
✅ 正しい使い方：
   全画像をK個にクラスタリング
   → 各クラスタ内でtrain/val/testに比例配分（例：8:1:1）
   → 結果：train・val・test それぞれに全クラスタの代表画像が含まれる
   → 視覚的に偏りのない分割が実現できる

❌ やってはいけない使い方：
   「同一クラスタの画像を同じsplitに固める」実装
   → val/testがtrainと視覚的に近い画像だけで構成される
   → 評価精度が楽観的にバイアスされ、汎化性能を正しく測れなくなる
```

**実装ルール：**
- クラスタリング後、各クラスタ内で `random.shuffle` してから比例配分する
- 「クラスタ単位でsplitを割り当てる」実装は禁止
- 分割比率・ランダムシードは方式A・Bと同じ設定UIを共用する
- CLIPクラスタリングは方式A（ランダム）または方式B（層化）と**組み合わせて使う補助機能**として位置づける。CLIPのみで分割を完結させない

**`ClipWorker`の完了時に必ず行う処理：**

```python
# ClipWorker.run() 内 ― クラスタリング完了後
for record, cluster_id in zip(dataset.records, cluster_labels):
    record.embedding = embeddings[i].tolist()   # CLIPベクトルを格納
    record.cluster_id = int(cluster_id)         # クラスタIDを格納

# finishedシグナルでDatasetをUIに返す
# → splitter.pyはrecord.cluster_idを参照するだけでよい
self.finished.emit(dataset)
```

> **Claude Codeへの指示:**
> `ClipWorker`の処理完了時に、すべての`ImageRecord.cluster_id`へ値を格納してから
> `finished`シグナルを送出すること。
> `splitter.py`はCLIPの計算ロジックを持たず、`record.cluster_id`を参照するだけでよい。
> `cluster_id`が`None`の場合（CLIPを実行していない場合）は、
> クラスタリングなしのランダム分割または層化分割にフォールバックすること。

---

### 3. train/val/test 分割機能

#### 分割方式（UIでどちらか選択）

**方式A：比率指定（ランダム分割）**
- train / val / test の比率をスピンボックスで入力（合計100%になるようにバリデーション）
- ランダムシード値を入力欄で指定（再現性確保のため）
- testセットを「なし」にすることも可能

**方式B：クラス層化分割（Stratified Split）**

> ⚠️ **重要：`scikit-learn` の `StratifiedShuffleSplit` は使用禁止**
>
> `StratifiedShuffleSplit` は「1画像 = 1クラス」のシングルラベル分類専用のため、
> 物体検出（1枚の画像に複数クラスが混在するマルチラベル）には対応していない。
> 使用するとエラーになるか、意図しない分割結果になる。

**代わりに以下のいずれかを使用すること：**

**推奨：`iterative-stratification` ライブラリの `MultilabelStratifiedShuffleSplit`**

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

# 各画像をマルチホットベクトルに変換（クラス数次元のbinary vector）
# 例: クラスが [dog, cat, car] の3クラスで、
#     画像Aに dog と car がいれば → [1, 0, 1]
def build_multilabel_matrix(records: list[ImageRecord], num_classes: int) -> np.ndarray:
    matrix = np.zeros((len(records), num_classes), dtype=int)
    for i, record in enumerate(records):
        for ann in record.annotations:
            matrix[i, ann.class_id] = 1
    return matrix

# 分割実行
X = np.arange(len(records))
y = build_multilabel_matrix(records, num_classes)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
train_idx, val_idx = next(msss.split(X, y))
```

**フォールバック：マルチホットベクトル + ランダム分割（iterative-stratificationが使えない場合）**

```python
# iterative-stratification のインストールに失敗した場合は方式Aのランダム分割にフォールバックし、
# UIにその旨を警告メッセージとして表示すること
```

- 比率・シード値の指定は方式Aと同様
- `iterative-stratification` を `requirements.txt` に追加すること（`iterative-stratification>=0.1.7`）

#### 分割後の表示
- 分割結果のサマリーをテーブルで表示（split別件数・クラス分布）
- 分割のやり直しボタン（シードを変えて再実行）

---

### 4. エクスポート機能

#### 対応フォーマット

| フォーマット | 出力内容 |
|---|---|
| YOLO | `train/images/`, `train/labels/`, `val/images/`, `val/labels/`, `test/images/`, `test/labels/`, `data.yaml` |
| COCO JSON | split別に `train.json`, `val.json`, `test.json` |

#### エクスポートオプション
- **画像ファイルの扱い**（ラジオボタンで選択）
  - 画像も出力フォルダにコピーする
  - アノテーションファイルのみ出力する（画像はコピーしない）
- 出力先フォルダをファイルダイアログで選択
- 「エクスポート実行」ボタン → プログレスバー表示
- 完了後、出力フォルダをエクスプローラーで開くボタンを表示

#### ファイル名の扱い
- エクスポート時は必ず元のファイル名を使用する
- Label StudioがリネームしたIDベースのファイル名は使用しない

---

## UIレイアウト

```
┌─────────────────────────────────────────────────┐
│  AnnotFlow                          [─][□][×]   │
├─────────────────────────────────────────────────┤
│  [インポート] [品質チェック] [分割] [エクスポート] ← タブ │
├─────────────────────────────────────────────────┤
│                                                 │
│   （各タブのコンテンツ）                          │
│                                                 │
├─────────────────────────────────────────────────┤
│  ステータスバー：件数、現在の操作状態を表示           │
└─────────────────────────────────────────────────┘
```

- メインウィンドウはリサイズ可能（最小サイズ: 900x600px）
- ダークモード対応（QSS使用）
- 日本語UI

---

## パッケージ管理（uv）

本プロジェクトは `pip` ではなく **`uv`** を使用してパッケージ管理を行う。

### プロジェクト初期化

```bash
# プロジェクト初期化（pyproject.tomlが生成される）
uv init annotflow
cd annotflow

# 仮想環境の作成
uv venv

# 仮想環境の有効化（Windows）
.venv\Scripts\activate
```

### 依存パッケージのインストール

```bash
# 本番依存関係のインストール
uv add PySide6 numpy scikit-learn iterative-stratification torch transformers Pillow matplotlib umap-learn tqdm

# 開発用依存関係のインストール
uv add --dev pytest pytest-cov
```

### `pyproject.toml` の構成

`requirements.txt` は作成せず、`pyproject.toml` で依存関係を一元管理すること。

```toml
[project]
name = "annotflow"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "PySide6>=6.5.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "iterative-stratification>=0.1.7",
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "Pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "umap-learn>=0.5.0",
    "tqdm>=4.65.0",
]

[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
```

### 主要コマンド

```bash
# パッケージ追加
uv add <package>

# 開発用パッケージ追加
uv add --dev <package>

# テスト実行
uv run pytest tests/ -v

# カバレッジ付きテスト実行
uv run pytest tests/ -v --cov=core --cov=parsers --cov=exporters --cov-report=term-missing

# アプリ起動
uv run python main.py
```

> **Claude Codeへの指示:**
> `pip install` コマンドを一切使用しないこと。
> パッケージの追加・削除はすべて `uv add` / `uv remove` で行うこと。
> `requirements.txt` は作成せず、`pyproject.toml` で依存関係を管理すること。

---

---

## エラーハンドリング方針

- すべての外部ファイル操作はtry-exceptで囲み、エラー時はQMessageBoxで日本語メッセージを表示する
- **画像ファイルが見つからないレコードはスキップして継続し、インポート完了後にエラー一覧としてUIに表示する（処理を中断しない）**
- インポート時にその他のファイルが見つからない場合も同様にスキップして件数をログに残す
- CLIP処理中にメモリ不足になった場合、バッチサイズを自動的に半減して再試行する
- 処理中は必ずキャンセルボタンを提供する（QThread + シグナルで実装）

---

## UI実装ガイドライン

### レイアウト・余白

**すべての主要レイアウトに余白とスペーシングを必ず設定すること。ウィジェットを密着させないこと。**

```python
# ✅ 正しい：すべてのQVBoxLayout / QHBoxLayoutに適用する
layout = QVBoxLayout()
layout.setContentsMargins(16, 16, 16, 16)
layout.setSpacing(10)

# ❌ 禁止：デフォルト値のまま使用する
layout = QVBoxLayout()  # マージン・スペーシング未設定
```

### SizePolicy（伸縮の制御）

ウィジェットの種類に応じてSizePolicyを必ず設定し、ウィンドウリサイズ時に適切に振る舞うようにすること。

| ウィジェットの種類 | 水平方向 | 垂直方向 |
|---|---|---|
| ボタン（QPushButton） | Preferred | **Fixed** |
| 入力フィールド（QLineEdit, QSpinBox等） | Expanding | **Fixed** |
| リスト（QListWidget, QTableWidget） | Expanding | **Expanding** |
| グラフ表示エリア（matplotlibキャンバス） | Expanding | **Expanding** |
| ラベル（QLabel） | Preferred | **Fixed** |

```python
from PySide6.QtWidgets import QSizePolicy

# ボタン・入力フィールド：垂直Fixedで高さを固定
button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

# リスト・グラフ：両方Expandingでリサイズに追従
table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
```

### QSplitterの活用

左側に設定エリア、右側にメイン表示エリア（リスト・統計・グラフ等）がある場合は
`QSplitter` を使用してユーザーが境界をドラッグでリサイズできるようにすること。

```python
from PySide6.QtWidgets import QSplitter
from PySide6.QtCore import Qt

splitter = QSplitter(Qt.Orientation.Horizontal)
splitter.addWidget(left_panel)    # 設定エリア
splitter.addWidget(right_panel)   # メイン表示エリア
splitter.setStretchFactor(0, 1)   # 左：比率1
splitter.setStretchFactor(1, 3)   # 右：比率3（デフォルトは右を広く）
```

### スタイリング（QSS）

**`main_window.py` 内でハードコードした色指定・スタイル記述を行ってはならない。**
すべてのスタイルは `assets/style.qss` に記述し、起動時に読み込む設計にすること。

```python
# main_window.py でのQSS読み込み
def _load_stylesheet(self) -> None:
    """assets/style.qssを読み込んでアプリ全体に適用する"""
    qss_path = Path(__file__).parent.parent / "assets" / "style.qss"
    if qss_path.exists():
        self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
```

**`assets/style.qss` の基本構成（ダークモード）：**

```css
/* ベースカラー */
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Yu Gothic UI", sans-serif;
    font-size: 13px;
}

/* ボタン */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 16px;
}
QPushButton:hover { background-color: #45475a; }
QPushButton:disabled { color: #585b70; }

/* タブ */
QTabBar::tab { padding: 8px 20px; }
QTabBar::tab:selected { border-bottom: 2px solid #89b4fa; }

/* テーブル */
QTableWidget { gridline-color: #313244; }
QHeaderView::section { background-color: #181825; padding: 6px; }

/* プログレスバー */
QProgressBar { border-radius: 4px; background-color: #313244; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
```

**ディレクトリ構成に `assets/` を追加すること：**

```
annotflow/
├── assets/
│   └── style.qss        # アプリ全体のスタイル定義
├── main.py
...
```

> **Claude Codeへの指示:**
> `setStyleSheet("color: red")` のようなインラインスタイルを
> ウィジェット個別に設定することを禁止する。
> ただし、エラー表示・警告表示など動的に状態が変わる箇所は例外として
> `setProperty` + QSSのセレクタで制御すること。
>
> ```python
> # 動的状態の変更はsetPropertyで行い、QSSセレクタで見た目を定義する
> widget.setProperty("state", "error")
> widget.style().unpolish(widget)
> widget.style().polish(widget)
> ```

**UIをブロックする可能性のある処理はすべてQThreadでバックグラウンド実行すること。**

### QThreadを必ず使用する処理一覧

| 処理 | 対象クラス（命名規則） |
|---|---|
| インポート（ファイルI/O・パース・画像サイズ取得） | `ImportWorker` |
| エクスポート（ファイルI/O・画像コピー） | `ExportWorker` |
| 重複画像検出（MD5ハッシュ計算） | `DuplicateCheckWorker` |
| CLIPベクトル化・クラスタリング | `ClipWorker` |

### 実装パターン（全Workerで統一すること）

```python
from PySide6.QtCore import QThread, Signal

class ImportWorker(QThread):
    # シグナル定義
    progress = Signal(int)          # 進捗（0〜100）
    status_message = Signal(str)    # ステータスバーへのメッセージ
    finished = Signal(object)       # 完了時：Datasetオブジェクトを渡す
    error = Signal(str)             # エラー時：エラーメッセージを渡す
    skipped = Signal(list)          # スキップしたレコード一覧を渡す

    def __init__(self, file_path: str, image_folder: str, fmt: str):
        super().__init__()
        self._cancelled = False     # キャンセルフラグ

    def run(self):
        try:
            for i, record in enumerate(records):
                if self._cancelled:
                    return          # キャンセル時は即座に終了
                # 処理...
                self.progress.emit(int(i / total * 100))
            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True      # キャンセルフラグを立てる（強制終了しない）
```

### UI側の必須実装

- 処理開始時：「実行」ボタンを無効化し、プログレスバーを表示する
- 処理中：`progress`シグナルでプログレスバーを更新する
- 処理中：`status_message`シグナルでステータスバーにメッセージを表示する
- 処理中：「キャンセル」ボタンを表示し、クリックで`worker.cancel()`を呼ぶ
- 処理完了時：「実行」ボタンを再有効化し、プログレスバーを非表示にする
- エラー時：`QMessageBox`で日本語エラーメッセージを表示する

### ⚠️ QThreadの禁止事項

```python
# ❌ 禁止：Workerスレッドから直接UIウィジェットを操作する
# （クラッシュやデータ競合の原因になる）
def run(self):
    self.some_label.setText("処理中")  # 絶対に書かない

# ✅ 正しい：シグナルを経由してメインスレッドにUIの更新を委ねる
def run(self):
    self.status_message.emit("処理中")  # シグナルで通知する
```

> **Claude Codeへの指示:**
> 上記の4処理については、メインスレッドで直接実行する実装を禁止する。
> 必ずWorkerクラスを定義してQThreadで実行すること。
> すべてのWorkerで`progress` / `finished` / `error` / `cancel()`を実装し、
> シグナル名・メソッド名は上記の命名規則に統一すること。

---

## 実装の優先順位

### Phase 0（テスト基盤）※ Phase 1より先に完了すること
0. `tests/` ディレクトリ・`conftest.py`・共通フィクスチャの作成
0. `tests/fixtures/` にサンプルデータ（COCO JSON・YOLO・Label Studio JSON）を作成
0. 全テストケースのスケルトン（関数定義のみ、中身は`pass`）を作成して`pytest`の実行を確認

### Phase 1（MVP）
1. インポート機能（Label Studio JSON / COCO JSON / YOLO）＋対応テストのパス確認
2. クラスバランス統計表示
3. 空ラベル・重複画像の検出＋対応テストのパス確認
4. train/val/test分割（比率指定・層化分割）＋対応テストのパス確認
5. YOLOエクスポート＋対応テストのパス確認

### Phase 2
6. COCOエクスポート＋対応テストのパス確認
7. アノテーション数ヒストグラム
8. エクスプローラーで開くボタン等のUX改善

### Phase 3
9. CLIPクラスタリング機能
10. UMAP可視化

---

## テスト方針（pytest）

### 基本方針

- **Phase 1の実装コードを書く前に、テストコードを先に作成すること（TDDアプローチ）**
- テスト対象はコアロジックのみ（UIコードはテスト対象外）
- テストは `tests/` ディレクトリに配置し、モジュール構成に対応させる

### ディレクトリ構成

```
tests/
├── conftest.py                  # 共通フィクスチャ（サンプルデータ等）
├── test_parsers/
│   ├── test_label_studio_parser.py
│   ├── test_coco_parser.py
│   └── test_yolo_parser.py
├── test_core/
│   ├── test_dataset.py          # ClassMap・マージ処理のテスト
│   ├── test_splitter.py         # 分割ロジックのテスト
│   └── test_quality_checker.py  # 品質チェックのテスト
└── test_exporters/
    ├── test_coco_exporter.py
    └── test_yolo_exporter.py
```

### conftest.py に定義する共通フィクスチャ

```python
import pytest
from core.dataset import BoundingBox, Annotation, ImageRecord, Dataset, ClassMap

@pytest.fixture
def sample_class_map():
    """テスト用クラスマップ（dog=0, cat=1, car=2）"""
    return ClassMap.from_names(["dog", "cat", "car"])

@pytest.fixture
def sample_image_record(sample_class_map):
    """テスト用ImageRecord（dogとcatが1枚に混在するマルチラベル画像）"""
    return ImageRecord(
        image_id="001",
        original_filename="image_001.jpg",
        file_path="/tmp/images/image_001.jpg",
        width=640,
        height=480,
        annotations=[
            Annotation(class_id=0, class_name="dog",
                       bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=150)),
            Annotation(class_id=1, class_name="cat",
                       bbox=BoundingBox(x_min=200, y_min=50, x_max=350, y_max=200)),
        ]
    )

@pytest.fixture
def sample_dataset(sample_image_record, sample_class_map):
    """テスト用Dataset（10件）"""
    records = [sample_image_record] * 10
    return Dataset(records=records, class_map=sample_class_map, source_format="coco")
```

### 各テストモジュールの必須テストケース

#### test_parsers/test_coco_parser.py

```python
def test_coco_bbox_conversion():
    """COCOの[x,y,w,h]が内部モデルの絶対座標[x_min,y_min,x_max,y_max]に正しく変換されること"""

def test_coco_category_id_renumbered_to_zero_based():
    """COCOの1始まりcategory_idが0始まりに正規化されること"""

def test_coco_segmentation_preserved_without_modification():
    """segmentationフィールドが変換されずそのまま保持されること"""

def test_coco_missing_image_skipped():
    """対応する画像ファイルが存在しないレコードがスキップされ、skipped_recordsに記録されること"""
```

#### test_parsers/test_yolo_parser.py

```python
def test_yolo_relative_to_absolute_conversion():
    """YOLO相対座標(cx,cy,w,h)が絶対座標に正しく変換されること"""
    # 具体値で検証する
    # 例: cx=0.5, cy=0.5, w=0.25, h=0.25, image_width=640, image_height=480
    # → x_min=240, y_min=180, x_max=400, y_max=300

def test_yolo_roundtrip():
    """絶対座標 → YOLO相対座標 → 絶対座標の変換で元の値に戻ること（丸め誤差1px以内）"""

def test_yolo_class_id_matches_yaml_order():
    """data.yamlの行順通りにクラスIDが採番されること"""
```

#### test_parsers/test_label_studio_parser.py

```python
def test_label_studio_percent_to_absolute_conversion():
    """Label Studioの%座標が絶対座標に正しく変換されること"""

def test_label_studio_cancelled_annotation_excluded():
    """was_cancelled: true のアノテーションが除外されること"""

def test_label_studio_multiple_annotators_detected():
    """複数アノテーターが存在する場合に検出され、annotator_listが返されること"""

def test_restore_original_filename_removes_prefix():
    """8文字16進数+ハイフンの接頭辞が除去されて元のファイル名が復元されること"""
    assert restore_original_filename("abcdef12-original_name.jpg") == "original_name.jpg"

def test_restore_original_filename_no_prefix():
    """接頭辞がない場合はそのままの値が返されること"""
    assert restore_original_filename("original_name.jpg") == "original_name.jpg"

def test_restore_original_filename_hyphen_in_name():
    """元のファイル名自体にハイフンが含まれる場合でも正しく復元されること"""
    assert restore_original_filename("abcdef12-my-photo-001.jpg") == "my-photo-001.jpg"

def test_find_image_file_by_original_name(tmp_path):
    """復元した元のファイル名で画像が正しく見つかること"""

def test_find_image_file_by_id_prefixed_name(tmp_path):
    """元のファイル名で見つからない場合にID付きファイル名でフォールバック検索されること"""

def test_find_image_file_not_found_returns_none(tmp_path):
    """どちらの名前でも見つからない場合にNoneが返されること"""

def test_label_studio_original_filename_stored_without_prefix():
    """`ImageRecord.original_filename` に接頭辞なしのファイル名が格納されること"""
```

#### test_core/test_dataset.py

```python
def test_classmap_zero_based_numbering():
    """ClassMap.from_namesで0始まりのIDが採番されること"""

def test_classmap_get_or_add_new_class():
    """未登録クラス名でget_or_addを呼ぶと新しいIDが採番されること"""

def test_merge_dry_run_same_name_different_id():
    """パターン1：同一クラス名・異なるIDのデータセットをdry_runするとIDが統一されること"""

def test_merge_dry_run_union_of_classes():
    """パターン2：片方にしか存在しないクラスがマージ後に和集合として含まれること"""

def test_merge_dry_run_conflict_report():
    """dry_run時のIDの変更がconflict_reportリストに記録されること"""

def test_merge_dry_run_pattern3_returns_unresolved():
    """パターン3（同ID・異クラス名）がunresolved_conflictsに格納され、自動解決されないこと"""

def test_merge_dry_run_no_ui_import():
    """core/dataset.pyがPySide6をimportしていないこと（MVCの分離確認）"""
    import ast, pathlib
    source = pathlib.Path("core/dataset.py").read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    for imp in imports:
        name = imp.names[0].name if isinstance(imp, ast.Import) else (imp.module or "")
        assert "PySide6" not in name, "core/dataset.py はPySide6をimportしてはならない"

def test_merge_resolve_applies_user_choice():
    """merge_datasets_resolve()がユーザーの選択（ConflictResolution）を正しく適用すること"""
```

#### test_core/test_splitter.py

```python
def test_split_ratio_sum():
    """train+val+testの件数の合計が元のデータセット件数と一致すること"""

def test_split_reproducibility():
    """同一シード値で2回分割した結果が一致すること（再現性の確認）"""

def test_stratified_split_multilabel():
    """マルチラベル画像（1枚に複数クラス混在）に対して層化分割がエラーなく動作すること"""

def test_split_no_record_duplicated_across_splits():
    """同一ImageRecordがtrain/val/testに重複して割り当てられないこと"""
```

#### test_exporters/test_yolo_exporter.py

```python
def test_yolo_export_absolute_to_relative_conversion():
    """エクスポート時に絶対座標がYOLO相対座標に正しく変換されること"""

def test_yolo_export_preserves_original_filename():
    """エクスポートされるtxtファイル名が元の画像ファイル名と一致すること"""

def test_yolo_export_data_yaml_contains_all_classes():
    """data.yamlにすべてのクラス名が含まれること"""
```

### テスト実行コマンド

```bash
# 全テスト実行
uv run pytest tests/ -v

# カバレッジ計測付き
uv run pytest tests/ -v --cov=core --cov=parsers --cov=exporters --cov-report=term-missing
```

> **Claude Codeへの指示:**
> Phase 1のコード実装より前に `tests/` の作成を完了すること。
> テストはすべて実際に`pytest`でパスすることを確認してから次のPhaseに進むこと。
> テスト用のサンプルデータ（小さいJSONファイル等）は `tests/fixtures/` に配置すること。

---

- コードはすべて日本語コメントで記述すること
- 各モジュールにはdocstringを記述すること
- テスト用にサンプルデータ生成スクリプト（`scripts/generate_sample_data.py`）も作成すること
- `pyproject.toml` と `README.md`（日本語）を必ず生成すること（`requirements.txt` は作成しない）
- パッケージ管理はすべて `uv` を使用すること（`pip install` は使用禁止）
- Phase 0（テスト基盤）→ Phase 1と順番に実装し、各Phaseが動作確認できる状態で区切ること