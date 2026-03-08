"""
Label Studio JSON フォーマットのパーサー

Label Studio からエクスポートした JSON ファイルを読み込んで
内部データモデル（Dataset）に変換する。
"""
from __future__ import annotations

import re
from pathlib import Path

from core.dataset import Dataset
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

    Args:
        file_upload: Label Studio が付与したファイル名（接頭辞付き）

    Returns:
        接頭辞を除去した元のファイル名。パターンに一致しない場合はそのまま返す。
    """
    # 8文字の16進数 + ハイフン で始まる場合のみ ID 接頭辞を除去する
    pattern = r'^[0-9a-f]{8}-(.+)$'
    match = re.match(pattern, file_upload, re.IGNORECASE)
    if match:
        return match.group(1)   # 接頭辞を除いた元のファイル名を返す
    # パターンに一致しない場合は file_upload の値をそのまま返す（安全なフォールバック）
    return file_upload


def find_image_file(image_folder: Path, file_upload: str) -> Path | None:
    """
    画像ファイルを以下の順序で検索する:
    1. 復元した元のファイル名で検索
    2. 見つからなければ file_upload の値（ID付き）で検索
    3. どちらも見つからなければ None を返す（スキップ対象）

    Args:
        image_folder: 画像フォルダのパス
        file_upload: Label Studio が付与したファイル名

    Returns:
        見つかった画像ファイルの Path。見つからない場合は None。
    """
    original_name = restore_original_filename(file_upload)

    # 1. 元のファイル名で検索
    candidate = image_folder / original_name
    if candidate.exists():
        return candidate

    # 2. ID付きファイル名で検索（Label Studio がそのまま保存した場合）
    candidate = image_folder / file_upload
    if candidate.exists():
        return candidate

    # 3. 見つからない場合は None（スキップ対象としてskipped_recordsに追加する）
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

        Args:
            annotation_path: Label Studio エクスポートJSONのパス
            image_folder: 画像フォルダのパス

        Returns:
            (Dataset, skipped_records) のタプル
        """
        raise NotImplementedError("Phase 1 で実装する")

    def scan_annotators(self, annotation_path: Path) -> list[dict]:
        """
        アノテーションファイルを事前スキャンしてアノテーター一覧を返す。

        Args:
            annotation_path: Label Studio エクスポートJSONのパス

        Returns:
            アノテーター情報のリスト
            例: [{"id": 1, "email": "user@example.com"}]
        """
        raise NotImplementedError("Phase 1 で実装する")
