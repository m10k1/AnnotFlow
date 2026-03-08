"""
分割ロジックのテスト（Phase 0: スケルトン）

Phase 1 で実装後、各テストケースに検証コードを追加する。
"""
from __future__ import annotations

import pytest


def test_split_ratio_sum():
    """train+val+testの件数の合計が元のデータセット件数と一致すること"""
    pytest.skip("Phase 1 で実装")


def test_split_reproducibility():
    """同一シード値で2回分割した結果が一致すること（再現性の確認）"""
    pytest.skip("Phase 1 で実装")


def test_stratified_split_multilabel():
    """マルチラベル画像（1枚に複数クラス混在）に対して層化分割がエラーなく動作すること"""
    pytest.skip("Phase 1 で実装")


def test_split_no_record_duplicated_across_splits():
    """同一ImageRecordがtrain/val/testに重複して割り当てられないこと"""
    pytest.skip("Phase 1 で実装")


def test_split_random_assigns_all_splits():
    """ランダム分割でtrain/val/testすべてに少なくとも1件割り当てられること"""
    pytest.skip("Phase 1 で実装")


def test_split_without_test_set():
    """testセットの比率が0の場合、testに割り当てられる画像がないこと"""
    pytest.skip("Phase 1 で実装")
