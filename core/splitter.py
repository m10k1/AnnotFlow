"""
train/val/test 分割ロジック

方式A：ランダム分割
方式B：マルチラベル層化分割（iterative-stratification を使用）

PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

from core.dataset import Dataset, ImageRecord


def split_random(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    方式A：ランダム分割。
    各 ImageRecord の split フィールドを "train" / "val" / "test" に設定して返す。

    Args:
        dataset: 分割対象のデータセット
        train_ratio: 訓練セットの比率（0.0〜1.0）
        val_ratio: 検証セットの比率（0.0〜1.0）
        test_ratio: テストセットの比率（0.0〜1.0）
        seed: 乱数シード（再現性のため）

    Returns:
        split が設定された Dataset
    """
    raise NotImplementedError("Phase 1 で実装する")


def split_stratified(
    dataset: Dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dataset:
    """
    方式B：マルチラベル層化分割（MultilabelStratifiedShuffleSplit 使用）。
    iterative-stratification が使えない場合はランダム分割にフォールバックする。

    Args:
        dataset: 分割対象のデータセット
        train_ratio: 訓練セットの比率
        val_ratio: 検証セットの比率
        test_ratio: テストセットの比率
        seed: 乱数シード

    Returns:
        split が設定された Dataset
    """
    raise NotImplementedError("Phase 1 で実装する")
