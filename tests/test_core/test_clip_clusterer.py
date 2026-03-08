"""
CLIPクラスタリングのテスト（Phase 3）

モック（unittest.mock）を使用して実際のモデルダウンロードを一切行わない。
テストはすべて高速に完了する。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.dataset import Annotation, BoundingBox, ClassMap, Dataset, ImageRecord


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_records(n: int) -> list[ImageRecord]:
    """テスト用 ImageRecord リストを生成する（画像ファイルは不要）。"""
    return [
        ImageRecord(
            image_id=str(i),
            original_filename=f"image_{i:03d}.jpg",
            file_path=f"/tmp/image_{i:03d}.jpg",
            width=640,
            height=480,
            annotations=[],
        )
        for i in range(n)
    ]


def _make_dataset(n: int) -> Dataset:
    """テスト用 Dataset を生成する。"""
    class_map = ClassMap.from_names(["dog"])
    return Dataset(
        records=_make_records(n),
        class_map=class_map,
        source_format="coco",
    )


# ---------------------------------------------------------------------------
# ClipClusterer.cluster() のテスト（モデル不要なので直接テスト可能）
# ---------------------------------------------------------------------------

def test_cluster_assigns_correct_number_of_clusters():
    """
    K-Means が指定した n_clusters 個のクラスタに分類すること。
    埋め込みベクトルはダミーを直接渡すのでモデルのロードは不要。
    """
    from core.clip_clusterer import ClipClusterer

    n_records = 12
    n_clusters = 3
    # n_clusters 個の明確に分離したクラスタ（各4サンプル）
    embeddings = np.vstack([
        np.random.RandomState(0).randn(4, 512) + np.array([10.0] + [0.0] * 511),
        np.random.RandomState(1).randn(4, 512) + np.array([0.0, 10.0] + [0.0] * 510),
        np.random.RandomState(2).randn(4, 512) + np.array([0.0, 0.0, 10.0] + [0.0] * 509),
    ]).astype(np.float32)

    clusterer = ClipClusterer(n_clusters=n_clusters, seed=42)
    labels = clusterer.cluster(embeddings)

    assert len(labels) == n_records
    # 生成されたクラスタIDは 0〜(n_clusters-1) の範囲内
    assert set(labels).issubset(set(range(n_clusters)))


def test_cluster_labels_are_integers():
    """クラスタラベルが int 型に変換可能であること。"""
    from core.clip_clusterer import ClipClusterer

    embeddings = np.random.RandomState(0).randn(15, 512).astype(np.float32)
    clusterer = ClipClusterer(n_clusters=3, seed=0)
    labels = clusterer.cluster(embeddings)

    for label in labels:
        assert isinstance(int(label), int), f"ラベルが int に変換できない: {label!r}"


def test_cluster_reproducibility():
    """同一シードで2回クラスタリングした結果が一致すること（再現性）。"""
    from core.clip_clusterer import ClipClusterer

    embeddings = np.random.RandomState(7).randn(20, 512).astype(np.float32)
    clusterer = ClipClusterer(n_clusters=4, seed=7)

    labels1 = clusterer.cluster(embeddings)
    labels2 = clusterer.cluster(embeddings)

    np.testing.assert_array_equal(labels1, labels2)


def test_cluster_n_clusters_capped_by_record_count():
    """
    n_clusters がレコード数を超える場合、レコード数でキャップされること。
    （K-Means はクラスタ数 > サンプル数だとエラーになるため）
    """
    from core.clip_clusterer import ClipClusterer

    # 5件のデータに対して n_clusters=100 を指定
    embeddings = np.random.RandomState(0).randn(5, 512).astype(np.float32)
    clusterer = ClipClusterer(n_clusters=100, seed=0)
    labels = clusterer.cluster(embeddings)

    assert len(labels) == 5
    # クラスタ数は max 5 になるはず
    assert len(set(labels)) <= 5


# ---------------------------------------------------------------------------
# ClipClusterer.encode_images() のテスト（CLIPモデルはモックに置き換え）
# ---------------------------------------------------------------------------

def test_encode_images_returns_correct_shape(tmp_path):
    """
    encode_images() が (n_images, embedding_dim) の形状の配列を返すこと。
    CLIPModel / CLIPProcessor はモックに置き換える。
    """
    import torch
    from PIL import Image
    from core.clip_clusterer import ClipClusterer

    # ダミー画像ファイルを作成
    n = 6
    file_paths = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 40, 0, 0)).save(str(img_path))
        file_paths.append(str(img_path))

    embedding_dim = 512

    with (
        patch("core.clip_clusterer.CLIPModel") as mock_model_cls,
        patch("core.clip_clusterer.CLIPProcessor") as mock_proc_cls,
    ):
        # プロセッサのモック：バッチサイズに応じた pixel_values テンソルを返す
        mock_proc = MagicMock()
        def _proc_side_effect(images=None, return_tensors=None):
            bs = len(images)
            return {"pixel_values": torch.zeros(bs, 3, 224, 224)}
        mock_proc.side_effect = _proc_side_effect
        mock_proc_cls.from_pretrained.return_value = mock_proc

        # モデルのモック：バッチサイズに応じた features テンソルを返す
        mock_model = MagicMock()
        def _feat_side_effect(**kwargs):
            pixel_values = kwargs.get("pixel_values")
            bs = pixel_values.shape[0]
            return torch.randn(bs, embedding_dim)
        mock_model.get_image_features.side_effect = _feat_side_effect
        mock_model_cls.from_pretrained.return_value = mock_model

        clusterer = ClipClusterer(n_clusters=2, seed=0)
        clusterer.load_model()
        embeddings = clusterer.encode_images(file_paths, batch_size=2)

    assert embeddings.shape == (n, embedding_dim)


def test_encode_images_calls_model_multiple_batches(tmp_path):
    """
    encode_images() がバッチ分だけ get_image_features を呼び出すこと。
    """
    import torch
    from PIL import Image
    from core.clip_clusterer import ClipClusterer

    n = 5
    batch_size = 2  # 5件 / 2 = 3バッチ（3, 2枚ずつ）
    expected_calls = (n + batch_size - 1) // batch_size  # ceil(5/2) = 3

    file_paths = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (32, 32)).save(str(img_path))
        file_paths.append(str(img_path))

    with (
        patch("core.clip_clusterer.CLIPModel") as mock_model_cls,
        patch("core.clip_clusterer.CLIPProcessor") as mock_proc_cls,
    ):
        mock_proc = MagicMock()
        def _proc_side_effect(images=None, return_tensors=None):
            bs = len(images)
            return {"pixel_values": torch.zeros(bs, 3, 224, 224)}
        mock_proc.side_effect = _proc_side_effect
        mock_proc_cls.from_pretrained.return_value = mock_proc

        mock_model = MagicMock()
        def _feat_side_effect(**kwargs):
            bs = kwargs["pixel_values"].shape[0]
            return torch.ones(bs, 512)
        mock_model.get_image_features.side_effect = _feat_side_effect
        mock_model_cls.from_pretrained.return_value = mock_model

        clusterer = ClipClusterer(n_clusters=2, seed=0)
        clusterer.load_model()
        clusterer.encode_images(file_paths, batch_size=batch_size)

    assert mock_model.get_image_features.call_count == expected_calls


def test_encode_images_progress_callback_called(tmp_path):
    """encode_images() が progress_callback を呼び出すこと。"""
    import torch
    from PIL import Image
    from core.clip_clusterer import ClipClusterer

    n = 4
    file_paths = []
    for i in range(n):
        img_path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (32, 32)).save(str(img_path))
        file_paths.append(str(img_path))

    progress_values: list[int] = []

    with (
        patch("core.clip_clusterer.CLIPModel") as mock_model_cls,
        patch("core.clip_clusterer.CLIPProcessor") as mock_proc_cls,
    ):
        mock_proc = MagicMock()
        mock_proc.side_effect = lambda images=None, return_tensors=None: {
            "pixel_values": torch.zeros(len(images), 3, 224, 224)
        }
        mock_proc_cls.from_pretrained.return_value = mock_proc

        mock_model = MagicMock()
        mock_model.get_image_features.side_effect = lambda **kw: torch.ones(
            kw["pixel_values"].shape[0], 512
        )
        mock_model_cls.from_pretrained.return_value = mock_model

        clusterer = ClipClusterer(n_clusters=2, seed=0)
        clusterer.load_model()
        clusterer.encode_images(
            file_paths,
            batch_size=2,
            progress_callback=progress_values.append,
        )

    assert len(progress_values) > 0
    assert progress_values[-1] == 100


# ---------------------------------------------------------------------------
# ClipClusterer.apply_to_dataset() のテスト
# ---------------------------------------------------------------------------

def test_apply_to_dataset_stores_embedding_and_cluster_id():
    """
    apply_to_dataset() が各 ImageRecord に embedding と cluster_id を格納すること。
    """
    from core.clip_clusterer import ClipClusterer

    n = 9
    n_clusters = 3
    embedding_dim = 512
    dataset = _make_dataset(n)

    dummy_embeddings = np.random.RandomState(1).randn(n, embedding_dim).astype(np.float32)
    dummy_labels = np.array([i % n_clusters for i in range(n)])

    clusterer = ClipClusterer(n_clusters=n_clusters, seed=42)
    clusterer.apply_to_dataset(dataset, dummy_embeddings, dummy_labels)

    for i, record in enumerate(dataset.records):
        assert record.embedding is not None, f"record[{i}].embedding が None"
        assert len(record.embedding) == embedding_dim
        assert record.cluster_id is not None, f"record[{i}].cluster_id が None"
        assert 0 <= record.cluster_id < n_clusters


def test_apply_to_dataset_embedding_matches_input():
    """apply_to_dataset() が入力埋め込みの値を正確に格納すること。"""
    from core.clip_clusterer import ClipClusterer

    n = 5
    embedding_dim = 512
    dataset = _make_dataset(n)

    embeddings = np.random.RandomState(99).randn(n, embedding_dim).astype(np.float32)
    labels = np.zeros(n, dtype=int)

    clusterer = ClipClusterer(n_clusters=1, seed=0)
    clusterer.apply_to_dataset(dataset, embeddings, labels)

    for i, record in enumerate(dataset.records):
        np.testing.assert_allclose(
            record.embedding, embeddings[i].tolist(), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# core/clip_clusterer.py が PySide6 をインポートしていないことを確認（MVC分離）
# ---------------------------------------------------------------------------

def test_clip_clusterer_no_pyside6_import():
    """core/clip_clusterer.py が PySide6 をインポートしていないこと（MVC分離の確認）。"""
    import ast
    import pathlib

    source = pathlib.Path("core/clip_clusterer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "PySide6" not in alias.name, (
                    "core/clip_clusterer.py は PySide6 をインポートしてはならない"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "PySide6" not in module, (
                "core/clip_clusterer.py は PySide6 をインポートしてはならない"
            )
