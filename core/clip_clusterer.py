"""
CLIPベクトル化・クラスタリングロジック

モデル: openai/clip-vit-base-patch32（transformers + torch を使用）
PySide6 のUIクラスは一切インポートしない（MVC分離ルール厳守）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans

from core.dataset import Dataset

# transformers は実行時にのみインポートする（テスト時はモックで置き換えるため）
# モジュールレベルで名前を公開しておくとパッチが当たりやすくなる
try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except ImportError:
    CLIPModel = None       # type: ignore[assignment,misc]
    CLIPProcessor = None   # type: ignore[assignment,misc]


class ClipClusterer:
    """
    CLIP を使って画像をベクトル化し、K-Means でクラスタリングするクラス。

    使用方法::

        clusterer = ClipClusterer(n_clusters=10, seed=42)
        clusterer.load_model()
        embeddings = clusterer.encode_images(file_paths, batch_size=32)
        labels = clusterer.cluster(embeddings)
        clusterer.apply_to_dataset(dataset, embeddings, labels)
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, n_clusters: int, seed: int = 42) -> None:
        """
        Args:
            n_clusters: K-Means のクラスタ数。レコード数より大きい場合はキャップされる。
            seed: 乱数シード（再現性のため）
        """
        self.n_clusters = n_clusters
        self.seed = seed
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """CLIPモデルとプロセッサをロードする。

        モジュールスコープの CLIPModel / CLIPProcessor を参照することで
        テスト時に patch("core.clip_clusterer.CLIPModel") が有効になる。
        """
        import sys
        _mod = sys.modules[__name__]
        _CLIPModel = _mod.CLIPModel    # type: ignore[attr-defined]
        _CLIPProcessor = _mod.CLIPProcessor  # type: ignore[attr-defined]

        if _CLIPModel is None or _CLIPProcessor is None:
            raise RuntimeError(
                "transformers が見つかりません。"
                "uv add transformers でインストールしてください。"
            )
        self._processor = _CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self._model = _CLIPModel.from_pretrained(self.MODEL_NAME)
        self._model.eval()

    def encode_images(
        self,
        file_paths: list[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        """
        画像ファイルパスリストを CLIP でベクトル化して返す。

        Args:
            file_paths: 画像ファイルパスのリスト
            batch_size: バッチサイズ（OOM発生時は ClipWorker が外部で半減して再呼び出し）
            progress_callback: 進捗コールバック（0〜100 の int を受け取る）

        Returns:
            shape=(n_images, embedding_dim) の float32 numpy 配列

        Raises:
            RuntimeError: load_model() を呼び出す前に使用した場合
        """
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
            raise RuntimeError("encode_images() の前に load_model() を呼び出してください。")

        all_embeddings: list[np.ndarray] = []
        total = len(file_paths)

        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch_paths = file_paths[start: start + batch_size]
                images: list = []

                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except Exception:
                        # 画像が開けない場合はゼロ埋め画像で代替（スキップしない）
                        images.append(Image.new("RGB", (224, 224), color=0))

                inputs = self._processor(images=images, return_tensors="pt")
                features = self._model.get_image_features(**inputs)

                # L2 正規化（コサイン距離での K-Means に備える）
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
                all_embeddings.append(features.cpu().numpy())

                # 進捗コールバック（バッチ完了時点で計算）
                if progress_callback is not None:
                    done = min(start + batch_size, total)
                    progress_callback(int(done / total * 100))

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        K-Means クラスタリングを実行してクラスタラベル配列を返す。

        n_clusters がサンプル数を超える場合はサンプル数でキャップする
        （K-Means はクラスタ数 > サンプル数だとエラーになるため）。

        Args:
            embeddings: shape=(n, dim) の埋め込みベクトル配列

        Returns:
            shape=(n,) の int 配列（クラスタID、0始まり）
        """
        n_samples = len(embeddings)
        effective_clusters = min(self.n_clusters, n_samples)

        kmeans = KMeans(
            n_clusters=effective_clusters,
            random_state=self.seed,
            n_init="auto",
        )
        return kmeans.fit_predict(embeddings).astype(int)

    def apply_to_dataset(
        self,
        dataset: Dataset,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """
        各 ImageRecord に embedding と cluster_id を格納する（in-place）。

        仕様書の ClipWorker.run() 内要求::

            for record, cluster_id in zip(dataset.records, cluster_labels):
                record.embedding = embeddings[i].tolist()
                record.cluster_id = int(cluster_id)

        Args:
            dataset: 更新対象の Dataset
            embeddings: shape=(n, dim) の埋め込みベクトル配列
            cluster_labels: shape=(n,) のクラスタラベル配列
        """
        for i, record in enumerate(dataset.records):
            record.embedding = embeddings[i].tolist()
            record.cluster_id = int(cluster_labels[i])
