
import numpy as np
import torch

from tabicl import TabICLClassifier
from model.full_decoder.base_classifier import BaseClassifier


class TabICLV2Wrapper(BaseClassifier):
    """
    TabICLv2 wrapper optimized for:
    - Moderate GPU memory (~12 GB)
    - Batched inference
    - Repeated predictions on the same training context
    - Accuracy-preserving ensemble inference
    """

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.seed is None:
            raise ValueError("No seed provided.")

        if (
            not str(args.device).startswith("cuda")
            or not torch.cuda.is_available()
        ):
            raise ValueError("TabICLV2Wrapper requires CUDA.")

        self.seed = int(args.seed)
        self.device = args.device

        # Wrapper-level settings
        self.max_ctx = int(params.get("max_context", 4096))

        self.batch_size_inference = params.get("batch_size_inference", None)

        self.clf = TabICLClassifier(
            # Ensemble
            n_estimators=int(params.get("n_estimators", 8)),

            # Preserve original preprocessing settings
            norm_methods=params.get("norm_methods", "robust"),
            feat_shuffle_method=params.get(
                "feat_shuffle_method", "latin"
            ),
            class_shuffle_method=params.get(
                "class_shuffle_method", "none"
            ),
            outlier_threshold=float(
                params.get("outlier_threshold", 7.0)
            ),
            softmax_temperature=float(
                params.get("softmax_temperature", 0.7)
            ),
            average_logits=True,
            support_many_classes=True,

            # Ensemble members processed at once
            batch_size=int(params.get("batch_size", 2)),

            # Reuse training representations
            kv_cache=params.get("kv_cache", "repr"),

            # GPU optimizations
            use_amp=params.get("use_amp", "auto"),
            use_fa3="auto",
            offload_mode=params.get("offload_mode", "auto"),

            # Model configuration
            checkpoint_version=(
                "tabicl-classifier-v2-20260212.ckpt"
            ),
            disk_offload_dir="offload_tabicl",
            random_state=self.seed,
            device=self.device,
        )

        self._Xtr = None
        self._ytr = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:

        # Restrict context only when necessary
        if self.max_ctx > 0 and len(X_train) > self.max_ctx:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(
                len(X_train),
                size=self.max_ctx,
                replace=False,
            )
            Xc, yc = X_train[idx], y_train[idx]
        else:
            Xc, yc = X_train, y_train

        self._Xtr, self._ytr = Xc, yc

        # Builds the representation cache if enabled
        self.clf.fit(Xc, yc)

        # Mark wrapper as trained
        super().fit(X_train, y_train, X_val, y_val)

    @torch.inference_mode()
    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.clf.predict_proba(X)

    @torch.inference_mode()
    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        proba = self.predict_proba(X)

        # Preserve original class labels
        return self.clf.classes_[np.argmax(proba, axis=1)]
