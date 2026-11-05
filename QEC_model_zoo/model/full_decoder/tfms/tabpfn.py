import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tabpfn.constants import ModelVersion

from model.full_decoder.base_classifier import BaseClassifier

try:
    from tabpfn import TabPFNClassifier
except Exception as e:
    TabPFNClassifier = None
    _IMPORT_ERR = e

class TabPFNWrapper(nn.Module, BaseClassifier):
    def __init__(self, params, args):
        super().__init__()
        BaseClassifier.__init__(self, params, args)
        if TabPFNClassifier is None:
            raise RuntimeError(f"tabpfn not available: {_IMPORT_ERR}")

        self.device = (
            "cuda" if args.use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        self.num_features = args.num_features
        self.num_classes = args.num_classes

        n_ensembles = int(params.get("ensembles", 2))
        self.batch_size_inference = int(
            params.get("batch_size_inference", None)
        )
        self.max_ctx = int(params.get("max_context", 4096))

        variant = str(params.get("variant", "v2")).lower()
        versions = {
            "v2": ModelVersion.V2,
        }

        if variant not in versions:
            raise ValueError(f"Unsupported TabPFN variant: {variant}")

        checkpoint_path = params.get("checkpoint_path")
        seed = int(args.seed)

        common_kwargs = {
            "device": self.device,
            "random_state": seed,
            "n_estimators": n_ensembles,
            "fit_mode": "fit_with_cache",
            "inference_precision": "auto",
            "ignore_pretraining_limits": True,
        }

        if checkpoint_path is None:
            self.clf = TabPFNClassifier.create_default_for_version(
                versions[variant],
                **common_kwargs,
            )
        else:
            self.clf = TabPFNClassifier(
                model_path=checkpoint_path,
                **common_kwargs,
            )

        self._Xtr = None
        self._ytr = None

        # Make sure tabpfn (auth) cache files are stored in project storage
        import tabpfn.browser_auth as tabpfn_auth

        # Resolve project directory
        PROJECT_DIR = Path(
            os.environ.get("PROJECT_DIR")
            or Path(__file__).resolve().parents[4]
        )

        # Store TabPFN authentication outside /home
        AUTH_CACHE_DIR = PROJECT_DIR / ".cache" / "tabpfn-auth"
        AUTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Redirect TabPFN's internal authentication cache
        tabpfn_auth._CACHE_DIR = AUTH_CACHE_DIR
        tabpfn_auth._TOKEN_FILE = AUTH_CACHE_DIR / "auth_token"

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
    ) -> None:
        if self.max_ctx and X_train.shape[0] > self.max_ctx:
            rng = np.random.default_rng(int(self.clf.__dict__.get("random_state", 42)))
            idx = rng.choice(X_train.shape[0], self.max_ctx, replace=False)
            Xc, yc = X_train[idx], y_train[idx]
        else:
            Xc, yc = X_train, y_train
        self._Xtr, self._ytr = Xc, yc

        self.clf.fit(Xc, yc)

        super().fit(X_train, y_train, X_val, y_val)  # Simply sets is_trained to true

    @torch.inference_mode()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    @torch.inference_mode()
    def predict(self, X: np.ndarray) -> np.ndarray:
     return self.predict_proba(X).argmax(axis=1)
