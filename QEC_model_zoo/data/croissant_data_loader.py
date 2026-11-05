import json
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data.real.base_data_loader import BaseDataLoader


VALIDATION_FRACTION = 0.1


class CroissantDataLoader(BaseDataLoader):
    """
    Data loader for a croissant dataset for downstream AI-based QEC decoder.
    """

    croissant_metadata: Dict[str, Any]
    shot_column: str
    num_rounds_column: str
    target_column: str
    seed: int
    data_fraction: float
    dataset_name: str

    _train_df: pd.DataFrame
    _test_df: pd.DataFrame
    _val_df: pd.DataFrame
    _X_train: np.ndarray
    _y_train: np.ndarray
    _X_test: np.ndarray
    _y_test: np.ndarray
    _X_val: np.ndarray
    _y_val: np.ndarray

    def __init__(
        self,
        croissant_path: Path,
        seed: int = 42,
        data_fraction: float = 1.0,
        max_shots: int | None = None,
    ):
        """
        Initialize a croissant data loader.

        :param croissant_path: Path to the croissant dataset metadata.
        :param seed: Random seed.
        :param data_fraction: Fraction of data to load.
        :param max_shots: (optional) Maximum number of shots to load in train/test/validation data.
        """
        # Load relevant croissant metadata
        with open(croissant_path, "r") as f:
            self.croissant_metadata = json.load(f)

        self.seed = seed
        self.shot_column = self.croissant_metadata["shotColumn"]
        self.num_rounds_column = self.croissant_metadata["nRoundsColumn"]
        self.target_column = self.croissant_metadata["targetColumn"]
        self.data_fraction = data_fraction
        self.dataset_name = self.croissant_metadata["name"]
        self.max_shots = max_shots

        # Load actual data
        train_path = self._find_parquet_file_path("train")
        test_path = self._find_parquet_file_path("test")
        raw_path = self._find_parquet_file_path("raw")

        print("Loading training data from {}".format(train_path))
        df_shots_train = pd.read_parquet(train_path)
        print("Loading test data from {}".format(test_path))
        df_shots_test = pd.read_parquet(test_path)
        print("Loading raw data from {}".format(raw_path))
        df_raw = pd.read_parquet(raw_path)

        # Take subset of dataframe while retaining order
        df_shots_train = self._df_subset(df_shots_train)
        df_shots_test = self._df_subset(df_shots_test)
        self.test_shots = df_shots_test[self.shot_column].tolist()
        self.test_rounds_per_shot = df_shots_test[self.num_rounds_column].tolist()

        # Create validation data from training data
        df_shots_train, df_shots_val = self._split_df(
            df_shots_train,
            VALIDATION_FRACTION,
        )

        # Store non-raw dataframes. These remain the source of labels.
        self._train_df = df_shots_train
        self._val_df = df_shots_val
        self._test_df = df_shots_test

        train_shot_ids = df_shots_train[self.shot_column].tolist()
        test_shot_ids = df_shots_test[self.shot_column].tolist()
        val_shot_ids = df_shots_val[self.shot_column].tolist()

        # Set raw data for the same shots as each non-raw split.
        def get_raw_df_from_shot_ids(
            df: pd.DataFrame,
            shot_ids: List[int],
        ) -> pd.DataFrame:
            return df[df[self.shot_column].isin(shot_ids)].copy()

        self.raw_train_df = get_raw_df_from_shot_ids(df_raw, train_shot_ids)
        self.raw_val_df = get_raw_df_from_shot_ids(df_raw, val_shot_ids)
        self.raw_test_df = get_raw_df_from_shot_ids(df_raw, test_shot_ids)

    def get_feature_columns(
        self,
        feature_groups: List[str] | None = None,
        raw: bool = False,
    ) -> List[str]:
        """Inspect the configured input columns, in their model input order.

        Raw columns are per-round channels, not flattened shot sequences.
        This method reads metadata; it does not load or convert feature arrays.
        """
        if raw:
            return list(self.croissant_metadata["rawFeatures"])

        mapping = {
            group.lower(): columns
            for group, columns in self.get_feature_group_mapping().items()
        }

        if feature_groups is None:
            feature_groups = self.croissant_metadata.get(
                "variableMeasured", list(mapping)
            )

        return [
            column
            for group in feature_groups
            for column in mapping.get(group.lower(), [])
        ]

    def with_feature_aliases(
        self,
        feature_aliases: Dict[str, str],
    ) -> "CroissantDataLoader":
        """Return a copy with explicit aliases for engineered feature columns.

        Dataframe containers and metadata are copied; underlying data arrays
        are shared. Originals, labels, raw channels and shot features are
        unchanged. Ambiguous renames (two columns to one name) are rejected.
        """
        declared = {
            column
            for columns in self.get_feature_group_mapping().values()
            for column in columns
        }

        aliases = {
            source: target
            for source, target in feature_aliases.items()
            if source in declared and source != target
        }

        if any(
            not isinstance(target, str) or not target
            for target in aliases.values()
        ):
            raise ValueError(
                "Feature alias targets must be non-empty column names."
            )

        protected = {
            self.shot_column,
            self.target_column,
            self.num_rounds_column,
        }

        if (
            protected.intersection(aliases)
            or protected.intersection(aliases.values())
        ):
            raise ValueError(
                "Feature aliases cannot rename or replace "
                "shot IDs, labels or round counts."
            )

        selected = copy(self)
        selected.croissant_metadata = deepcopy(self.croissant_metadata)
        selected.croissant_metadata["featureGroupsMapping"] = {
            group: [aliases.get(column, column) for column in columns]
            for group, columns in self.get_feature_group_mapping().items()
        }

        for split in ("train", "val", "test"):
            frame = getattr(self, f"_{split}_df")
            columns = [
                aliases.get(column, column)
                for column in frame.columns
            ]

            if len(columns) != len(set(columns)):
                raise ValueError(
                    f"Feature aliases create duplicate columns "
                    f"in the {split} split."
                )

            renamed = frame.copy(deep=False)
            renamed.columns = columns
            setattr(selected, f"_{split}_df", renamed)

        return selected

    def with_common_features(
        self,
        other: "CroissantDataLoader",
        feature_groups: List[str] | None = None,
        feature_aliases: Dict[str, str] | None = None,
    ) -> Tuple["CroissantDataLoader", "CroissantDataLoader"]:
        """Return two loader copies configured for an exact-name intersection.

        With no groups supplied, each loader's own default groups are used.
        An explicit list selects those groups in both datasets. Only columns
        present in every train/validation/test split of both loaders are kept.
        Both copies use this loader's column order, including raw channels and
        optional shot features. Duplicate engineered columns are removed.

        Original loaders and metadata are unchanged. Dataframes are shared to
        avoid reloading or copying the datasets; treat them as read-only.
        Labels, shot order, round counts and sequence lengths are unchanged.
        Matching names do not establish matching physical feature semantics.
        Explicit ``feature_aliases`` are applied to engineered inputs in both
        copies before intersecting names. Aliases are disabled by default.
        """
        if feature_aliases:
            return self.with_feature_aliases(
                feature_aliases
            ).with_common_features(
                other.with_feature_aliases(feature_aliases),
                feature_groups=feature_groups,
            )

        def available_columns(loader, raw=False):
            frames = (
                (loader.raw_train_df, loader.raw_val_df, loader.raw_test_df)
                if raw
                else (loader._train_df, loader._val_df, loader._test_df)
            )
            return set.intersection(
                *(set(frame.columns) for frame in frames)
            )

        def intersection(left, right, raw=False):
            available = (
                set(right)
                & available_columns(self, raw)
                & available_columns(other, raw)
            )
            return list(
                dict.fromkeys(col for col in left if col in available)
            )

        common = intersection(
            self.get_feature_columns(feature_groups),
            other.get_feature_columns(feature_groups),
        )

        if not common:
            raise ValueError(
                "No common engineered features in the selected groups and "
                "all train/validation/test dataframes of both datasets."
            )

        raw_common = intersection(
            self.get_feature_columns(raw=True),
            other.get_feature_columns(raw=True),
            raw=True,
        )

        shot_common = intersection(
            self.get_shot_feature_columns(),
            other.get_shot_feature_columns(),
            raw=True,
        )

        # Retain group names for inspection, with one shared order/schema.
        groups = (
            feature_groups
            if feature_groups is not None
            else self.croissant_metadata.get(
                "variableMeasured",
                list(self.get_feature_group_mapping()),
            )
        )

        source_mapping = {
            group.lower(): columns
            for group, columns in self.get_feature_group_mapping().items()
        }

        remaining = set(common)
        mapping = {}

        for group in groups:
            key = group.lower()
            mapping.setdefault(key, [])

            for column in source_mapping.get(key, []):
                if column in remaining:
                    mapping[key].append(column)
                    remaining.remove(column)

        aligned = []

        for loader in (self, other):
            selected = copy(loader)
            selected.croissant_metadata = deepcopy(loader.croissant_metadata)
            selected.croissant_metadata["variableMeasured"] = list(mapping)
            selected.croissant_metadata["featureGroupsMapping"] = deepcopy(
                mapping
            )
            selected.croissant_metadata["rawFeatures"] = list(raw_common)
            selected.croissant_metadata["shotFeaturesSequence"] = list(
                shot_common
            )
            selected._common_features_aligned = True
            aligned.append(selected)

        return aligned[0], aligned[1]

    def preprocess_raw_df_to_features(
        self,
        raw_df: pd.DataFrame,
        shot_ids: List[int],
    ) -> np.ndarray:
        raw_features = self.croissant_metadata["rawFeatures"]

        if getattr(self, "_common_features_aligned", False) and not raw_features:
            raise ValueError(
                "The aligned datasets have no common raw feature channels."
            )

        missing_features = [
            feature
            for feature in raw_features
            if feature not in raw_df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Raw features missing from dataframe: {missing_features}"
            )

        raw_df = raw_df.sort_values(
            [self.shot_column, self.croissant_metadata["rawRoundColumn"]]
        )

        grouped = {
            shot_id: group[raw_features].to_numpy(dtype=np.float32)
            for shot_id, group in raw_df.groupby(
                self.shot_column,
                sort=False,
            )
        }

        try:
            sequences = [
                grouped[shot_id]
                for shot_id in shot_ids
            ]
        except KeyError as e:
            raise ValueError(
                f"No raw data found for shot_id={e.args[0]}"
            ) from e

        X = np.empty(
            len(sequences),
            dtype=object,
        )

        X[:] = sequences
        return X

    def raw_df_to_features_and_labels(
        self,
        raw_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw round-level data to shot sequences and pair them with labels.

        Features come from ``raw_df``. Labels and shot ordering always come from
        the corresponding non-raw ``label_df``.

        :param raw_df: Raw round-level dataframe.
        :param label_df: Corresponding non-raw split containing one row per shot.
        :return: Tuple of variable-length raw feature sequences and shot labels.
        """
        shot_ids = label_df[self.shot_column].tolist()

        X = self.preprocess_raw_df_to_features(
            raw_df=raw_df,
            shot_ids=shot_ids,
        )

        y = label_df[self.target_column].to_numpy(dtype=np.uint8)

        if len(X) != len(y):
            raise RuntimeError(
                "Number of raw shot sequences does not match number of labels."
            )

        return X, y

    def get_train_df(self) -> pd.DataFrame:
        """
        Get dataframe with training data.

        :return: Train data df.
        """
        if self._train_df is None:
            raise ValueError("Training data not available.")
        return self._train_df

    def df_to_features_and_labels(
        self,
        df: pd.DataFrame,
        feature_groups: List[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turn a dataframe containing data (train, test, validation) into a feature
        matrix and labels, optionally selecting specific feature groups.

        :param df: Dataframe to convert.
        :param feature_groups: Optional list of feature groups to select, defaults
                               to all available feature groups.
        :return: Tuple of feature matrix and labels arrays.
        """
        if feature_groups is None and "variableMeasured" in self.croissant_metadata:
            # Set feature groups to default defined in croissant
            feature_groups = self.croissant_metadata["variableMeasured"]

        if feature_groups is None:
            # Use all feature groups by default if not defined
            feature_groups = self.croissant_metadata["featureGroups"].keys()


        group_mapping = self.get_feature_group_mapping()
        features = self._get_features_from_groups(
            group_mapping,
            feature_groups,
        )

        # Calculate target column
        y = df[self.target_column].to_numpy(dtype=np.uint8)

        # Calculate feature matrices
        valid_features = [
            f for f in features
            if f in df.columns.tolist()
        ]
        X = df[valid_features].to_numpy(dtype=np.float32)
        return X, y

    def get_test_df(self) -> pd.DataFrame:
        """
        Get dataframe with test data.

        :return: Test data df.
        """
        if self._test_df is None:
            raise ValueError("Testing data not available.")
        return self._test_df

    def get_val_df(self) -> pd.DataFrame:
        """
        Get dataframe with validation data.

        :return: Validation data df.
        """
        if self._val_df is None:
            raise ValueError("Validation data not available.")
        return self._val_df

    def get_feature_group_mapping(self) -> Dict[str, List[str]]:
        """
        Get feature group -> feature mapping.

        :return: Mapping of feature groups to feature column names.
        """
        return self.croissant_metadata["featureGroupsMapping"]

    def get_test_shot_info(self) -> Tuple[List[int], List[int]]:
        """
        Get information about the shots in the test dataset.

        :return: Tuple of a list of test shot ids and a list of the corresponding
                 amount of rounds for the shot.
        """
        return self.test_shots, self.test_rounds_per_shot

    def get_shot_feature_columns(self) -> List[str]:
        """Return the shot-level feature columns declared in Croissant metadata.

        The metadata key ``shotFeaturesSequence`` must contain a list of numeric
        columns in the raw dataframe.
        """
        columns = self.croissant_metadata.get(
            "shotFeaturesSequence",
            self.croissant_metadata.get("shotFeatures", []),
        )

        if columns is None:
            return []

        if not isinstance(columns, list) or not all(
            isinstance(column, str)
            for column in columns
        ):
            raise ValueError(
                "Croissant metadata 'shotFeaturesSequence' "
                "must be a list of column names."
            )

        columns = list(dict.fromkeys(columns))

        protected_columns = {
            self.shot_column,
            self.target_column,
            self.croissant_metadata.get("rawRoundColumn"),
        }

        forbidden = [
            column
            for column in columns
            if column in protected_columns
        ]

        if forbidden:
            raise ValueError(
                f"Shot features cannot include shot IDs or targets: {forbidden}."
            )

        return columns

    def get_shot_features(self, split: str) -> np.ndarray:
        """Get shot-level features from each shot's final raw-data round.

        The final round is the row with the greatest numeric value in the
        metadata's ``rawRoundColumn``. Results follow the processed split's
        shot order, matching the raw sequence getter.
        Returns an ``(n_shots, 0)`` array when ``shotFeaturesSequence`` is absent or
        empty, allowing callers to keep shot-level inputs optional.
        """
        split = split.lower()

        split_frames = {
            "train": (self._train_df, self.raw_train_df),
            "val": (self._val_df, self.raw_val_df),
            "test": (self._test_df, self.raw_test_df),
        }

        if split not in split_frames:
            raise ValueError(
                f"Unknown split {split!r}. Expected 'train', 'val', or 'test'."
            )

        label_frame, raw_frame = split_frames[split]
        columns = self.get_shot_feature_columns()

        if not columns:
            return np.empty(
                (len(label_frame), 0),
                dtype=np.float32,
            )

        round_column = self.croissant_metadata["rawRoundColumn"]
        required = [self.shot_column, round_column, *columns]

        missing = [
            column
            for column in required
            if column not in raw_frame
        ]

        if missing:
            raise ValueError(
                f"Columns missing from raw {split} dataframe: {missing}."
            )

        try:
            round_order = pd.to_numeric(
                raw_frame[round_column],
                errors="raise",
            )

            sorted_raw = raw_frame.assign(
                _shot_feature_round_order=round_order
            ).sort_values(
                [self.shot_column, "_shot_feature_round_order"]
            )

            final_rounds = sorted_raw.drop_duplicates(
                subset=[self.shot_column],
                keep="last",
            ).set_index(self.shot_column)

            shot_ids = label_frame[self.shot_column].tolist()

            missing_shots = [
                shot_id
                for shot_id in shot_ids
                if shot_id not in final_rounds.index
            ]

            if missing_shots:
                raise ValueError(
                    f"Raw {split} data has no final round for "
                    f"{len(missing_shots)} shots; examples: {missing_shots[:10]}"
                )

            values = final_rounds.loc[shot_ids, columns].to_numpy(
                dtype=np.float32
            )

        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("Raw "):
                raise

            raise ValueError(
                "Shot-level features must be numeric. Encode categorical "
                "columns before training, and ensure raw round IDs are numeric."
            ) from exc

        if not np.isfinite(values).all():
            raise ValueError(
                f"Shot-level features for split {split!r} "
                "contain NaN or infinity."
            )

        return values

    def get_train(
        self,
        raw: bool = False,
        include_shot_features: bool | None = False,
    ):
        """Get training data and, when declared, its per-shot features.

        Pass ``include_shot_features=None`` to include shot features
        automatically when metadata declares them. ``False`` retains the
        legacy ``(X, y)`` result; ``True`` requests a three-item result.
        """
        if raw:
            X, y = self.raw_df_to_features_and_labels(
                raw_df=self.raw_train_df,
                label_df=self._train_df,
            )
        else:
            X, y = self.df_to_features_and_labels(self._train_df)

        if include_shot_features is None:
            include_shot_features = bool(self.get_shot_feature_columns())

        if include_shot_features:
            return X, y, self.get_shot_features("train")

        return X, y

    def get_test(
        self,
        raw: bool = False,
        include_shot_features: bool | None = False,
    ):
        """Get test data and, when declared, its per-shot features.

        Pass ``include_shot_features=None`` to include shot features
        automatically when metadata declares them. ``False`` retains the
        legacy ``(X, y)`` result; ``True`` requests a three-item result.
        """
        if raw:
            X, y = self.raw_df_to_features_and_labels(
                raw_df=self.raw_test_df,
                label_df=self._test_df,
            )
        else:
            X, y = self.df_to_features_and_labels(self._test_df)

        if include_shot_features is None:
            include_shot_features = bool(self.get_shot_feature_columns())

        if include_shot_features:
            return X, y, self.get_shot_features("test")

        return X, y

    def get_val(
        self,
        raw: bool = False,
        include_shot_features: bool | None = False,
    ):
        """Get validation data and, when declared, its per-shot features.

        Pass ``include_shot_features=None`` to include shot features
        automatically when metadata declares them. ``False`` retains the
        legacy ``(X, y)`` result; ``True`` requests a three-item result.
        """
        if raw:
            X, y = self.raw_df_to_features_and_labels(
                raw_df=self.raw_val_df,
                label_df=self._val_df,
            )
        else:
            X, y = self.df_to_features_and_labels(self._val_df)

        if include_shot_features is None:
            include_shot_features = bool(self.get_shot_feature_columns())

        if include_shot_features:
            return X, y, self.get_shot_features("val")

        return X, y

    def get_info(self, raw: bool = False) -> Dict[str, Any]:
        """
        Get dataset information.

        :param raw: If true, get information from raw data, otherwise from feature engineered data.
        :return: Dataset information.
        """
        info = {
            split: self.get_info_for_split(split, raw=raw)
            for split in ["train", "test", "val"]
        }
        info["dataset_name"] = self.dataset_name
        return info

    def get_info_for_split(
        self,
        split: str,
        raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Get dataset information for a specific split (train, test, val).

        :param split: Name of the split.
        :param raw: If true, get information from raw data, otherwise from feature engineered data.
        :return: Dataset information.
        """
        for s, getter in zip(
            ["train", "test", "val"],
            [self.get_train, self.get_test, self.get_val],
        ):
            if split == s:
                # Dataset summaries only need sequence/feature matrices and
                # labels, even if shot-level columns are configured.
                X, y = getter(raw, include_shot_features=False)

                return {
                    "n_features": (
                        len(self.croissant_metadata["rawFeatures"])
                        if raw
                        else X.shape[1]
                    ),
                    "n_classes": len(np.unique(y)),
                    "n_samples": y.shape[0],
                }

        return {}

    def get_stim_file_mapping(self) -> Path:
        """
        Get path to file containing stim file mapping to shot_ids, used in baseline
        full decoders.

        :return: File path if defined, otherwise raises an error.
        """
        if "stimFileMappingPath" not in self.croissant_metadata:
            raise ValueError(
                "Stim file mapping path not defined in croissant metadata."
            )

        return Path(self.croissant_metadata["stimFileMappingPath"])

    def _find_parquet_file_path(self, matching_name: str) -> Path:
        """
        Find a parquet file whose filename stem exactly matches matching_name.

        Examples:
            matching_name="train" -> train.parquet
            matching_name="test"  -> test.parquet
            matching_name="raw"   -> raw.parquet
        """
        options = self.croissant_metadata["distribution"]

        for option in options:
            # Normalize Windows-style paths so they also work on Linux.
            file_path = Path(
                option["contentUrl"].replace("\\", "/")
            )

            if file_path.suffix.lower() != ".parquet":
                continue

            if file_path.stem.lower() == matching_name.lower():
                return file_path

        raise ValueError(
            f"No parquet file found exactly matching "
            f"'{matching_name}.parquet'."
        )

    def _df_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Take random subset of dataframe while retaining order.

        :param df: Input dataframe.
        :return: Random subset of dataframe.
        """
        fract = self.data_fraction
        seed = self.seed

        if (fract is None or fract >= 1) and self.max_shots is None:
            return df

        if fract <= 0:
            return df.iloc[:0]

        # Randomly sample row positions, then sort them to retain original row order.
        n = len(df)
        sample_size = int(n * fract)

        rng = np.random.default_rng(seed)
        positions = rng.choice(
            n,
            size=sample_size,
            replace=False,
        )

        # Limit amount of rows if specified
        if self.max_shots is not None:
            positions = positions[: self.max_shots]

        positions.sort()

        return df.iloc[positions]

    def _split_df(
        self,
        df: pd.DataFrame,
        fraction: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a dataframe into two parts randomly.
        Retains order.
        First element has 1 - fraction portion of elements while the other has
        fraction portion of elements.

        :param df: Dataframe to be split.
        :param fraction: Fraction of elements in the second returned element.
        :return: Tuple of two dataframes.
        """
        seed = self.seed

        if fraction <= 0:
            return df, df.iloc[:0]

        if fraction >= 1:
            return df.iloc[:0], df

        n = len(df)
        sample_size = int(n * fraction)

        rng = np.random.default_rng(seed)
        second_positions = rng.choice(
            n,
            size=sample_size,
            replace=False,
        )
        second_positions.sort()

        mask = np.zeros(n, dtype=bool)
        mask[second_positions] = True

        first = df.iloc[~mask]
        second = df.iloc[mask]

        return first, second

    def _get_features_from_groups(
        self,
        feature_groups: Dict[str, List[str]],
        feature_sets: List[str],
    ) -> List[str]:
        """
        Given feature groups, return a list of feature names. Unknown feature groups
        are ignored.

        :param feature_groups: Feature groups.
        :param feature_sets: Feature sets.
        :return: List of feature names.
        """
        cols = []

        for feature_set in feature_sets:
            if feature_set.lower() not in [
                group.lower()
                for group in feature_groups.keys()
            ]:
                continue  # ignore unknown feature sets

            for feature in feature_groups[feature_set.lower()]:
                cols.append(feature)

        return cols
