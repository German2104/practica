"""Map fitted encodings back to original factors before building pair terms."""

from __future__ import annotations

import typing

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ec_analysis.schemas.dataset import PortfolioDataset


@typing.final
class FeatureOwnership:
    def original_factors(self, pipeline: Pipeline, dataset: PortfolioDataset) -> tuple[str, ...]:
        return self.from_preprocessor(
            pipeline.named_steps["preprocess"], dataset.numeric_features, dataset.categorical_features
        )

    def from_preprocessor(
        self, preprocessing: ColumnTransformer, numeric_features: tuple[str, ...], categorical_features: tuple[str, ...]
    ) -> tuple[str, ...]:
        ownership: list[str] = []
        if numeric_features:
            numeric_pipeline = preprocessing.named_transformers_["numeric"]
            numeric_count = len(numeric_features)
            if "splines" in numeric_pipeline.named_steps:
                output_count = int(numeric_pipeline.named_steps["splines"].n_features_out_)
                if output_count % numeric_count:
                    raise ValueError("Spline outputs cannot be mapped to original factors")
                block_width = output_count // numeric_count
            else:
                block_width = 1
            ownership.extend(name for name in numeric_features for _ in range(block_width))
        if categorical_features:
            encoding = preprocessing.named_transformers_["categorical"].named_steps["encode"]
            if encoding.drop is not None or encoding.min_frequency is not None or encoding.max_categories is not None:
                raise ValueError("Attribution requires the configured full one-hot encoding")
            for factor_name, categories in zip(categorical_features, encoding.categories_, strict=True):
                ownership.extend([factor_name] * len(categories))
        return tuple(ownership)
