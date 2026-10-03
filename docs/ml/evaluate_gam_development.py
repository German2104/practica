"""Replay the recorded development-only validation sweep; never evaluate test targets."""

from __future__ import annotations

import hashlib
import json
import pathlib
import typing

from ec_analysis.data.acquisition import file_checksum
from ec_analysis.data.generator import PortfolioGenerator
from ec_analysis.engine.attribution import PipelineAttributor
from ec_analysis.engine.candidates import CandidateFactory, RegressionModel
from ec_analysis.engine.evaluation import ModelEvaluator
from ec_analysis.engine.parameter_stability import ParameterStability
from ec_analysis.engine.splits import ForwardLoanSplit, select_loans
from ec_analysis.schemas.dataset import DatasetLoader
from ec_analysis.schemas.state import TuningTrial


@typing.final
class GAMDevelopmentCheck:
    def evaluate_validation(self, directory: pathlib.Path) -> dict[str, object]:
        input_path = PortfolioGenerator(sample_count=5000, random_seed=2027, scenario="credit").generate_csv(
            directory / "credit.csv"
        )
        dataset = DatasetLoader().load_csv(input_path)
        split = ForwardLoanSplit(random_seed=2027).create_split(dataset)
        training = select_loans(dataset, split.training_ids, "T0")
        validation = select_loans(dataset, split.validation_ids)
        rows: list[dict[str, object]] = []
        for penalty in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            variants: list[tuple[int, RegressionModel]] = []
            trials = []
            for profile in range(3):
                candidate = CandidateFactory(model_policy="fixed_gam", parameter_profile=profile).create_candidates(
                    dataset
                )[0]
                assert candidate.estimator is not None
                candidate.estimator.set_params(regressor__interaction_penalty=penalty)  # type: ignore[attr-defined]
                candidate.estimator.fit(training[list(dataset.feature_names)], training.npv)
                outcome = ModelEvaluator().evaluate_model(candidate.estimator, dataset, validation, "validation")
                variants.append((profile, candidate.estimator))
                trials.append(
                    TuningTrial(
                        profile=profile,
                        parameters={**candidate.parameters, "interaction_penalty": penalty},
                        status="completed",
                        validation_rmse=outcome.metrics.rmse,
                        signed_remainder=outcome.metrics.signed_remainder,
                        elapsed_seconds=0,
                    )
                )
            chosen = min(trials, key=lambda trial: typing.cast(float, trial.validation_rmse))
            reference = PipelineAttributor().explain_model(
                "stable_gam", dict(variants)[chosen.profile], dataset, validation, "validation"
            )
            sensitivity = ParameterStability().compare_variants(
                "stable_gam", dataset, validation, reference, variants, trials
            )
            rows.append(
                dict(
                    penalty=penalty,
                    trials=[trial.model_dump() for trial in trials],
                    sensitivity=sensitivity.model_dump(),
                )
            )
        return dict(
            method="development_credit_validation_only_v1",
            replayed_after_confirmation=True,
            original_development_seed=2027,
            input_sha256=file_checksum(input_path),
            split=split.model_dump(),
            test_targets_evaluated=False,
            original_fixed_penalty=1000.0,
            rows=rows,
            limitation="Replay of the observed development sweep; does not select or change the frozen v4 settings.",
        )


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    directory = root / "data/experiments/single-gam-development-replay"
    directory.mkdir(parents=True, exist_ok=False)
    result = GAMDevelopmentCheck().evaluate_validation(directory)
    result["helper_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    output = directory / "validation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
