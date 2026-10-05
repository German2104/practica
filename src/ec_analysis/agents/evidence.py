"""Business wording computed from checked fields; no user/model prose enters facts."""

from __future__ import annotations

import dataclasses
import hashlib
import typing

from ec_analysis.schemas.interpretation import EvidenceCatalog, EvidenceFact
from ec_analysis.schemas.state import AnalysisState, ReliabilityAssessment

LABELS: typing.Final = {
    "volume": "Объём",
    "interest_rate": "Процентная ставка",
    "pd": "Вероятность дефолта",
    "lgd": "Доля потерь при дефолте",
    "ltv": "Отношение кредита к залогу",
    "linear": "линейная зависимость",
    "threshold": "пороговый эффект",
    "saturation": "насыщение",
    "u_shaped": "U-образная зависимость",
    "inverted_u_shaped": "перевёрнутая U-образная зависимость",
    "regime": "разные режимы наклона",
    "complex": "сложная нелинейная форма",
    "unknown": "форма не определена",
}


@typing.final
@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EvidenceBuilder:
    def build_catalog(self, state: AnalysisState) -> EvidenceCatalog:
        checked_state = AnalysisState.model_validate(state.model_dump())
        facts = [
            self._fact(
                "change",
                f"Показатель {state.target_metric}: исходное значение {self._number(state.initial_value)}, "
                f"итоговое {self._number(state.final_value)}, изменение {self._number(state.total_change)}.",
                ["initial_value", "final_value", "total_change"],
                "summary",
                True,
            ),
            self._fact(
                "remainder",
                f"Необъяснённый остаток: {self._number(state.unexplained_remainder)}. "
                + (
                    "Подписанная доля относительно модуля изменения: "
                    f"{self._number(100 * state.unexplained_fraction)}%."
                    if state.unexplained_fraction is not None
                    else "Доля остатка не определена для этого изменения."
                ),
                ["unexplained_remainder", "unexplained_fraction"],
                "quality",
                True,
            ),
            self._fact(
                "reliability",
                "Надёжность объяснения подтверждена проверками этого перехода."
                if state.selection_status == "explanation_validated"
                else (
                    "Надёжное объяснение пока не подтверждено. Числа доступны для исследования; вывод предварительный."
                ),
                ["selection_status", "confidence_level", "validation_reliability", "test_reliability"],
                "quality",
                True,
            ),
            self._fact(
                "model",
                f"В результате анализа указана модель {state.selected_model}.",
                ["selected_model", "selection_reason", "protocol_version"],
                "quality",
            ),
            self._fact(
                "causality",
                "Вклады описывают обученную модель и не доказывают причинность. "
                "Результат одной пары периодов нельзя переносить на все будущие периоды "
                "без дополнительных проверок.",
                ["limitations"],
                "limitations",
                True,
            ),
            self._fact(
                "unsupported",
                "В сохранённых расчётах недостаточно сведений для ответа на этот вопрос.",
                ["limitations"],
                "limitations",
            ),
        ]
        for assessment in (state.validation_reliability, state.test_reliability):
            if assessment is not None:
                facts.append(self._quality_fact(assessment))
        for factor_index, factor in enumerate(state.factor_contributions):
            facts.append(
                self._fact(
                    f"factor:{factor_index}",
                    f"Вклад фактора «{LABELS.get(factor.factor_name, factor.factor_name)}» в изменение: "
                    f"{self._number(factor.contribution)} в единицах {state.target_metric}.",
                    [f"factor_contributions/{factor_index}/contribution", factor.calculation_ref],
                    "factors",
                )
            )
        for pair_index, interaction in enumerate(state.interactions):
            facts.append(
                self._fact(
                    f"pair:{pair_index}",
                    f"Совместный вклад «{LABELS.get(interaction.factors[0], interaction.factors[0])}» и "
                    f"«{LABELS.get(interaction.factors[1], interaction.factors[1])}»: "
                    f"{self._number(interaction.contribution)}. Эта сумма учтена отдельно от основных вкладов.",
                    [f"interactions/{pair_index}/contribution", interaction.calculation_ref],
                    "interactions",
                )
            )
        if state.final_test is not None:
            facts.append(
                self._fact(
                    "test",
                    f"На независимом тесте RMSE = {self._number(state.final_test.rmse)}, "
                    f"MAE = {self._number(state.final_test.mae)}. Это ошибки прогноза в единицах целевой метрики.",
                    ["final_test/rmse", "final_test/mae"],
                    "quality",
                )
            )
        if state.target_metric == "model_npv":
            facts.append(
                self._fact(
                    "model-target",
                    "model_npv — модельная целевая величина по сценарным допущениям проекта, "
                    "а не измеренный банковский NPV. Проверки модели не подтверждают правильность этих допущений.",
                    ["target_metric"],
                    "limitations",
                    True,
                )
            )
        for candidate_index, candidate in enumerate(state.benchmark_summary):
            if candidate.stability is not None and candidate.stability.stability_score is not None:
                score_value = candidate.stability.stability_score
                if candidate.is_selected:
                    facts.append(
                        self._fact(
                            "stability",
                            f"Бутстрэп устойчивости выбранной модели: "
                            f"{self._number(score_value)}." + (" Вклады неустойчивы." if score_value < 0.7 else ""),
                            [f"benchmark_summary/{candidate_index}/stability"],
                            "quality",
                            score_value < 0.7,
                        )
                    )
            if candidate.is_selected and candidate.parameter_sensitivity is not None:
                sensitivity = candidate.parameter_sensitivity
                parameter_status = {
                    "completed": "пройдена",
                    "unstable": "вклады неустойчивы",
                    "incomplete": "не завершена",
                    "no_signal": "нет ненулевого сигнала для оценки",
                }[sensitivity.status]
                facts.append(
                    self._fact(
                        "parameters",
                        f"Проверка вкладов при смене параметров: {parameter_status}. "
                        f"Минимальное сходство {self._number(sensitivity.minimum_similarity)} "
                        f"при пороге {self._number(sensitivity.similarity_threshold)}; "
                        f"минимальное согласие знаков {self._number(sensitivity.minimum_sign_agreement)} "
                        f"при пороге {self._number(sensitivity.sign_threshold)}.",
                        [f"benchmark_summary/{candidate_index}/parameter_sensitivity"],
                        "quality",
                        True,
                    )
                )
        if state.global_importance is not None:
            facts.append(
                self._fact(
                    "importance-method",
                    "Глобальная важность показывает рост ошибки при "
                    "перемешивании фактора на валидации. Она отличается от вклада в изменение портфеля; "
                    "коррелированные факторы могут разделять важность.",
                    ["global_importance/method", "global_importance/warnings"],
                    "importance",
                )
            )
            for factor_index, importance in enumerate(state.global_importance.factors):
                facts.append(
                    self._fact(
                        f"importance:{factor_index}",
                        f"Глобальная важность «{LABELS.get(importance.factor_name, importance.factor_name)}»: "
                        f"средний рост RMSE {self._number(importance.mean_rmse_increase)}, "
                        f"разброс {self._number(importance.standard_deviation)}.",
                        [f"global_importance/factors/{factor_index}", importance.calculation_ref],
                        "importance",
                    )
                )
        if state.shape_diagnostics is not None:
            for factor_index, shape in enumerate(state.shape_diagnostics.factors):
                facts.append(
                    self._fact(
                        f"shape:{factor_index}",
                        f"Для «{LABELS.get(shape.factor_name, shape.factor_name)}» условное сечение модели показывает: "
                        f"{LABELS[shape.shape]}. Это диагностический вывод в обучающем диапазоне, при фиксированных "
                        "остальных факторах; он не доказывает причинность.",
                        [f"shape_diagnostics/factors/{factor_index}", shape.calculation_ref],
                        "shapes",
                    )
                )
        for limitation_index, limitation in enumerate(state.limitations):
            facts.append(
                self._fact(
                    f"limitation:{limitation_index}",
                    self._explain_limitation(limitation),
                    [f"limitations/{limitation_index}"],
                    "limitations",
                )
            )
        return EvidenceCatalog(
            analysis_sha256=hashlib.sha256(checked_state.model_dump_json().encode()).hexdigest(),
            target_metric=state.target_metric,
            facts=facts,
        )

    def _quality_fact(self, assessment: ReliabilityAssessment) -> EvidenceFact:
        scope_name = "на валидации" if assessment.scope == "validation" else "на независимом тесте"
        status_name = {"accepted": "пройдены", "rejected": "отклонены", "incomplete": "не завершены"}[assessment.status]
        measured = assessment.measurements
        thresholds = assessment.thresholds
        statements = [
            f"Проверки {scope_name}: {status_name}.",
            f"Нормированная RMSE {self._number(measured.get('normalized_rmse'))}, "
            f"предел {self._number(thresholds.get('maximum_normalized_rmse'))}.",
            f"Модуль остатка относительно масштаба существенности "
            f"{self._number(measured.get('residual_share'))}, "
            f"предел {self._number(thresholds.get('maximum_residual_share'))}.",
        ]
        if assessment.reasons:
            statements.append("Причины: " + " ".join(self._quality_reason(reason) for reason in assessment.reasons))
        return self._fact(
            f"quality:{assessment.scope}",
            " ".join(statements),
            [
                f"{assessment.scope}_reliability/{field}"
                for field in ("status", "measurements", "thresholds", "reasons")
            ],
            "quality",
            assessment.status != "accepted",
        )

    def _quality_reason(self, reason_text: str) -> str:
        translations = {
            "Prediction error exceeds": "Прогнозная ошибка превышает настроенный предел.",
            "Unexplained remainder exceeds": "Необъяснённый остаток превышает настроенный предел существенности.",
            "A complete bootstrap": "Недостаточно завершённых повторов бутстрэпа.",
            "Stability or material-effect": "Устойчивость или согласие знаков существенных эффектов не определены.",
            "Attribution is below": "Устойчивость вкладов ниже настроенного порога.",
            "Signs of material effects": "Знаки существенных эффектов недостаточно согласованы.",
            "Normalized prediction error was not": "Нормированная прогнозная ошибка не измерена.",
            "Checked attribution is unavailable": "Проверенные вклады недоступны.",
            "Attribution failed": "Расчёт вкладов завершился ошибкой.",
            "Parameter attribution sensitivity: unstable": "Вклады неустойчивы при смене параметров модели.",
            "Parameter attribution sensitivity: no_signal": "Нет ненулевого сигнала для проверки смены параметров.",
            "Parameter attribution sensitivity: incomplete": "Проверка смены параметров не завершена.",
        }
        for reason_prefix, explanation in translations.items():
            if reason_text.startswith(reason_prefix):
                return explanation
        return "Дополнительная причина отказа сохранена в поле reasons; проверенной формулировки в каталоге нет."

    def _fact(
        self,
        identifier: str,
        text: str,
        references: list[str],
        category: str,
        required: bool = False,
    ) -> EvidenceFact:
        return EvidenceFact(
            identifier=identifier, text=text, references=references, category=category, required=required
        )

    def _number(self, measured_value: float | None) -> str:
        if measured_value is None:
            return "не определено"
        return format(measured_value, ".8g").replace(".", ",")

    def _explain_limitation(self, limitation: str) -> str:
        translations = {
            "Attributions describe": "Вклады относятся к модели, а не к причинным эффектам или глобальной важности.",
            "Portfolio effects": (
                "Вклады рассчитаны по всему входному портфелю; ошибки проверяются на отдельных клиентах."
            ),
            "One T0/T1": "Проверка одного перехода не подтверждает надёжность на всех временных окнах.",
            "Prototype quality": "Критерии учебного прототипа не являются банковской сертификацией риска.",
            "Reliable explanation": "Надёжное объяснение не подтверждено: изучите причины отказа проверок.",
            "No calendar": "Календарные даты не предоставлены; проверяется заявленный порядок периодов.",
            "Bootstrap stability was not": "Бутстрэп устойчивости не выполнен.",
            "Bootstrap measures": "Бутстрэп оценивает чувствительность к составу обучения на фиксированной валидации.",
            "Exploratory training": "Квантили бутстрэпа не являются откалиброванными доверительными интервалами.",
            "Parameter attribution sensitivity was not": "Чувствительность вкладов к параметрам модели не проверена.",
            "Parameter sensitivity did not": (
                "Проверка смены параметров не пройдена; надёжное объяснение не подтверждено."
            ),
            "Unstable attribution": "Вклады неустойчивы по настроенному критерию.",
            "Attribution is below": "Устойчивость вкладов ниже настроенного порога.",
            "Some repeats": "Часть повторных обучений завершилась ошибкой; общая оценка устойчивости не выдана.",
            "No nonzero": "Нет ненулевых эффектов или успешных повторов; устойчивость не определена.",
        }
        for prefix, translated in translations.items():
            if limitation.startswith(prefix):
                return translated
        return (
            "В расчёте зарегистрировано дополнительное ограничение; "
            "его точный текст доступен по ссылке на поле результата."
        )
