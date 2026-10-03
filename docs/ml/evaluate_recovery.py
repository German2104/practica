"""Measure known synthetic shapes for every saved model on identical training grids.

This post-training diagnostic does not alter model selection or the final test.
Only models created by this project and matching their own manifests are loaded.
"""
from pathlib import Path
import hashlib
import json
import runpy

import joblib
import numpy as np
import pandas as pd

from ec_analysis.engine.sections import ModelSections, ModelCurve, ModelSurface

ROOT=Path(__file__).resolve().parents[2]
EVALUATE=runpy.run_path(str(ROOT/'scripts/evaluate_recovery.py'))['evaluate_sections']
NAMES={'mean':'Среднее','ridge':'Ridge','gam_spline':'GAM','xgboost':'XGBoost','ebm':'EBM',
       'quadratic_ridge':'Ridge²','gam_pairwise':'GAM с парами'}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_sections(estimator, template, name):
    reference=template.reference
    baseline=float(estimator.predict(pd.DataFrame([reference]))[0])
    curves=[];surfaces=[]
    for curve in template.curves:
        frame=pd.DataFrame([reference]*len(curve.values));frame[curve.factor]=curve.values
        curves.append(ModelCurve(factor=curve.factor,values=curve.values,
                                 centered_predictions=(estimator.predict(frame)-baseline).tolist()))
    for surface in template.surfaces:
        first,second=surface.factors
        frame=pd.DataFrame([reference]*(len(surface.first_values)*len(surface.second_values)))
        frame[first]=np.repeat(surface.first_values,len(surface.second_values))
        frame[second]=np.tile(surface.second_values,len(surface.first_values))
        first_frame=frame.copy();first_frame[second]=reference[second]
        second_frame=frame.copy();second_frame[first]=reference[first]
        contrast=estimator.predict(frame)-estimator.predict(first_frame)-estimator.predict(second_frame)+baseline
        surfaces.append(ModelSurface(factors=surface.factors,first_values=surface.first_values,
                                      second_values=surface.second_values,
                                      interaction_contrast=contrast.reshape(len(surface.first_values),len(surface.second_values)).tolist()))
    return ModelSections(model_name=name,reference=reference,reference_prediction=baseline,curves=curves,surfaces=surfaces)


def main():
    experiments=ROOT/'data/experiments/confirmation-v2'
    summary=json.loads((experiments/'summary.json').read_text())
    results=[]
    for record in summary['scenarios']:
        scenario=record['scenario'];directory=experiments/scenario
        manifest=json.loads((directory/'manifest.json').read_text())
        template_path=directory/'sections.json'
        assert checksum(template_path)==manifest['artifact_sha256']['sections.json']
        template=ModelSections.model_validate_json(template_path.read_text())
        evaluations=[]
        for candidate in record['analysis']['benchmark_summary']:
            name=candidate['model_name'];path=directory/f'model-{name}.joblib'
            if not path.exists():
                evaluations.append({'model':name,'status':'unavailable','reason':candidate['failure_reason']});continue
            assert checksum(path)==manifest['artifact_sha256'][path.name]
            sections=project_sections(joblib.load(path),template,name)
            evaluated=EVALUATE(sections,scenario)
            evaluated.update(status='completed',model_sha256=checksum(path),quality_status=candidate['reliability']['status'])
            evaluations.append(evaluated)
        results.append({'scenario':scenario,'selected_model':record['analysis']['selected_model'],
                        'sections_sha256':checksum(template_path),'models':evaluations})
    result={'method':'all_saved_models_same_conditional_grids_v1','diagnostic_only':True,'used_for_selection':False,
            'limitation':'T0 function recovery; not causal attribution or a new test score. Correlation can make grids off-support.',
            'source_sha256':{str(path.relative_to(ROOT)):checksum(path) for path in [Path(__file__),ROOT/'scripts/evaluate_recovery.py']},
            'scenarios':results}
    output=ROOT/'data/reports/recovery-all-v2.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    matrix=[]
    for model in NAMES:
        row=[NAMES[model]]
        for scenario,factor in [('threshold','ltv'),('saturation','volume'),('u_shaped','ltv'),('interaction',None),('mixed','ltv')]:
            evaluated=next(m for item in results if item['scenario']==scenario for m in item['models'] if m['model']==model)
            if factor:
                measured=next(c for c in evaluated['curves'] if c['factor']==factor)
            else:
                measured=next(p for p in evaluated['pairs'] if set(p['factors'])=={'pd','ltv'})
            row.append(f"{measured['rmse']:.6g}")
        matrix.append(row)
    lines=['# Восстановление формы для всех семейств','', 'Статус: Реализовано','',
           'Каждая сохранённая модель проверена на одной сетке 1–99% обучающих значений при одинаковой опорной точке T0. Истинная формула используется только после обучения. Выбор модели и тестовые оценки не изменяются.',
           '', '| Модель | Порог: LTV | Насыщение: volume | U-форма: LTV | Взаимодействие: PD × LTV | Смешанный: LTV |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += ['| '+' | '.join(row)+' |' for row in matrix]
    lines += ['', 'Все значения — RMSE сечения или парного контраста в единицах синтетической цели. Сравнение моделей корректно внутри столбца. Полные результаты: 10 сценариев × 7 моделей, по 5 одномерных сечений и 10 пар.',
              '', 'Результаты численно отклонённых атрибуций также имеют прогнозные сечения, но это не отменяет отказ. При корреляции сетка может содержать редко наблюдаемые сочетания. Данная проверка не определяет истинные SHAP-вклады и не доказывает причинность.',
              '', '[Машинный отчёт](../../data/reports/recovery-all-v2.json) · [Повторение расчёта](evaluate_recovery.py) · [Методика](README.md)','']
    (ROOT/'docs/ml/recovery-v2.md').write_text('\n'.join(lines))
    print('Evaluated',sum(len(item['models']) for item in results),'saved models')


if __name__=='__main__':main()
