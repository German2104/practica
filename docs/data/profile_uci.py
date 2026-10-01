"""Reproduce UCI window profiles and the fixed one-factor sensitivity grid."""
from pathlib import Path
import argparse
import dataclasses
import datetime
import json
from ec_analysis.data.uci_credit import UciCreditReader, UciPortfolioPreparation, ReceivableValuation
from ec_analysis.data.acquisition import file_checksum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('data/raw/uci-credit/default-credit-clients.zip'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new report path')
    records=UciCreditReader().read_archive(args.source)
    windows=[]
    for first,second in [(4,5),(5,6),(6,7),(7,8),(8,9)]:
        frame,metadata=UciPortfolioPreparation().prepare_periods(records,first,second)
        metadata['distributions']=json.loads(frame[['volume','pd','model_npv']].describe(percentiles=[.01,.5,.95,.99]).to_json())
        windows.append(metadata)
    sensitivity=[]
    cases=[('baseline',ReceivableValuation()),('coupon_12pct',ReceivableValuation(annual_coupon=.12)),
           ('discount_12pct',ReceivableValuation(annual_discount=.12)),('lgd_40pct',ReceivableValuation(loss_given_default=.4)),
           ('lgd_80pct',ReceivableValuation(loss_given_default=.8)),('pd_half',ReceivableValuation(probability_multiplier=.5)),
           ('pd_double',ReceivableValuation(probability_multiplier=2))]
    for label,valuation in cases:
        frame,_=UciPortfolioPreparation(valuation=valuation).prepare_periods(records,8,9)
        totals=frame.groupby('period').model_npv.sum().to_dict()
        sensitivity.append(dict(scenario=label,assumptions=dataclasses.asdict(valuation),initial_value=totals['T0'],
                                final_value=totals['T1'],total_change=totals['T1']-totals['T0']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),
                           source_sha256=file_checksum(args.source),windows=windows,sensitivity=sensitivity),
                           ensure_ascii=False,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
