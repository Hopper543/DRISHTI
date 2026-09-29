"""Copied from the DRISHTI dataset package (scripts/generate_synthetic.py); generator body unchanged.

    python scripts/generate_synthetic.py --check        # regenerate in memory, compare with data/demo
    python scripts/generate_synthetic.py --out <dir>    # write CSV/Parquet in the package layout

Stylized SYNTHETIC fixtures. Parameters below are design assumptions, not measured burn-in physics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]  # overridden by --out
SEED=26170
TIMES=[0,24,48,96,168]
# Name, unit, nominal, relative part spread, typical healthy endpoint drift, lower, upper.
FAMILIES={
 'DEMO_MEMORY':[('leakage_current','uA',10,.09,2,0,50),('supply_current','mA',4,.035,.12,0,6)],
 'DEMO_OPAMP':[('input_bias_current','nA',.6,.10,.1,0,2),('input_offset_voltage','uV',20,.15,5,-125,125),('quiescent_current','mA',.9,.03,.04,0,1.3)],
 'DEMO_MOSFET':[('RDS_on','ohm',.32,.04,.025,0,.5),('threshold_voltage','V',1.8,.03,.07,1,2.5)]}
def main(ROOT=ROOT):
 rng=np.random.default_rng(SEED);obs=[];truth=[];lots=[];pairs=[];targets=[]
 families=list(FAMILIES)
 for li in range(204):
  fam=families[li%3];ordinal=li//3
  split=('train' if ordinal<30 else 'validation' if ordinal<35 else 'calibration' if ordinal<50 else 'test') if li<180 else 'ood_test' if li<192 else 'edge_test'
  lot=f'SYN_L{li:03d}';n=30 if li<180 else 24 if li<192 else [3,5,8,12][(li-192)//3]
  lot_shift=rng.normal(0,.045);lot_bad=split=='ood_test' and ordinal%2==0
  lots.append(dict(lot_id=lot,part_number=fam,split=split,n_devices=n,provenance_class='SYNTHETIC',lot_id_kind='generated',
                   stress_temperature_C=125,measurement_temperature_C=25))
  for di in range(n):
   dev=f'{lot}_D{di:03d}'
   # Chosen independently of IDs; untouched files keep all rows in one lot partition.
   scenario=rng.choice(['healthy','gradual','delayed','step','stable_outlier','measurement_fault'],p=[.75,.075,.045,.035,.045,.05])
   affected=int(rng.integers(len(FAMILIES[fam])));latent=scenario in ('gradual','delayed','step')
   for pi,(param,unit,nom,spread,drift,lower,upper) in enumerate(FAMILIES[fam]):
    key=f'{dev}:{param}';is_affected=(pi==affected)
    v0=nom*(1+lot_shift+rng.normal(0,spread));slope_amp=drift*rng.lognormal(0,.3)
    sign=1 if param!='input_offset_voltage' else rng.choice([-1,1])
    severity=rng.uniform(2.0,6.0);onset=rng.uniform(30,130)
    if scenario=='stable_outlier' and is_affected:v0 += .5*(upper-nom)
    if lot_bad:v0 += .48*(upper-nom)
    physical=[];measured=[]
    for ti,t in enumerate(TIMES):
     a=t/168;v=v0+sign*slope_amp*a
     if lot_bad:v+=.75*(upper-nom)*a**1.4
     if is_affected and scenario=='gradual':v+=severity*drift*(.35*a+.65*a*a)
     if is_affected and scenario=='delayed':v+=severity*drift*(max(t-onset,0)/(168-onset))**1.5
     if is_affected and scenario=='step':v+=severity*drift*(t>=onset)
     physical.append(v)
     noise=max(abs(nom)*.003,1e-10)*rng.normal();m=v+noise;flag=''
     if scenario=='measurement_fault' and is_affected and t==24:m+=rng.uniform(.15,.45)*(upper-nom);flag=''
     if split=='edge_test' and di==0 and t==24:m=np.nan;flag='missing_measurement'
     if split=='edge_test' and di==1:m=nom;flag='quantized_constant_series'
     measured.append(m)
     obs.append(dict(record_id=f'{key}:h{t}',provenance_class='SYNTHETIC',device_id=dev,lot_id=lot,lot_id_kind='generated',
       part_number=fam,parameter=param,unit=unit,burn_in_hours=t,value=m,stress_type='synthetic_thermal_burn_in',
       stress_temperature_C=125,measurement_temperature_C=25,replicate=1,quality_flags=flag))
    # Distinguish physical endpoint truth, measured endpoint target, and planted scenario.
    truth.append(dict(sample_id=key,device_id=dev,lot_id=lot,parameter=param,scenario=scenario if is_affected else 'healthy_parameter',
        latent_defect_planted=bool(latent and is_affected),lot_wide_degradation_planted=lot_bad,
        measurement_fault_planted=bool(scenario=='measurement_fault' and is_affected),true_value_168h=physical[-1],
        true_out_of_spec_168h=bool(physical[-1]<lower or physical[-1]>upper),label_provenance='SYNTHETIC_GENERATOR_TRUTH'))
    pairs.append(dict(sample_id=key,device_id=dev,lot_id=lot,part_number=fam,parameter=param,unit=unit,
       value_0h=measured[0],value_24h=measured[1],spec_lower=lower,spec_upper=upper,
       spec_origin='DEMONSTRATION_ASSUMPTION',provenance_class='SYNTHETIC',split=split))
    targets.append(dict(sample_id=key,value_168h=measured[-1],target_provenance='SYNTHETIC_MEASURED_ENDPOINT'))
 for name,records in [('synthetic_measurements',obs),('synthetic_early_inputs',pairs),('synthetic_lots',lots)]:
  f=pd.DataFrame(records);f.to_csv(ROOT/f'data/{name}.csv',index=False);f.to_parquet(ROOT/f'data/{name}.parquet',index=False)
 pd.DataFrame(truth).to_csv(ROOT/'evaluation/synthetic_ground_truth.csv',index=False)
 pd.DataFrame(targets).to_csv(ROOT/'evaluation/synthetic_targets_168h.csv',index=False)
 (ROOT/'docs/generator_assumptions.json').write_text(json.dumps(dict(seed=SEED,times_hours=TIMES,families=FAMILIES,
  temperature_model='NONE: fixed stress context only; no Arrhenius law fitted',
  provenance='SYNTHETIC statistical fixtures; not calibrated from radiation dose to thermal time',
  onset_note='Delayed/step defects can have identical early observations to healthy devices; near-zero escape is not expected',
  mechanism_labels='Scenario names describe mathematical shapes, not TDDB/NBTI/electromigration diagnoses'),indent=2))
 print('Synthetic devices',len({r['device_id'] for r in pairs}),'lots',len(lots),'measurements',len(obs),'forecast cases',len(pairs))
def check():
    """Regenerate into a temp dir and compare with the committed data/demo fixture."""
    import sys, tempfile
    repo = Path(__file__).resolve().parents[1]
    (repo / "runtime").mkdir(exist_ok=True)  # git-ignored scratch on the repo drive
    with tempfile.TemporaryDirectory(dir=repo / "runtime") as tmp:
        t = Path(tmp); (t / "data").mkdir(); (t / "evaluation").mkdir(); (t / "docs").mkdir()
        main(t)
        pairs = [("data/synthetic_early_inputs.parquet", "synthetic_early_inputs"),
                 ("data/synthetic_measurements.parquet", "synthetic_measurements"),
                 ("data/synthetic_lots.parquet", "synthetic_lots"),
                 ("evaluation/synthetic_targets_168h.csv", "synthetic_targets_168h"),
                 ("evaluation/synthetic_ground_truth.csv", "synthetic_ground_truth")]
        ok = True
        for new, old in pairs:
            a = pd.read_parquet(t / new) if new.endswith(".parquet") else pd.read_csv(t / new)
            b = pd.read_parquet(repo / "data" / "demo" / f"{old}.parquet")
            # Empty strings and missing values are the same thing after a CSV round trip.
            a, b = a.replace("", pd.NA), b.replace("", pd.NA)
            try:
                pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                              check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
                print(f"[same] {old}")
            except AssertionError as exc:
                ok = False
                print(f"[DIFFERENT] {old}: {str(exc).splitlines()[0]}")
        sys.exit(0 if ok else 1)


if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--out',type=Path)
 a=ap.parse_args()
 if a.check:check()
 elif a.out:
  for d in ('data','evaluation','docs'):(a.out/d).mkdir(parents=True,exist_ok=True)
  main(a.out)
 else:ap.error('use --check or --out <dir>')

