#!/usr/bin/env python3
"""Full pinned CPU replay: corrected raw score first, followed by L1/L2 and descendants."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/scripts_rev'))
from entropy_scale import correction_root
from validate_correction import equal
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path)
    a=p.parse_args();out=(a.out_dir or Path(tempfile.mkdtemp(prefix='vlmgf-corrected-replay-'))).resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Write replay outputs outside the artifact')
    out.mkdir(parents=True,exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    pins={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name,version=line.split('==');assert importlib.metadata.version(name)==version,(name,version)
            pins[name]=version
    assert sys.version_info[:2]==(3,11)
    count=0
    for line in (ROOT/'checksums.sha256').read_text().splitlines():
        h,n=line.split(maxsplit=1)
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n;count+=1
    print('Pinned dependencies and',count,'artifact checksums PASS',flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    runs=[]
    def run(label,script,*args):
        command=[sys.executable,'-B','-I',str(ROOT/script),*map(str,args)]
        print('RUN',label,flush=True)
        with (logs/(label+'.txt')).open('w') as fh:subprocess.run(command,cwd=ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT,check=True)
        runs.append(label);print('PASS',label,flush=True)
    c=correction_root(ROOT)
    run('raw_score_reconstruction','src/scripts_rev/correct_entropy_scale.py','--artifact-root',ROOT,'--out-dir',out/'raw')
    for name in ['corrected_raw_entropy_operating_points.csv','corrected_raw_entropy_scores.csv']:
        equal(pd.read_csv(out/'raw'/name),pd.read_csv(c/name))
    for model in ['qwen3vl4b','internvl35_4b']:
        equal(pd.read_parquet(out/'raw'/f'per_row_scores_{model}.parquet'),pd.read_parquet(c/f'per_row_scores_{model}.parquet'))
    run('invariance_regressions','repro/test_entropy_scale.py')
    run('full_L1L2','repro/reproduce_core.py')
    assert '170 checks, 0 FAIL' in (logs/'full_L1L2.txt').read_text()
    run('v3_analysis','src/scripts_rev/v3_analysis.py',ROOT,out/'v3')
    run('v3_semantic_checks','src/scripts_rev/v3_semantic_checks.py',ROOT,'--replay',out/'v3')
    assert '135/135' in (logs/'v3_semantic_checks.txt').read_text()
    run('p0_reanalysis','src/scripts_rev/p0_web_reanalysis_2026_10_20.py',ROOT,out/'p0')
    run('same_accepted_sets_and_bands','src/scripts_rev/p0_accepted_set_and_bands_2026_10_20.py',ROOT,out/'accepted')
    for folder in ['p0','accepted']:
        for path in (out/folder).glob('*.csv'):equal(pd.read_csv(path),pd.read_csv(c/'p0'/path.name))
    run('aggregate_points_and_intervals','src/scripts_rev/run_aggregate_audit.py',ROOT,out/'aggregates')
    for path in (out/'aggregates').glob('*.csv'):equal(pd.read_csv(path),pd.read_csv(c/'aggregate_audit'/path.name))
    run('changed_unchanged_claims','src/scripts_rev/validate_correction.py',ROOT,'--out-dir',out/'claim_checks')
    assert '24 changed/unchanged checks PASS' in (logs/'changed_unchanged_claims.txt').read_text()
    run('figure_table_regeneration','src/scripts_rev/v3_figures.py',ROOT,out/'generated')
    generated_checks=0
    for folder,pattern in [('figures','*.pdf'),('tables','*.tex')]:
        for path in (out/'generated'/folder).glob(pattern):
            assert path.read_bytes()==(ROOT/('figures_revised' if folder=='figures' else 'tables_revised')/path.name).read_bytes(),path.name
            generated_checks+=1
    run('existing_repair_audit','repro/check_repair_and_audit.py')
    run('r9_score_tests','r9_1/tests/test_r9_score.py')
    run('r9_tie_tests','r9_1/tests/test_r9_threshold_ties.py')
    run('quickcheck','repro/cpu_quickcheck.py')
    run('current_number_claims','repro/audit_number_check.py')
    report={'status':'PASS','scope':'Full L1/L2 replay plus versioned raw-entropy correction, no inference or new annotation',
       'python':sys.version,'pinned_packages':pins,'checksums':count,'core_L1L2_checks':170,
       'semantic_checks':135,'invariance_unittests':6,'rank_invariance_checks':25,
       'changed_unchanged_checks':24,'excluded_screenshot_asset_checks':1,'regenerated_pdf_and_table_byte_checks':generated_checks,
       'runs':runs,'legacy_evidence':'Existing natural-image, fitted-head, queue and sampling-rank results replayed. Superseded split-normalized raw operating points are not treated as current results.',
       'replay_evidence':'Raw scores reconstructed here with fit-only imputation and calibration-median acceptance; no independent reference run is claimed.',
       'external_actions':0,'new_GPU_inference':0,'new_human_annotation':0}
    (out/'REPLAY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('Replay output:',out)

if __name__=='__main__':main()
