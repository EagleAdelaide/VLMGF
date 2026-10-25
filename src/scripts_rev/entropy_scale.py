"""Versioned raw-entropy failure scores: fit-only imputation, calibration-only rule."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='entropy_scale_v1'
RAW='raw_entropy_failure_score'

@dataclass(frozen=True)
class EntropyRule:
    imputation_median: float
    threshold: float

    @classmethod
    def from_development(cls, fit, calibration):
        fit=np.asarray(fit,dtype=float)
        if np.isinf(fit).any() or not np.isfinite(fit).any():
            raise ValueError('Fitting entropy must contain finite values and no infinity')
        med=float(np.nanmedian(fit))
        provisional=cls(med,0.)
        cal=provisional.score(calibration)
        if not len(cal):raise ValueError('Empty calibration population')
        return cls(med,float(np.median(cal)))

    def score(self, values):
        values=np.asarray(values,dtype=float).copy()
        if np.isinf(values).any():raise ValueError('Infinite entropy requires separate audit')
        values[np.isnan(values)]=self.imputation_median
        return values

    def accept(self, values):
        return self.score(values)<=self.threshold

def correction_root(root):
    return Path(root)/'data_compact/corrections'/VERSION

def load_scores(root,model):
    path=correction_root(root)/f'per_row_scores_{model}.parquet'
    if not path.exists():raise RuntimeError('Run repro/correct_entropy_scale.py before corrected analyses')
    frame=pd.read_parquet(path)
    if RAW not in frame:raise ValueError('Missing corrected raw-entropy failure score')
    return frame

def failure_score(frame,head):
    head=head.removeprefix('prob_success_')
    if head in ['raw_coord_mean_entropy','raw_coord_entropy',RAW]:
        return frame[RAW]
    return 1-frame['prob_success_'+head]
