"""Regression tests for scores that must not depend on other test predictions."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/scripts_rev'))
from entropy_scale import EntropyRule,load_scores,RAW

class EntropyScaleTests(unittest.TestCase):
    def test_missing_imputation_is_fit_only_and_ties_are_inclusive(self):
        rule=EntropyRule.from_development([1,np.nan,3],[1,np.nan,3])
        self.assertEqual(rule.imputation_median,2)
        self.assertEqual(rule.threshold,2)
        np.testing.assert_array_equal(rule.score([np.nan,2,3]),[2,2,3])
        np.testing.assert_array_equal(rule.accept([np.nan,2,3]),[True,True,False])

    def test_raw_entropy_is_not_clipped_as_a_probability(self):
        rule=EntropyRule(1.4,1.34)
        np.testing.assert_array_equal(rule.score([1.34,20]),[1.34,20])
        np.testing.assert_array_equal(rule.accept([1.34,20]),[True,False])

    def test_infinite_values_are_rejected(self):
        with self.assertRaises(ValueError):EntropyRule(1.,1.).score([np.inf])
        with self.assertRaises(ValueError):EntropyRule.from_development([np.nan],[1.])

    def test_legacy_layer_cannot_silently_satisfy_the_corrected_reader(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):load_scores(Path(d),'model')

    def test_actual_scores_and_acceptance_are_batch_permutation_and_chunk_invariant(self):
        frame=pd.read_parquet(ROOT/'data_compact/e3_web/per_row_web.parquet')
        for model,dm in frame.groupby('model'):
            rule=EntropyRule.from_development(dm[dm.split=='fit'].coord_mean_entropy,
                                              dm[dm.split=='calib'].coord_mean_entropy)
            for split in ['calib','test_website','test_domain']:
                with self.subTest(model=model,split=split):
                    values=dm[dm.split==split].coord_mean_entropy.to_numpy()
                    score=rule.score(values);accept=rule.accept(values)
                    extended=np.r_[values,0.,1.e12,np.nan]
                    np.testing.assert_array_equal(rule.score(extended)[:len(values)],score)
                    np.testing.assert_array_equal(rule.accept(extended)[:len(values)],accept)
                    p=np.random.default_rng(73).permutation(len(values));back=np.argsort(p)
                    np.testing.assert_array_equal(rule.score(values[p])[back],score)
                    np.testing.assert_array_equal(rule.accept(values[p])[back],accept)
                    for chunks in [1,7,23]:
                        parts=np.array_split(values,chunks)
                        np.testing.assert_array_equal(np.concatenate([rule.score(c) for c in parts]),score)
                        np.testing.assert_array_equal(np.concatenate([rule.accept(c) for c in parts]),accept)

    def test_verified_frozen_operating_point_counts(self):
        for model,n,nfail in [('qwen3vl4b',375,45),('internvl35_4b',314,109)]:
            d=load_scores(ROOT,model).query("split=='test_website'")
            self.assertEqual(int(d.accepted_frozen_raw_entropy.sum()),n)
            self.assertEqual(int(d.loc[d.accepted_frozen_raw_entropy,'click_fail'].sum()),nfail)

if __name__=='__main__':unittest.main(verbosity=2)
