import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_summary import make_summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='plankton-summary-')
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name)
        (self.results / 'tables').mkdir()
        for name in ['detection_four_groups.csv', 'segmentation_summary.csv']:
            shutil.copy2(ROOT / 'results/tables' / name, self.results / 'tables' / name)

    def test_saved_summaries_reproduce_six_experiments(self):
        actual = make_summary(self.results)
        expected = pd.read_csv(ROOT / 'results/tables/experiment_comparison.csv')
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(len(actual), 6)

    def test_changed_inputs_refresh_output(self):
        make_summary(self.results)
        dp = self.results / 'tables/detection_four_groups.csv'
        sp = self.results / 'tables/segmentation_summary.csv'
        detection = pd.read_csv(dp)
        segmentation = pd.read_csv(sp)
        detection.loc[0, 'median_ms'] += 10
        index = segmentation.index[segmentation.prompt_variant == 'base'][0]
        segmentation.loc[index, 'median_ms'] += 20
        detection.to_csv(dp, index=False)
        segmentation.to_csv(sp, index=False)
        make_summary(self.results)
        actual = pd.read_csv(self.results / 'tables/experiment_comparison.csv')
        d = detection.iloc[0]
        s = segmentation.loc[index]
        self.assertAlmostEqual(actual.loc[(actual.method == d.model) & (actual.condition == d.group), 'median_ms'].iloc[0], d.median_ms)
        self.assertAlmostEqual(actual.loc[(actual.method == s.method) & (actual.task == 'segmentation'), 'median_ms'].iloc[0], s.median_ms)


if __name__ == '__main__':
    unittest.main()
