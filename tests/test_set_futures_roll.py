"""Offline checks for primary selection and safe in-place writes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from workspace.data_processing import set_futures_roll as module


class PrimaryTests(unittest.TestCase):
    def fixture(self, volumes, start="2026-12-28"):
        dates = pd.bdate_range(start, periods=len(volumes))
        columns = pd.MultiIndex.from_product([["ES_T0", "ES_T1"], module.FIELDS], names=["ticker", "field"])
        rows, symbols = [], []
        for day, pair in zip(dates, volumes):
            row = []
            for slot, volume in enumerate(pair):
                row.extend([None]*5 if volume is None else [1., 2., .5, 1.5, volume])
                symbols.append({"date": day, "tenor": f"ES_T{slot}",
                                "canonical_symbol": ["ESZ26", "ESH27"][slot],
                                "raw_symbol": ["ESZ6", "ESH7"][slot],
                                "contract_year": [2026, 2027][slot],
                                "contract_month": [12, 3][slot]})
            rows.append(row)
        frame = pd.DataFrame(rows, index=dates.strftime("%Y-%m-%d"), columns=columns)
        frame.index.name = "date"
        return frame, pd.DataFrame(symbols)

    def choices(self, frame, symbols):
        result = module.set_primary(frame, symbols, "ES")
        return list(zip(result[("ES_T0", "is_primary")], result[("ES_T1", "is_primary")]))

    def test_three_days_at_exactly_65_percent(self):
        frame, symbols = self.fixture([(50,50), (33,67), (30,70), (35,65), (25,75)])
        self.assertEqual(self.choices(frame, symbols), [(True,False)]*3+[(False,True)]*2)

    def test_switch_back_requires_three_new_days(self):
        frame, symbols = self.fixture([(35,65)]*3+[(65,35)]*3)
        self.assertEqual(self.choices(frame, symbols),
                         [(True,False)]*2+[(False,True)]*3+[(True,False)])

    def test_lower_share_zero_total_and_missing_volume_reset_streak(self):
        for breaker in [(36,64), (0,0), (float("nan"),65)]:
            with self.subTest(breaker=breaker):
                frame, symbols = self.fixture([(35,65)]*2+[breaker]+[(35,65)]*3)
                self.assertEqual(self.choices(frame, symbols), [(True,False)]*5+[(False,True)])

    def test_missing_bar_switches_immediately_and_shared_gaps_are_ignored(self):
        frame, symbols = self.fixture([(35,65), (None,None), (35,65), (35,65), (70,None), (None,20)])
        self.assertEqual(self.choices(frame, symbols),
                         [(True,False), (False,False), (True,False), (False,True), (True,False), (False,True)])

    def test_actual_primary_survives_tenor_change_and_pair_resets_streak(self):
        frame, symbols = self.fixture([(35,65)]*3+[(35,65)]*3)
        # Former T1 moves to T0, with a new T1 entering after the third date.
        mask = symbols.date >= pd.Timestamp(frame.index[3])
        for slot, canonical, raw, month in [(0,"ESH27","ESH7",3),(1,"ESM27","ESM7",6)]:
            rows = mask & symbols.tenor.eq(f"ES_T{slot}")
            symbols.loc[rows, ["canonical_symbol","raw_symbol","contract_year","contract_month"]] = [canonical,raw,2027,month]
        self.assertEqual(self.choices(frame, symbols),
                         [(True,False)]*2+[(False,True)]+[(True,False)]*2+[(False,True)])

    def test_year_boundary_does_not_reset_streak(self):
        frame, symbols = self.fixture([(35,65)]*3, start="2026-12-30")
        self.assertEqual(self.choices(frame, symbols), [(True,False)]*2+[(False,True)])

    def test_preservation_rerun_and_fed_funds_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            frame, symbols = self.fixture([(35,65)]*3)
            frame[("ES_T0", "expired")] = [False, True, True]
            path = Path(folder) / "ES_2026-12-28_2026-12-30.csv"
            symbols_path = path.with_name(f"{path.stem}_symbols.csv")
            frame.to_csv(path)
            symbols.to_csv(symbols_path, index=False)
            original_symbols = symbols_path.read_bytes()
            zq = Path(folder) / "ZQ_2026-12-28_2026-12-30.csv"
            zq.write_text("must remain untouched")
            with patch.object(module, "DATA_DIR", Path(folder)):
                module.main()
            saved = module.read_ohlcv(path)
            pd.testing.assert_frame_equal(saved[frame.columns], frame, check_exact=True)
            module.process_pair(path, symbols_path, "ES")
            pd.testing.assert_frame_equal(module.read_ohlcv(path), saved)
            self.assertEqual(symbols_path.read_bytes(), original_symbols)
            self.assertEqual(zq.read_text(), "must remain untouched")
            self.assertEqual(len(list(Path(folder).iterdir())), 3)

    def test_validation_failure_does_not_replace_file(self):
        with tempfile.TemporaryDirectory() as folder:
            frame, symbols = self.fixture([(35,65)]*3)
            path = Path(folder) / "ES_test.csv"
            symbols_path = Path(folder) / "symbols.csv"
            frame.to_csv(path)
            symbols.iloc[:1].to_csv(symbols_path, index=False)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Missing symbol mapping"):
                module.process_pair(path, symbols_path, "ES")
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
