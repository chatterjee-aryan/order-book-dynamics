"""Offline checks for expiry snapshots and preservation of existing CSVs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from workspace.data_processing import set_futures_expiry as module


class ExpiryTests(unittest.TestCase):
    def fixture(self, folder):
        path = Path(folder) / "ES_2026-12-17_2026-12-21.csv"
        columns = pd.MultiIndex.from_product([["ES_T0"], module.FIELDS], names=["ticker", "field"])
        frame = pd.DataFrame([[1., 2., .5, 1.5, 10.], [1., 2., .5, 1.5, 10.], [None]*5],
                             index=pd.Index(["2026-12-17", "2026-12-18", "2026-12-21"], name="date"),
                             columns=columns)
        frame.to_csv(path)
        symbols_path = path.with_name(f"{path.stem}_symbols.csv")
        symbols = pd.DataFrame([{
            "date": day, "tenor": "ES_T0", "canonical_symbol": "ESZ26", "raw_symbol": "ESZ6",
            "contract_year": 2026, "contract_month": 12,
        } for day in frame.index])
        symbols.to_csv(symbols_path, index=False)
        return path, symbols_path, frame

    def client(self):
        client = MagicMock()
        rows = []
        for year, instrument, expiry in [(2016, 1, "2016-12-16T14:30Z"), (2026, 2, "2026-12-18T14:30Z")]:
            rows.append({"ts_recv": pd.Timestamp("2026-12-17T00:00Z"), "instrument_id": instrument,
                         "instrument_class": "F", "raw_symbol": "ESZ6", "maturity_year": year,
                         "maturity_month": 12, "expiration": pd.Timestamp(expiry)})
        client.timeseries.get_range.return_value.to_df.return_value = pd.DataFrame(rows).set_index("ts_recv")
        return client

    def test_snapshot_checks_full_year_and_deduplicates_symbols(self):
        contracts = pd.DataFrame([{"raw_symbol": "ESZ6", "contract_year": 2026, "contract_month": 12}]*2)
        client = self.client()
        result = module.fetch_expirations(client, pd.Timestamp("2026-12-17"), contracts)
        self.assertEqual(result, {("ESZ6", 2026, 12): pd.Timestamp("2026-12-18T14:30Z")})
        call = client.timeseries.get_range.call_args.kwargs
        self.assertEqual(call["symbols"], ["ESZ6"])
        self.assertEqual(call["schema"], "definition")
        self.assertEqual(call["end"] - call["start"], pd.Timedelta(days=1))

    def test_update_preserves_ohlcv_and_nulls_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            path, symbols_path, frame = self.fixture(folder)
            original_symbols = symbols_path.read_bytes()
            client = self.client()
            module.process_pair(client, path, symbols_path, {})
            saved = module.read_ohlcv(path)
            pd.testing.assert_frame_equal(saved[frame.columns], frame, check_exact=True)
            flags = saved[("ES_T0", "expired")].astype("boolean")
            self.assertEqual(flags.iloc[0], False)
            self.assertEqual(flags.iloc[1], True)
            self.assertTrue(pd.isna(flags.iloc[2]))
            self.assertEqual(symbols_path.read_bytes(), original_symbols)
            module.process_pair(client, path, symbols_path, {})
            pd.testing.assert_frame_equal(module.read_ohlcv(path), saved)
            self.assertEqual(len(list(Path(folder).iterdir())), 2)

    def test_failure_does_not_replace_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path, symbols_path, _ = self.fixture(folder)
            before = path.read_bytes()
            client = self.client()
            client.timeseries.get_range.side_effect = RuntimeError("API failure")
            with self.assertRaises(RuntimeError):
                module.process_pair(client, path, symbols_path, {})
            self.assertEqual(path.read_bytes(), before)

    def test_unknown_expiry_does_not_become_false(self):
        with tempfile.TemporaryDirectory() as folder:
            path, symbols_path, _ = self.fixture(folder)
            before = path.read_bytes()
            client = self.client()
            client.timeseries.get_range.return_value.to_df.return_value = pd.DataFrame()
            with self.assertRaisesRegex(ValueError, "No valid expiration"):
                module.process_pair(client, path, symbols_path, {})
            self.assertEqual(path.read_bytes(), before)

    def test_expiry_day_after_day_and_missing_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            path, symbols_path, frame = self.fixture(folder)
            frame.iloc[-1] = [1., 2., .5, 1.5, 10.]
            symbols = pd.read_csv(symbols_path)
            observed = module.observed_symbols(frame, symbols)
            updated = module.add_expired_flags(frame, observed,
                {("ESZ6", 2026, 12): pd.Timestamp("2026-12-18T14:30Z")}, 2026)
            self.assertEqual(updated[("ES_T0", "expired")].tolist(), [False, True, True])
            with self.assertRaisesRegex(ValueError, "missing contract identifiers"):
                module.observed_symbols(frame, symbols.iloc[:1])

    def test_round_trip_failure_leaves_original_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path, symbols_path, frame = self.fixture(folder)
            before = path.read_bytes()
            updated = frame.copy()
            updated[("ES_T0", "expired")] = pd.Series([False, True, pd.NA], index=frame.index, dtype="boolean")
            updated.iloc[0, 0] = 99
            with self.assertRaises(AssertionError):
                module.write_verified(path, frame, updated)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
