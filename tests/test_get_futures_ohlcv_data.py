"""Offline checks for explicit-symbol downloads, caching, and failure isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from workspace.data_processing import get_futures_ohlcv_data as module


class DownloadHandlingTests(unittest.TestCase):
    def setUp(self):
        self.days = module.candidate_dates('2026-10-01', '2026-10-06')
        self.mapping = module.make_symbol_map('ES', self.days)

    def bars(self, days):
        return pd.DataFrame([
            {'date': day, 'raw_symbol': 'ESZ6', 'instrument_id': 1,
             **dict(zip(module.FIELDS, [1., 2., .5, 1.5, 10]))}
            for day in days
        ])

    def test_calendar_symbols_and_rollover(self):
        days = module.candidate_dates('2026-12-31', '2027-01-01')
        mapping = module.make_symbol_map('ZQ', days).set_index(['date', 'tenor'])
        self.assertEqual(mapping.loc[(days[0], 'ZQ_T0'), 'canonical_symbol'], 'ZQZ26')
        self.assertEqual(mapping.loc[(days[0], 'ZQ_T1'), 'raw_symbol'], 'ZQF7')
        self.assertEqual(mapping.loc[(days[1], 'ZQ_T6'), 'canonical_symbol'], 'ZQN27')
        self.assertEqual(set(self.mapping.raw_symbol), {'ESZ6', 'ESH7'})
        # No automatic intra-month expiry roll, even after ES December expiration.
        late = module.make_symbol_map('ES', days[:1])
        self.assertEqual(late.iloc[0].canonical_symbol, 'ESZ26')

    def test_btc_before_launch_stays_empty(self):
        days = module.candidate_dates('2017-11-01', '2017-12-01')
        mapping = module.make_symbol_map('BTC', days)
        self.assertEqual(set(mapping.date), {pd.Timestamp('2017-12-01')})

    def test_preserves_dates_before_history_and_internal_gaps(self):
        frame = module.make_ohlcv_frame('ES', self.days, self.mapping, self.bars(self.days[-1:]))
        self.assertTrue(frame.index.equals(self.days))
        self.assertTrue(frame.iloc[:-1].isna().all().all())
        self.assertEqual(frame.iloc[-1][('ES_T0', 'close')], 1.5)
        self.assertTrue(frame['ES_T1'].isna().all().all())

    def test_empty_data_has_full_index(self):
        frame = module.make_ohlcv_frame('ES', self.days,
            pd.DataFrame(columns=module.MAP_COLUMNS), pd.DataFrame())
        self.assertEqual(frame.shape, (len(self.days), 10))
        self.assertTrue(frame.isna().all().all())

    def test_configurable_blocks_cover_every_date_once(self):
        days = module.candidate_dates('2010-06-07', '2026-10-21')
        for years, count in [(1, 17), (5, 4)]:
            with self.subTest(years=years), patch.object(module, 'BLOCK_YEARS', years):
                blocks = module.date_blocks(days)
                self.assertEqual(len(blocks), count)
                combined = blocks[0].append(blocks[1:])
                self.assertTrue(combined.equals(days))
                self.assertTrue(combined.is_unique)
                for block in blocks:
                    self.assertLess(block[-1], block[0] + pd.DateOffset(years=years))

    def test_schema_coverage_excludes_incomplete_daily_bars(self):
        coverage = {'start': '2010-01-01T00:00Z', 'end': '2026-10-10T00:00Z',
                    'schema': {module.SCHEMA: {
                        'start': '2026-10-01T10:00Z', 'end': '2026-10-06T12:00Z'}}}
        with self.assertLogs(module.LOG):
            start, end = module.available_bounds(self.days, module.SCHEMA, coverage)
        self.assertEqual(start.tz_convert('UTC'), pd.Timestamp('2026-10-02', tz='UTC'))
        self.assertEqual(end.tz_convert('UTC'), pd.Timestamp('2026-10-06', tz='UTC'))
        self.assertEqual(str(start.tz), module.TIMEZONE)

    def test_no_coverage_does_not_call_timeseries(self):
        client = MagicMock()
        coverage = {'start': '2027-01-01T00:00Z', 'end': '2028-01-01T00:00Z'}
        with self.assertLogs(module.LOG):
            self.assertTrue(module.download_bars(client, 'ES', self.days, self.mapping, coverage).empty)
        client.timeseries.get_range.assert_not_called()

    def test_download_uses_only_ohlcv_deduplicates_and_reuses_cache(self):
        client = MagicMock()
        raw = self.bars(self.days).rename(columns={'date': 'ts_event', 'raw_symbol': 'symbol'})
        raw['ts_event'] = pd.to_datetime(raw['ts_event'], utc=True)
        client.timeseries.get_range.return_value.to_df.return_value = raw.set_index('ts_event')
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'OUTPUT_DIR', Path(tmp)), \
             patch.object(module, 'USE_CACHE', True):
            first = module.download_bars(client, 'ES', self.days, self.mapping)
            second = module.download_bars(client, 'ES', self.days, self.mapping)
            pd.testing.assert_frame_equal(first.reset_index(drop=True), second)
            self.assertEqual(client.timeseries.get_range.call_count, 1)
            call = client.timeseries.get_range.call_args.kwargs
            self.assertEqual(call['schema'], 'ohlcv-1d')
            self.assertEqual(call['stype_in'], 'raw_symbol')
            self.assertEqual(call['symbols'], ['ESH7', 'ESZ6'])
            self.assertEqual(str(call['start'].tz), module.TIMEZONE)

    def test_failed_requests_are_not_cached(self):
        client = MagicMock()
        client.timeseries.get_range.side_effect = RuntimeError('HTTP 422')
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'OUTPUT_DIR', Path(tmp)):
            with self.assertRaises(RuntimeError):
                module.download_bars(client, 'ES', self.days, self.mapping)
            self.assertEqual(list(Path(tmp).rglob('*.csv')), [])

    def test_failed_block_preserves_other_blocks_and_contract_level_data(self):
        days = module.candidate_dates('2024-10-01', '2026-10-06')
        with patch.object(module, 'BLOCK_YEARS', 1):
            blocks = module.date_blocks(days)

        def bars(client, root, block_days, mapping, coverage):
            if block_days[0] == blocks[1][0]:
                raise RuntimeError('request failed')
            # Return every requested symbol on the first day, so non-T-selected
            # observations must also be retained in the contract-level file.
            return pd.DataFrame([
                {'date': block_days[0], 'raw_symbol': symbol, 'instrument_id': i,
                 **dict(zip(module.FIELDS, [1., 2., .5, 1.5, 10]))}
                for i, symbol in enumerate(sorted(set(mapping.raw_symbol)))
            ])

        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'OUTPUT_DIR', Path(tmp)), \
             patch.object(module, 'BLOCK_YEARS', 1), patch.object(module.db, 'Historical'), \
             patch.object(module, 'download_bars', side_effect=bars) as download, \
             self.assertLogs(module.LOG, level='WARNING'):
            status = module.download_asset('ES', 'test-key', days, None)
            self.assertEqual(download.call_count, 3)
            self.assertEqual(status['failed_blocks'], 1)
            self.assertEqual(status['status'], 'error')
            suffix = '2024-10-01_2026-10-06'
            saved = pd.read_csv(Path(tmp) / f'ES_{suffix}.csv', header=[0, 1], index_col=0, parse_dates=[0])
            self.assertTrue(saved.index.equals(days))
            self.assertTrue(saved.loc[blocks[1]].isna().all().all())
            self.assertEqual(saved.loc[blocks[2][0], ('ES_T0', 'close')], 1.5)
            contracts = pd.read_csv(Path(tmp) / f'ES_{suffix}_contracts.csv')
            self.assertGreater(len(contracts), 4)
            self.assertTrue({'canonical_symbol', 'raw_symbol', 'instrument_id',
                             'contract_year', 'contract_month'}.issubset(contracts.columns))

    def test_asset_errors_do_not_stop_other_assets(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.multiple(module, START_DATE='2026-10-01', END_DATE='2026-10-06',
                            OUTPUT_DIR=Path(tmp), PRODUCTS={'ZT': None, 'ES': None}), \
             patch.object(module.db, 'Historical') as historical, \
             patch.object(module, 'load_dotenv'), \
             patch.object(module.os, 'getenv', return_value='test-key'), \
             patch.object(module, 'download_asset') as asset, \
             self.assertLogs(module.LOG, level='WARNING'):
            historical.return_value.metadata.get_dataset_range.side_effect = RuntimeError('metadata unavailable')
            def result(root, *args):
                if root == 'ZT': raise RuntimeError('connection failed')
                return {'root': root, 'status': 'ok', 'error': ''}
            asset.side_effect = result
            module.main()
            statuses = pd.read_csv(Path(tmp) / 'download_status_2026-10-01_2026-10-06.csv').set_index('root')
            self.assertEqual(statuses.loc['ZT', 'status'], 'error')
            self.assertEqual(statuses.loc['ES', 'status'], 'ok')


if __name__ == '__main__':
    unittest.main()
