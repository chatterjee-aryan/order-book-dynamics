# Economic Event Date Sources

Use announcement dates, rather than the reference month of the published data.

| Event | Historical dates | Official source |
|---|---|---|
| FOMC | [Meeting dates CSV](https://github.com/edwardbickerton/Fed-Scraper/blob/main/meeting_dates.csv) | [Federal Reserve calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) |
| CPI | [ALFRED release dates TXT](https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=10) | [BLS CPI schedule](https://www.bls.gov/schedule/news_release/cpi.htm) |
| PCE | [ALFRED release dates TXT](https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=54) | [BEA release schedule](https://www.bea.gov/news/schedule) |
| NFP | [ALFRED release dates TXT](https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=50) | [BLS Employment Situation schedule](https://www.bls.gov/schedule/news_release/empsit.htm) |

PCE inflation is published within **Personal Income and Outlays**. NFP is
published within **Employment Situation**.

The ALFRED TXT files have descriptive headers followed by one `YYYY-MM-DD`
release date per line. No API key is required for these downloads. ALFRED lists
revision dates too and may use provider/FRED availability dates when original
release dates are unavailable; cross-check uncertain dates against the agencies.
