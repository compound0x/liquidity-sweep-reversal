# Liquidity Sweep → Reversal Scanner

Automated 1-minute liquidity sweep and structure-reversal scanner using Yahoo Finance data.

## What it does

- Uses the 1H reference candles closing at 03:00, 10:00 and 14:00 New York time.
- Scans the following hour on 1-minute candles for a sweep of the reference high/low.
- Looks for an opposite-direction structure shift with displacement and volume confirmation.
- Generates a standalone `index.html` dashboard.
- Runs automatically through GitHub Actions and publishes the dashboard to GitHub Pages.

## Run locally

```bash
python -m pip install -r requirements.txt
python liquidity_sweep_scanner.py
```

The generated dashboard is `index.html`.

## GitHub Actions

The workflow can be started manually from **Actions → Liquidity Sweep Scanner → Run workflow**.

It is also scheduled for `01:15 UTC` on Tuesday through Saturday. This covers the Monday through Friday US trading sessions, including the Friday session after the US close.

The workflow retries transient Yahoo Finance failures, verifies that `index.html` was produced, and publishes the result to the `gh-pages` branch.

## GitHub Pages

In repository **Settings → Pages**, set the Pages source to the `gh-pages` branch if it is not already configured. The dashboard will then be available at:

`https://compound0x.github.io/liquidity-sweep-reversal/`

## Important data limitation

Yahoo Finance 1-minute history is limited and can be delayed. The scanner therefore treats missing/stale data as a data-availability issue rather than a trading signal.

Signals are observations for research and are not trade recommendations.
