from __future__ import annotations

# --- dependencies -------------------------------------------------------------
import importlib, subprocess, sys

for _pkg, _mod in [("yfinance", "yfinance"), ("pandas", "pandas"),
                   ("numpy", "numpy"), ("matplotlib", "matplotlib")]:
    try:
        importlib.import_module(_mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", _pkg], check=False)

import yfinance as yf
print("yfinance", yf.__version__)

"""Liquidity sweep -> structure reversal scanner (yfinance).

Reference candle = the 1H bar CLOSING at 03:00 / 10:00 / 14:00 New York time.
The following hour is scanned on 1m candles for a sweep of that candle's high/low,
then for a structure shift with displacement + volume in the opposite direction.

Run this ONCE. In live mode the polling loop is internal - you do not re-run anything
per minute; the function refetches and re-evaluates on its own schedule until you stop it.
"""

# =============================== SETTINGS =====================================
MODE = "scan"              # "scan" = replay recent history | "live" = poll continuously
SCAN_DAYS = 2              # history mode: days of 1m data to replay (Yahoo caps at ~30)
LIVE_POLL_SECONDS = 30     # live mode: seconds between refresh cycles
LIVE_MAX_MINUTES = None    # live mode: auto-stop after N minutes (None = until interrupted)
ASSET_FILTER = None        # e.g. ["NAS100", "XAUUSD", "BTCUSD"] — None = every asset
SAVE_HTML = "index.html"   # also write a standalone page; None = inline only
# ==============================================================================

# --- imports & configuration --------------------------------------------------
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

NY = ZoneInfo("America/New_York")   # handles EST/EDT automatically
UTC = ZoneInfo("UTC")


@dataclass
class Config:
    # ---- sessions -----------------------------------------------------------
    # NY local hour at which the reference 1H candle CLOSES.
    # The monitored 1m window is [hour, hour + window_minutes).
    session_hours: tuple = (3, 10, 14)
    window_minutes: int = 60

    # ---- sweep detection ----------------------------------------------------
    sweep_buffer_atr: float = 0.0        # require excursion > buffer * ATR beyond the level
    sweep_requires_close_inside: bool = False   # True => only count a sweep if the candle closes back inside

    # ---- market structure ---------------------------------------------------
    swing_n: int = 2                     # fractal pivot: n bars each side (confirmed n bars later)
    structure_lookback: int = 45         # how far back (bars) from the sweep we may take the protective swing

    # ---- displacement -------------------------------------------------------
    atr_len: int = 20                    # 1m ATR length
    disp_atr_mult: float = 1.5           # body must be >= mult * ATR
    disp_body_ratio: float = 0.55        # body / range
    disp_lookback: int = 3               # displacement may occur on the break candle or up to N-1 bars before

    # ---- volume -------------------------------------------------------------
    vol_len: int = 20
    vol_mult: float = 1.5                # volume >= mult * rolling mean volume
    require_volume: str = "auto"         # "auto" (skip when the feed has no volume) | True | False

    # ---- behaviour ----------------------------------------------------------
    one_reversal_per_session: bool = True   # stop a session once a reversal fires
    data_lag_grace_min: int = 20            # keep evaluating a window this long past its close (Yahoo delay)


CFG = Config()
CFG

# --- universe -----------------------------------------------------------------
@dataclass
class Asset:
    name: str
    symbol: str          # yfinance ticker
    kind: str            # index | fx | metal | energy | crypto
    digits: int = 2
    weekends: bool = False   # crypto trades 24/7 -> sessions also run Sat/Sun

    def fmt(self, x: float) -> str:
        return f"{x:,.{self.digits}f}"


# Index / commodity legs use FUTURES tickers rather than cash indices, because the cash
# indices (^NDX, ^GSPC, ^DJI) publish no prints at 03:00 NY and carry no volume.
UNIVERSE_CORE = {
    "NAS100": Asset("NAS100", "NQ=F", "index",  2),
    "US500":  Asset("US500",  "ES=F", "index",  2),
    "US30":   Asset("US30",   "YM=F", "index",  0),
    "XAUUSD": Asset("XAUUSD", "GC=F", "metal",  2),
    "XAGUSD": Asset("XAGUSD", "SI=F", "metal",  3),
    "OIL":    Asset("OIL",    "CL=F", "energy", 2),
}

# Spot FX (=X) has price but Volume is always 0 on Yahoo -> the volume filter auto-disables.
UNIVERSE_FX_SPOT = {
    "EURUSD": Asset("EURUSD", "EURUSD=X", "fx", 5),
    "GBPUSD": Asset("GBPUSD", "GBPUSD=X", "fx", 5),
    "USDJPY": Asset("USDJPY", "USDJPY=X", "fx", 3),
    "AUDUSD": Asset("AUDUSD", "AUDUSD=X", "fx", 5),
    "USDCAD": Asset("USDCAD", "USDCAD=X", "fx", 5),
    "USDCHF": Asset("USDCHF", "USDCHF=X", "fx", 5),
    "NZDUSD": Asset("NZDUSD", "NZDUSD=X", "fx", 5),
    "EURJPY": Asset("EURJPY", "EURJPY=X", "fx", 3),
    "GBPJPY": Asset("GBPJPY", "GBPJPY=X", "fx", 3),
}

# CME currency futures — real traded volume, so displacement+volume confirmation works properly.
UNIVERSE_FX_FUTURES = {
    "EURUSD": Asset("EURUSD", "6E=F", "fx", 5),
    "GBPUSD": Asset("GBPUSD", "6B=F", "fx", 5),
    "JPYUSD": Asset("JPYUSD", "6J=F", "fx", 7),
    "AUDUSD": Asset("AUDUSD", "6A=F", "fx", 5),
    "CADUSD": Asset("CADUSD", "6C=F", "fx", 5),
    "CHFUSD": Asset("CHFUSD", "6S=F", "fx", 5),
}

# Crypto — 24/7, so weekends=True. Yahoo reports real aggregated volume on these.
UNIVERSE_CRYPTO = {
    "BTCUSD":  Asset("BTCUSD",  "BTC-USD",  "crypto", 2, weekends=True),
    "ETHUSD":  Asset("ETHUSD",  "ETH-USD",  "crypto", 2, weekends=True),
    "SOLUSD":  Asset("SOLUSD",  "SOL-USD",  "crypto", 3, weekends=True),
    "XRPUSD":  Asset("XRPUSD",  "XRP-USD",  "crypto", 5, weekends=True),
    "BNBUSD":  Asset("BNBUSD",  "BNB-USD",  "crypto", 2, weekends=True),
    "DOGEUSD": Asset("DOGEUSD", "DOGE-USD", "crypto", 6, weekends=True),
}

USE_FX_FUTURES = False   # flip to True to get volume-confirmed FX signals

UNIVERSE = {**UNIVERSE_CORE,
            **(UNIVERSE_FX_FUTURES if USE_FX_FUTURES else UNIVERSE_FX_SPOT),
            **UNIVERSE_CRYPTO}

print(f"{len(UNIVERSE)} assets "
      f"({sum(a.weekends for a in UNIVERSE.values())} trade weekends):")
print("  " + ", ".join(f"{k}({v.symbol})" for k, v in UNIVERSE.items()))

# --- data layer ---------------------------------------------------------------
_CACHE: dict = {}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns, force NY timezone, keep OHLCV as float."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    df.index = df.index.tz_convert(NY)
    df = df.astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_1m(symbol: str, days: int = 7, ttl: int = 45, force: bool = False) -> pd.DataFrame:
    """1-minute OHLCV in New York time.

    Yahoo limits: interval='1m' -> max ~30 days back, max 7 days per request.
    """
    days = max(1, min(int(days), 30))
    key = (symbol, days)
    now = time.time()
    if not force and key in _CACHE and now - _CACHE[key][0] < ttl:
        return _CACHE[key][1]

    frames = []
    tk = yf.Ticker(symbol)
    if days <= 7:
        try:
            frames.append(tk.history(period=f"{days}d", interval="1m",
                                     prepost=True, auto_adjust=False, raise_errors=False))
        except Exception as exc:
            print(f"  ! {symbol}: {exc}")
    else:
        end = datetime.now(UTC)
        cur = end - timedelta(days=days)
        while cur < end:
            chunk_end = min(cur + timedelta(days=7), end)
            try:
                frames.append(tk.history(start=cur, end=chunk_end, interval="1m",
                                         prepost=True, auto_adjust=False, raise_errors=False))
            except Exception as exc:
                print(f"  ! {symbol} [{cur:%Y-%m-%d}]: {exc}")
            cur = chunk_end

    frames = [f for f in frames if f is not None and len(f)]
    df = _normalise(pd.concat(frames)) if frames else pd.DataFrame()
    _CACHE[key] = (now, df)
    return df


def has_volume(df: pd.DataFrame) -> bool:
    """Yahoo returns Volume==0 for spot FX; detect that so the filter can be skipped."""
    if df.empty or "Volume" not in df:
        return False
    return float(df["Volume"].fillna(0).gt(0).mean()) > 0.5


def prepare(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    """Attach ATR, rolling volume mean, body/range — computed on the FULL series so the
    session slice never starts with a cold indicator."""
    if df.empty:
        return df
    d = df.copy()
    prev = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"],
                    (d["High"] - prev).abs(),
                    (d["Low"] - prev).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(cfg.atr_len, min_periods=max(2, cfg.atr_len // 2)).mean()
    d["VolMA"] = d["Volume"].rolling(cfg.vol_len, min_periods=max(2, cfg.vol_len // 2)).mean()
    d["Body"] = (d["Close"] - d["Open"]).abs()
    d["Range"] = (d["High"] - d["Low"])
    return d

# --- structure primitives -----------------------------------------------------
def swing_flags(a: pd.DataFrame, n: int):
    """Fractal pivots: bar i is a swing high if its high is the max of the +/- n window.
    A pivot at index q is only *confirmed* n bars later, so the engine never reads a pivot
    with q > current_pos - n (no look-ahead)."""
    w = 2 * n + 1
    hi, lo = a["High"], a["Low"]
    is_sh = (hi == hi.rolling(w, center=True).max()) & hi.notna()
    is_sl = (lo == lo.rolling(w, center=True).min()) & lo.notna()
    return is_sh.to_numpy(), is_sl.to_numpy()


def _last_confirmed_pivot(flags: np.ndarray, pos: int, cfg: Config, floor_pos: int):
    """Most recent pivot confirmed as of bar `pos`."""
    start = pos - cfg.swing_n
    for k in range(start, max(floor_pos, 0) - 1, -1):
        if k < 0:
            break
        if flags[k]:
            return k
    return None


def check_reversal(a: pd.DataFrame, pos: int, sweep: dict, cfg: Config,
                   is_sh: np.ndarray, is_sl: np.ndarray, use_volume: bool):
    """Market-structure shift + displacement + volume, in the direction opposite the sweep."""
    side = "SHORT" if sweep["side"] == "HIGH" else "LONG"
    floor_pos = max(0, sweep["pos"] - cfg.structure_lookback)
    flags = is_sl if side == "SHORT" else is_sh

    q = _last_confirmed_pivot(flags, pos, cfg, floor_pos)
    if q is None or q >= pos:
        return None

    level = float(a["Low"].iat[q] if side == "SHORT" else a["High"].iat[q])
    close = float(a["Close"].iat[pos])

    # 1) structure break
    if side == "SHORT" and not close < level:
        return None
    if side == "LONG" and not close > level:
        return None

    # 2) displacement (+3 volume) on the break candle or within disp_lookback bars
    disp = None
    for k in range(pos, max(-1, pos - cfg.disp_lookback), -1):
        o, h, l, c = (float(a["Open"].iat[k]), float(a["High"].iat[k]),
                      float(a["Low"].iat[k]), float(a["Close"].iat[k]))
        rng, body, atr = h - l, abs(c - o), float(a["ATR"].iat[k])
        if not np.isfinite(atr) or atr <= 0 or rng <= 0:
            continue
        if side == "SHORT" and c >= o:
            continue
        if side == "LONG" and c <= o:
            continue
        if body < cfg.disp_atr_mult * atr:
            continue
        if body / rng < cfg.disp_body_ratio:
            continue
        vol_ratio = np.nan
        if use_volume:
            vma = float(a["VolMA"].iat[k])
            vol_ratio = float(a["Volume"].iat[k]) / vma if vma > 0 else np.nan
            if not (np.isfinite(vol_ratio) and vol_ratio >= cfg.vol_mult):
                continue
        disp = dict(disp_pos=k, body_atr=body / atr, body_ratio=body / rng, vol_ratio=vol_ratio)
        break

    if disp is None:
        return None
    return dict(side=side, mss_level=level, swing_pos=q, **disp)

# --- session engine -----------------------------------------------------------
SIGNAL_KINDS = ("SWEEP", "REVERSAL", "SWEEP_NO_REVERSAL")


def analyze_session(df: pd.DataFrame, asset: Asset, session_start: datetime,
                    cfg: Config = CFG, use_volume: bool | None = None) -> dict | None:
    """Run the sweep -> reversal state machine over one [session_start, +window) 1m window."""
    ref_start = session_start - timedelta(hours=1)
    win_end = session_start + timedelta(minutes=cfg.window_minutes)

    ref = df.loc[(df.index >= ref_start) & (df.index < session_start)]
    if len(ref) < 5:
        return None                                   # no reference candle -> nothing to sweep
    ref_high, ref_low = float(ref["High"].max()), float(ref["Low"].min())

    buf = timedelta(minutes=cfg.structure_lookback + cfg.swing_n + 5)
    a = df.loc[(df.index >= session_start - buf) & (df.index < win_end)]
    if a.empty:
        return None
    live_pos = np.where(a.index >= session_start)[0]
    if len(live_pos) == 0:
        return None

    if use_volume is None:
        use_volume = has_volume(df) if cfg.require_volume == "auto" else bool(cfg.require_volume)

    is_sh, is_sl = swing_flags(a, cfg.swing_n)
    session_key = f"{session_start:%Y-%m-%d %H:%M}"
    signals, sweep, reversed_ = [], None, False

    for pos in live_pos:
        ts = a.index[pos]
        hi, lo, cl = float(a["High"].iat[pos]), float(a["Low"].iat[pos]), float(a["Close"].iat[pos])
        atr = float(a["ATR"].iat[pos]) if np.isfinite(a["ATR"].iat[pos]) else 0.0
        pad = cfg.sweep_buffer_atr * atr

        # ---------- 1. sweep -------------------------------------------------
        took_high = hi > ref_high + pad
        took_low = lo < ref_low - pad
        if cfg.sweep_requires_close_inside:
            took_high = took_high and cl < ref_high
            took_low = took_low and cl > ref_low
        side = None
        if took_high and took_low:
            side = "HIGH" if (hi - ref_high) >= (ref_low - lo) else "LOW"
        elif took_high:
            side = "HIGH"
        elif took_low:
            side = "LOW"

        if side and (sweep is None or sweep["side"] != side):
            sweep = dict(side=side, pos=int(pos), time=ts,
                         level=ref_high if side == "HIGH" else ref_low,
                         extreme=hi if side == "HIGH" else lo)
            reversed_ = False
            excursion = abs(sweep["extreme"] - sweep["level"])
            signals.append(dict(
                time=ts, asset=asset.name, symbol=asset.symbol, session=session_key,
                session_start=session_start, kind="SWEEP", side=side, price=sweep["extreme"],
                ref_high=ref_high, ref_low=ref_low,
                excursion=float(excursion),
                excursion_atr=float(excursion / atr) if atr > 0 else float("nan"),
                minute=int((ts - session_start).total_seconds() // 60),
                detail=(f"took {'buy-side' if side == 'HIGH' else 'sell-side'} liquidity @ "
                        f"{asset.fmt(sweep['level'])} by {asset.fmt(excursion)}"
                        f"{f' ({excursion / atr:.2f} ATR)' if atr > 0 else ''} "
                        f"-> now watching for {'SHORT' if side == 'HIGH' else 'LONG'} reversal")))
        elif side and sweep is not None:
            sweep["extreme"] = max(sweep["extreme"], hi) if side == "HIGH" else min(sweep["extreme"], lo)

        # ---------- 2. reversal ---------------------------------------------
        if sweep is not None and pos > sweep["pos"] and not (reversed_ and cfg.one_reversal_per_session):
            rv = check_reversal(a, int(pos), sweep, cfg, is_sh, is_sl, use_volume)
            if rv:
                reversed_ = True
                vtxt = f", vol {rv['vol_ratio']:.2f}x" if np.isfinite(rv["vol_ratio"]) else ", vol n/a"
                signals.append(dict(
                    time=ts, asset=asset.name, symbol=asset.symbol, session=session_key,
                    session_start=session_start, kind="REVERSAL", side=rv["side"], price=cl,
                    ref_high=ref_high, ref_low=ref_low,
                    mss_level=float(rv["mss_level"]), body_atr=float(rv["body_atr"]),
                    body_ratio=float(rv["body_ratio"]), vol_ratio=float(rv["vol_ratio"]),
                    minute=int((ts - session_start).total_seconds() // 60),
                    lag_min=int((ts - sweep["time"]).total_seconds() // 60),
                    detail=(f"MSS close {'below' if rv['side'] == 'SHORT' else 'above'} swing "
                            f"{asset.fmt(rv['mss_level'])} | displacement {rv['body_atr']:.2f}x ATR, "
                            f"body {rv['body_ratio']:.0%}{vtxt} | "
                            f"{int((ts - sweep['time']).total_seconds() // 60)}m after sweep")))
                if cfg.one_reversal_per_session:
                    break

    last_ts = a.index[-1]
    complete = last_ts >= win_end - timedelta(minutes=1)

    if complete and sweep is not None and not reversed_:
        signals.append(dict(
            time=win_end, asset=asset.name, symbol=asset.symbol, session=session_key,
            session_start=session_start, kind="SWEEP_NO_REVERSAL",
            side=sweep["side"], price=sweep["extreme"], ref_high=ref_high, ref_low=ref_low,
            minute=cfg.window_minutes,
            detail="window closed — liquidity taken but no valid structure shift with displacement + volume"))

    outcome = ("SWEEP+REVERSAL" if reversed_ else "SWEEP_ONLY" if sweep else "NONE")
    return dict(asset=asset.name, symbol=asset.symbol, session=session_key,
                session_start=session_start, ref_high=ref_high, ref_low=ref_low,
                outcome=outcome, complete=bool(complete), volume_used=bool(use_volume),
                signals=signals)


def session_starts(first: datetime, last: datetime, cfg: Config = CFG,
                   weekends: bool = False):
    """All NY session anchors between two timestamps.

    weekends=False -> Mon-Fri only (futures/FX). weekends=True -> every day (crypto).
    """
    out, day = [], first.astimezone(NY).date()
    end_day = last.astimezone(NY).date()
    while day <= end_day:
        if weekends or day.weekday() < 5:
            for h in cfg.session_hours:
                ts = datetime(day.year, day.month, day.day, h, 0, tzinfo=NY)
                if first <= ts <= last:
                    out.append(ts)
        day += timedelta(days=1)
    return sorted(out)

# --- notification -------------------------------------------------------------
ICONS = {"SWEEP": "🟡", "REVERSAL": "🟢", "SWEEP_NO_REVERSAL": "⚪"}


def console_notify(sig: dict) -> None:
    asset = UNIVERSE.get(sig["asset"])
    price = asset.fmt(sig["price"]) if asset else f"{sig['price']:.5f}"
    print(f"{ICONS.get(sig['kind'], '•')} [{sig['time']:%Y-%m-%d %H:%M} NY] "
          f"{sig['asset']:<7} {sig['session'][-5:]} session | {sig['kind']:<18} "
          f"{sig['side']:<5} @ {price}\n        {sig['detail']}")


# Add your own sinks here (Telegram / Discord / desktop toast / etc.)
#
# import requests
# def discord_notify(sig):
#     requests.post(WEBHOOK_URL, json={"content": f"{sig['kind']} {sig['asset']} {sig['detail']}"})
# NOTIFY_HOOKS.append(discord_notify)

NOTIFY_HOOKS = [console_notify]


def notify(sig: dict) -> None:
    for hook in NOTIFY_HOOKS:
        try:
            hook(sig)
        except Exception as exc:
            print(f"  ! notify hook failed: {exc}")

# --- HTML rendering -----------------------------------------------------------
try:
    from IPython.display import HTML, display
    _IPY = True
except ImportError:
    _IPY = False

SWP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap');
.swp-root{
  --bg:#131A24; --panel:#1A2330; --panel-2:#212C3C; --rule:#2A3648;
  --ink:#E6EBF2; --mute:#8A97AB; --dim:#5D6A7E;
  --sweep:#E8A33D; --long:#4FC08D; --short:#F2635F; --none:#66748A;
  background:var(--bg); color:var(--ink); padding:26px 26px 20px;
  border-radius:10px; font-family:'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  font-size:13px; line-height:1.5; -webkit-font-smoothing:antialiased;
}
.swp-root *{box-sizing:border-box}
.swp-eyebrow{font-family:'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif;
  font-weight:600; font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--dim)}
.swp-mono{font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace; font-variant-numeric:tabular-nums}

.swp-head{display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap}
.swp-head>div{margin-right:20px}
.swp-head>div:last-child{margin-right:0}
.swp-title{font-family:'IBM Plex Sans Condensed','IBM Plex Sans',sans-serif; font-weight:600;
  font-size:19px; letter-spacing:.1em; text-transform:uppercase; margin:0; color:var(--ink)}
.swp-sub{color:var(--mute); font-size:11.5px; margin-top:3px}
.swp-gen{font-size:10.5px; color:var(--dim); text-align:right}

.swp-stats{display:flex; flex-wrap:wrap; margin:18px 0 4px;
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); padding:14px 0}
.swp-stat{margin-right:38px}
.swp-stat:last-child{margin-right:0}
.swp-stat b{display:block; font-family:'IBM Plex Mono',monospace; font-weight:500;
  font-size:25px; line-height:1.1; letter-spacing:-.01em}
.swp-stat span{display:block; margin-top:4px; font-family:'IBM Plex Sans Condensed',sans-serif;
  font-weight:600; font-size:10px; letter-spacing:.15em; text-transform:uppercase; color:var(--dim)}

.swp-group{margin-top:26px}
.swp-group-hd{display:flex; align-items:center; margin-bottom:11px}
.swp-group-hd>*{margin-right:12px}
.swp-group-hd>*:last-child{margin-right:0}
.swp-group-hd .swp-rule{flex:1; height:1px; background:var(--rule)}
.swp-group-hd .swp-n{font-size:10.5px; color:var(--dim)}
.swp-note{font-size:10.5px; color:var(--dim); font-style:italic}

.swp-card{background:var(--panel); border-left:3px solid var(--accent);
  border-radius:0 6px 6px 0; padding:13px 16px 11px; margin-bottom:9px}
.swp-card-hd{display:flex; align-items:baseline; flex-wrap:wrap}
.swp-card-hd>*{margin-right:12px}
.swp-card-hd>*:last-child{margin-right:0}
.swp-asset{font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:14px; letter-spacing:.02em}
.swp-when{color:var(--mute); font-size:11px}
.swp-levels{margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--mute)}
.swp-levels i{font-style:normal; color:var(--dim); padding:0 5px}
.swp-levels u{text-decoration:none; color:var(--ink)}

.swp-track{position:relative; height:58px; margin:9px 0 4px}
.swp-axis{position:absolute; left:0; right:0; top:29px; height:1px; background:var(--rule)}
.swp-axis:before,.swp-axis:after{content:''; position:absolute; top:-3px; width:1px; height:7px; background:var(--rule)}
.swp-axis:before{left:0} .swp-axis:after{right:0}
.swp-mark{position:absolute; transform:translateX(-50%); text-align:center; white-space:nowrap}
.swp-mark.sweep{top:6px}
.swp-mark.rev{top:33px}
.swp-glyph{width:10px; height:10px; margin:0 auto; background:var(--sweep); transform:rotate(45deg)}
.swp-mark.rev .swp-glyph{width:13px; height:13px; border-radius:50%; transform:none;
  background:var(--panel); border:3px solid var(--c)}
.swp-mark.sweep .swp-glyph{margin-bottom:4px}
.swp-mark.rev .swp-glyph{margin-bottom:0; margin-top:0}
.swp-mlabel{font-family:'IBM Plex Mono',monospace; font-size:9.5px; color:var(--mute); letter-spacing:.02em}
.swp-mark.rev .swp-mlabel{margin-top:4px}
.swp-ends{display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace;
  font-size:9.5px; color:var(--dim); margin-top:-2px}

.swp-ev{display:flex; align-items:baseline; padding:4px 0 0; flex-wrap:wrap}
.swp-ev>*{margin-right:9px}
.swp-ev>*:last-child{margin-right:0}
.swp-tag{font-family:'IBM Plex Sans Condensed',sans-serif; font-weight:600; font-size:9.5px;
  letter-spacing:.13em; text-transform:uppercase; padding:2px 7px; border-radius:3px;
  background:var(--tagbg); color:var(--tagfg); flex:none}
.swp-ev p{margin:0; color:var(--mute); font-size:11.5px}
.swp-chip{font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--ink);
  background:var(--panel-2); padding:2px 7px; border-radius:3px}

.swp-empty{padding:30px 0 6px; color:var(--mute)}
.swp-empty b{display:block; color:var(--ink); font-size:14px; margin-bottom:5px; font-weight:500}
.swp-foot{margin-top:22px; padding-top:12px; border-top:1px solid var(--rule);
  font-size:10.5px; color:var(--dim)}
.swp-live{margin:6px 0}
@media (max-width:620px){
  .swp-root{padding:18px 16px}
  .swp-levels{margin-left:0; width:100%}
  .swp-stats{gap:22px}
}
</style>
"""

_ACCENT = {"LONG": "var(--long)", "SHORT": "var(--short)"}
_TAG = {"SWEEP": ("rgba(232,163,61,.15)", "#E8A33D"),
        "REVERSAL": ("rgba(79,192,141,.15)", "#4FC08D"),
        "SWEEP_NO_REVERSAL": ("rgba(102,116,138,.16)", "#8A97AB")}


def _num(sig, key):
    v = sig.get(key)
    return v if isinstance(v, (int, float)) and np.isfinite(v) else None


def _pct(sig, cfg):
    m = sig.get("minute")
    if m is None or not np.isfinite(m):
        m = (sig["time"] - sig["session_start"]).total_seconds() / 60
    return max(3.5, min(96.5, 100.0 * float(m) / cfg.window_minutes))


def _chips(sig, asset):
    out = []
    if sig["kind"] == "SWEEP":
        if _num(sig, "excursion") is not None:
            out.append(f"+{asset.fmt(sig['excursion'])} beyond")
        if _num(sig, "excursion_atr") is not None:
            out.append(f"{sig['excursion_atr']:.2f}× ATR")
    elif sig["kind"] == "REVERSAL":
        if _num(sig, "body_atr") is not None:
            out.append(f"{sig['body_atr']:.1f}× ATR body")
        if _num(sig, "body_ratio") is not None:
            out.append(f"{sig['body_ratio']:.0%} body")
        vr = _num(sig, "vol_ratio")
        out.append(f"{vr:.1f}× vol" if vr is not None else "no vol data")
        if _num(sig, "lag_min") is not None:
            out.append(f"+{int(sig['lag_min'])} min")
    return "".join(f'<span class="swp-chip">{c}</span>' for c in out)


def _prose(sig):
    """The chips carry the numbers; the sentence carries only what they cannot."""
    d = sig["detail"]
    if sig["kind"] == "SWEEP":
        return d.split(" by ")[0] or d
    if sig["kind"] == "REVERSAL":
        return d.split(" | ")[0] or d
    return d


def _card(asset_name, sigs, ses, cfg):
    asset = UNIVERSE.get(asset_name)
    rev = next((s for s in sigs if s["kind"] == "REVERSAL"), None)
    accent = _ACCENT[rev["side"]] if rev else ("var(--sweep)" if ses["outcome"] != "NONE" else "var(--none)")
    ss = ses["session_start"]

    marks = []
    for s in sigs:
        if s["kind"] == "SWEEP":
            marks.append(f'<div class="swp-mark sweep" style="left:{_pct(s, cfg):.2f}%">'
                         f'<div class="swp-mlabel">{s["side"]} {s["time"]:%H:%M}</div>'
                         f'<div class="swp-glyph"></div></div>')
        elif s["kind"] == "REVERSAL":
            marks.append(f'<div class="swp-mark rev" style="left:{_pct(s, cfg):.2f}%;'
                         f'--c:{_ACCENT[s["side"]]}"><div class="swp-glyph"></div>'
                         f'<div class="swp-mlabel">{s["side"]} {s["time"]:%H:%M}</div></div>')

    events = []
    for s in sigs:
        bg, fg = _TAG[s["kind"]]
        label = {"SWEEP": "Sweep", "REVERSAL": "Reversal",
                 "SWEEP_NO_REVERSAL": "No reversal"}[s["kind"]]
        events.append(
            f'<div class="swp-ev" style="--tagbg:{bg};--tagfg:{fg}">'
            f'<span class="swp-tag">{label}</span>{_chips(s, asset)}'
            f'<p>{_prose(s)}</p></div>')

    end = ss + timedelta(minutes=cfg.window_minutes)
    return (f'<div class="swp-card" style="--accent:{accent}">'
            f'<div class="swp-card-hd"><span class="swp-asset">{asset_name}</span>'
            f'<span class="swp-when">{ss:%a %d %b}</span>'
            f'<span class="swp-levels">H <u>{asset.fmt(ses["ref_high"])}</u><i>/</i>'
            f'L <u>{asset.fmt(ses["ref_low"])}</u></span></div>'
            f'<div class="swp-track"><div class="swp-axis"></div>{"".join(marks)}</div>'
            f'<div class="swp-ends"><span>{ss:%H:%M}</span><span>{end:%H:%M}</span></div>'
            f'{"".join(events)}</div>')


def dashboard_html(signals_df, sessions_df, cfg=CFG, days=None, fragment=True,
                   reversals_first=True):
    """Build the session review dashboard.

    Sessions with no sweep are omitted. reversals_first lifts every session that
    followed through into a block at the top, ahead of the per-window groups.
    """
    gen = datetime.now(NY)
    n_ses = len(sessions_df) if sessions_df is not None else 0
    swept = int((sessions_df["outcome"] != "NONE").sum()) if n_ses else 0
    reved = int((sessions_df["outcome"] == "SWEEP+REVERSAL").sum()) if n_ses else 0
    conv = f"{reved / swept:.0%}" if swept else "—"
    novol = sorted(sessions_df.loc[~sessions_df["volume_used"], "asset"].unique()) if n_ses else []

    body = [
        '<div class="swp-head"><div><h1 class="swp-title">Sweep → Reversal</h1>',
        f'<div class="swp-sub">1H reference candles closing '
        f'{" · ".join(f"{h:02d}:00" for h in cfg.session_hours)} New York, '
        f'swept on the 1-minute chart</div></div>',
        f'<div class="swp-gen swp-mono">{gen:%d %b %Y · %H:%M} NY'
        f'{f"<br>{days}d of history" if days else ""}</div></div>',
        '<div class="swp-stats">',
        f'<div class="swp-stat"><b>{n_ses}</b><span>Sessions</span></div>',
        f'<div class="swp-stat"><b style="color:var(--sweep)">{swept}</b><span>Swept</span></div>',
        f'<div class="swp-stat"><b style="color:var(--long)">{reved}</b><span>Reversed</span></div>',
        f'<div class="swp-stat"><b>{conv}</b><span>Follow-through</span></div></div>',
    ]

    if signals_df is None or signals_df.empty:
        body.append('<div class="swp-empty"><b>No sweeps in this range.</b>'
                    'Nothing took the reference high or low. Extend <span class="swp-chip">days</span> '
                    'or lower <span class="swp-chip">sweep_buffer_atr</span> to widen the net.</div>')
    else:
        sig = signals_df.copy()
        sig["hour"] = sig["session_start"].dt.strftime("%H:%M")

        def _keys(block):
            k = block.groupby(["asset", "session"], as_index=False)["session_start"].max()
            return k.sort_values(["session_start", "asset"], ascending=[False, True])

        # Sessions that followed through are the ones worth reading first; the rest
        # stay grouped by window underneath, newest first.
        done = {(a, s) for a, s in
                sig.loc[sig.kind == "REVERSAL", ["asset", "session"]].drop_duplicates().values}
        is_rev = [(a, s) in done for a, s in zip(sig.asset, sig.session)]

        groups = []
        if reversals_first and done:
            groups.append(("Sweep → reversal", "all windows", _keys(sig[is_rev])))
            rest = sig[[not b for b in is_rev]]
        else:
            rest = sig
        for hour in sorted(rest["hour"].unique()):
            block = rest[rest["hour"] == hour]
            if block.empty:
                continue
            note = "swept, no reversal" if (reversals_first and done) else ""
            groups.append((f"{hour} NY window", note, _keys(block)))

        for title, note, keys in groups:
            note_html = f'<span class="swp-note">{note}</span>' if note else ""
            body.append(f'<div class="swp-group"><div class="swp-group-hd">'
                        f'<span class="swp-eyebrow">{title}</span>{note_html}'
                        f'<span class="swp-rule"></span>'
                        f'<span class="swp-n swp-mono">{len(keys)}</span></div>')
            for _, k in keys.iterrows():
                rows = sig[(sig.asset == k.asset) & (sig.session == k.session)]
                ses = sessions_df[(sessions_df.asset == k.asset) &
                                  (sessions_df.session == k.session)]
                if ses.empty:
                    continue
                body.append(_card(k.asset, rows.to_dict("records"), ses.iloc[0], cfg))
            body.append('</div>')

    foot = ("Yahoo 1m data is delayed and capped at ~30 days. "
            "Signals are observations, not trades.")
    if novol:
        foot += f" No volume feed for {', '.join(novol)} — those reversals cleared structure + displacement only."
    body.append(f'<div class="swp-foot">{foot}</div>')

    inner = SWP_CSS + '<div class="swp-root">' + "".join(body) + '</div>'
    if fragment:
        return inner
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Sweep → Reversal</title>'
            '<style>body{margin:0;padding:24px;background:#0C1119}</style></head>'
            f'<body>{inner}</body></html>')


def show_dashboard(signals_df, sessions_df, cfg=CFG, days=None):
    html = dashboard_html(signals_df, sessions_df, cfg, days)
    display(HTML(html)) if _IPY else print("IPython not available")


def export_html(signals_df, sessions_df, path="sweep_dashboard.html", cfg=CFG, days=None):
    """Write a standalone page you can open in a browser or share."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dashboard_html(signals_df, sessions_df, cfg, days, fragment=False))
    print(f"saved → {path}")
    return path


def html_notify(sig):
    """Live-mode sink: renders one signal as a card."""
    if not _IPY:
        return console_notify(sig)
    asset = UNIVERSE.get(sig["asset"])
    bg, fg = _TAG[sig["kind"]]
    accent = _ACCENT.get(sig["side"], "var(--sweep)") if sig["kind"] == "REVERSAL" else "var(--sweep)"
    label = {"SWEEP": "Sweep", "REVERSAL": "Reversal", "SWEEP_NO_REVERSAL": "No reversal"}[sig["kind"]]
    display(HTML(
        SWP_CSS + f'<div class="swp-root swp-live"><div class="swp-card" style="--accent:{accent}">'
        f'<div class="swp-card-hd"><span class="swp-asset">{sig["asset"]}</span>'
        f'<span class="swp-when">{sig["time"]:%H:%M} NY · {sig["session"][-5:]} window</span>'
        f'<span class="swp-levels">@ <u>{asset.fmt(sig["price"])}</u></span></div>'
        f'<div class="swp-ev" style="--tagbg:{bg};--tagfg:{fg}">'
        f'<span class="swp-tag">{label} {sig["side"]}</span>{_chips(sig, asset)}'
        f'<p>{_prose(sig)}</p></div></div></div>'))


# HTML cards in the notebook, plain text elsewhere.
NOTIFY_HOOKS[:] = [html_notify if _IPY else console_notify]

# --- historical scan ----------------------------------------------------------
def scan_history(assets: dict = None, days: int = 5, cfg: Config = CFG,
                 verbose: bool = True, show_no_signal: bool = False):
    """Replay the last `days` of 1m data through the engine.

    Returns (signals_df, sessions_df). Only sessions producing a signal are printed.
    """
    assets = assets or UNIVERSE
    all_signals, all_sessions = [], []

    for name, asset in assets.items():
        raw = fetch_1m(asset.symbol, days=days)
        if raw.empty:
            print(f"  ! {name} ({asset.symbol}): no 1m data returned")
            continue
        df = prepare(raw, cfg)
        uv = has_volume(df) if cfg.require_volume == "auto" else bool(cfg.require_volume)

        for ss in session_starts(df.index[0], df.index[-1], cfg, weekends=asset.weekends):
            res = analyze_session(df, asset, ss, cfg, use_volume=uv)
            if res is None:
                continue
            all_sessions.append({k: v for k, v in res.items() if k != "signals"})
            all_signals.extend(res["signals"])

    sig_df = pd.DataFrame(all_signals)
    ses_df = pd.DataFrame(all_sessions)
    if not sig_df.empty:
        sig_df = sig_df.sort_values(["time", "asset"]).reset_index(drop=True)
    if not ses_df.empty:
        ses_df = ses_df.sort_values(["session_start", "asset"]).reset_index(drop=True)

    if verbose:
        rows = sig_df if show_no_signal or sig_df.empty else sig_df
        if rows.empty:
            print("No signals in the scanned range.")
        else:
            print(f"===== {len(rows)} signals across {ses_df['session'].nunique() if not ses_df.empty else 0} sessions =====\n")
            cur = None
            for _, s in rows.iterrows():
                key = (s["asset"], s["session"])
                if key != cur:
                    cur = key
                    print(f"\n--- {s['asset']} | reference 1H candle closing {s['session']} NY "
                          f"| H {UNIVERSE[s['asset']].fmt(s['ref_high'])} / "
                          f"L {UNIVERSE[s['asset']].fmt(s['ref_low'])} ---")
                notify(s.to_dict())
    return sig_df, ses_df

# --- optional: plot a single session -----------------------------------------
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_session(asset_name: str, session_start, days: int = 5, cfg: Config = CFG):
    asset = UNIVERSE[asset_name]
    if isinstance(session_start, str):
        session_start = pd.Timestamp(session_start).tz_localize(NY).to_pydatetime()
    df = prepare(fetch_1m(asset.symbol, days=days), cfg)
    res = analyze_session(df, asset, session_start, cfg)
    if res is None:
        print("No data for that session.")
        return

    win_end = session_start + timedelta(minutes=cfg.window_minutes)
    view = df.loc[(df.index >= session_start - timedelta(hours=1)) & (df.index < win_end)]

    fig, ax = plt.subplots(figsize=(15, 6))
    w = 0.0005
    for ts, r in view.iterrows():
        up = r["Close"] >= r["Open"]
        c = "#1a9850" if up else "#d73027"
        ax.plot([ts, ts], [r["Low"], r["High"]], color=c, lw=0.7)
        ax.add_patch(plt.Rectangle((mdates.date2num(ts) - w, min(r["Open"], r["Close"])),
                                   2 * w, max(abs(r["Close"] - r["Open"]), 1e-9),
                                   color=c, alpha=0.85))
    ax.axhline(res["ref_high"], color="#1f77b4", ls="--", lw=1.2, label="ref 1H high")
    ax.axhline(res["ref_low"], color="#ff7f0e", ls="--", lw=1.2, label="ref 1H low")
    ax.axvline(session_start, color="grey", ls=":", lw=1.2)

    for s in res["signals"]:
        if s["kind"] == "SWEEP":
            ax.scatter(s["time"], s["price"], marker="v" if s["side"] == "HIGH" else "^",
                       s=170, color="gold", edgecolor="k", zorder=5, label="SWEEP")
        elif s["kind"] == "REVERSAL":
            ax.scatter(s["time"], s["price"], marker="*", s=320, color="lime",
                       edgecolor="k", zorder=5, label="REVERSAL")

    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), loc="best")
    ax.set_title(f"{asset_name} — {session_start:%Y-%m-%d %H:%M} NY session — {res['outcome']}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=NY))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# --- live runner --------------------------------------------------------------
def current_window(now: datetime, cfg: Config = CFG, weekends: bool = False):
    """Return the session anchor whose monitoring window contains `now` (+ data-lag grace)."""
    if not weekends and now.weekday() >= 5:
        return None
    for h in cfg.session_hours:
        ss = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if ss <= now < ss + timedelta(minutes=cfg.window_minutes + cfg.data_lag_grace_min):
            return ss
    return None


def next_window(now: datetime, cfg: Config = CFG, weekends: bool = False) -> datetime:
    cand = []
    for d in range(0, 8):
        day = now + timedelta(days=d)
        if not weekends and day.weekday() >= 5:
            continue
        for h in cfg.session_hours:
            ss = day.replace(hour=h, minute=0, second=0, microsecond=0)
            if ss > now:
                cand.append(ss)
    return min(cand)


def run_live(assets: dict = None, cfg: Config = CFG, poll_seconds: int = 30,
             lookback_days: int = 2, max_minutes: int | None = None, idle_sleep: int = 60):
    """Poll on every closed 1m candle inside an active window and push new signals only.

    max_minutes: stop after N minutes (None = run forever; Ctrl-C to stop).
    """
    assets = assets or UNIVERSE
    seen: set = set()
    started = datetime.now(NY)
    any_weekend = any(a.weekends for a in assets.values())
    print(f"Live scanner started {started:%Y-%m-%d %H:%M:%S} NY | "
          f"windows {[f'{h:02d}:00-{h + 1:02d}:00' for h in cfg.session_hours]} | "
          f"{len(assets)} assets ({sum(a.weekends for a in assets.values())} run weekends)\n")

    try:
        while True:
            now = datetime.now(NY)
            if max_minutes and (now - started).total_seconds() > max_minutes * 60:
                print("\nmax_minutes reached — stopping.")
                return
            weekend_now = now.weekday() >= 5
            ss = current_window(now, cfg, weekends=any_weekend)

            if ss is None:
                nxt = next_window(now, cfg, weekends=any_weekend)
                mins = (nxt - now).total_seconds() / 60
                print(f"\r[{now:%H:%M:%S}] idle — next window {nxt:%a %H:%M} NY "
                      f"(in {mins:.0f}m)   ", end="")
                time.sleep(min(idle_sleep, max(5, mins * 60 / 2)))
                continue

            for name, asset in assets.items():
                if weekend_now and not asset.weekends:
                    continue          # futures/FX closed
                try:
                    df = prepare(fetch_1m(asset.symbol, days=lookback_days, ttl=poll_seconds - 5), cfg)
                    if df.empty:
                        continue
                    res = analyze_session(df, asset, ss, cfg)
                    if not res:
                        continue
                    for sig in res["signals"]:
                        sid = (sig["asset"], sig["session"], sig["kind"], sig["side"],
                               sig["time"].isoformat())
                        if sid in seen:
                            continue
                        seen.add(sid)
                        print()
                        notify(sig)
                except Exception as exc:
                    print(f"\n  ! {name}: {exc}")

            print(f"\r[{datetime.now(NY):%H:%M:%S}] window {ss:%H:%M} active"
                  f"{' (weekend — crypto only)' if weekend_now else ''} — "
                  f"{len(seen)} signals so far   ", end="")
            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print("\nStopped by user.")

# =================================== RUN ======================================
def _selected_assets():
    if not ASSET_FILTER:
        return UNIVERSE
    missing = [a for a in ASSET_FILTER if a not in UNIVERSE]
    if missing:
        raise KeyError(f"unknown asset(s) {missing}; available: {list(UNIVERSE)}")
    return {a: UNIVERSE[a] for a in ASSET_FILTER}


def main():
    assets = _selected_assets()
    if MODE == "scan":
        sig, ses = scan_history(assets, days=SCAN_DAYS, cfg=CFG, verbose=False)
        show_dashboard(sig, ses, CFG, days=SCAN_DAYS)
        if SAVE_HTML:
            export_html(sig, ses, SAVE_HTML, CFG, days=SCAN_DAYS)
        return sig, ses
    if MODE == "live":
        run_live(assets, cfg=CFG, poll_seconds=LIVE_POLL_SECONDS,
                 max_minutes=LIVE_MAX_MINUTES)
        return None, None
    raise ValueError(f"MODE must be 'scan' or 'live', got {MODE!r}")


if __name__ == "__main__":
    signals_df, sessions_df = main()