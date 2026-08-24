"""Run the intraday scanner and send newly detected SWEEP/REVERSAL signals."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib import request

import liquidity_sweep_scanner as scanner

STATE_FILE = Path("alert_state.json")
DASHBOARD_FILE = "index.html"
USER_AGENT = "Mozilla/5.0 (compatible; LiquiditySweepScanner/1.0)"


def post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=15) as response:
        if response.status >= 300:
            raise RuntimeError(f"HTTP {response.status}")


#def discord_send(message: str) -> None:
#    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
#    if not webhook:
#        print("Discord secret not configured; skipping Discord notification.")
#        return
#    post_json(webhook, {"content": message})
#    print("Discord notification sent.")
#

def telegram_send(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram secrets not configured; skipping Telegram notification.")
        return
    post_json(
        f"https://api.telegram.org/bot{token}/sendMessage",
        {"chat_id": chat_id, "text": message, "disable_web_page_preview": False},
    )
    print("Telegram notification sent.")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_reference(now: datetime):
    for hour in scanner.CFG.session_hours:
        ss = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if ss + timedelta(minutes=5) <= now <= ss + timedelta(minutes=scanner.CFG.window_minutes):
            return ss
    return None


def signal_key(sig: dict) -> str:
    return "|".join([
        str(sig["asset"]), str(sig["session"]), str(sig["kind"]),
        str(sig["side"]), sig["time"].isoformat(),
    ])


def fmt_price(sig: dict) -> str:
    asset = scanner.UNIVERSE.get(sig["asset"])
    return asset.fmt(float(sig["price"])) if asset else f"{float(sig['price']):.5f}"


def format_alert(sig: dict) -> str:
    kind = sig["kind"]
    emoji = "🟡" if kind == "SWEEP" else "🟢"
    title = "LIQUIDITY SWEEP" if kind == "SWEEP" else "REVERSAL CONFIRMED"
    timestamp = sig["time"].strftime("%Y-%m-%d %H:%M EST")
    reference = sig["session_start"].strftime("%Y-%m-%d %H:%M EST")
    dashboard = "https://compound0x.github.io/liquidity-sweep-reversal/"
    lines = [
        f"{emoji} {title}", "", f"Asset: {sig['asset']}",
        f"Side: {sig['side']}", f"Reference: {reference}",
        f"Detected: {timestamp}", f"Price: {fmt_price(sig)}",
        f"Detail: {sig['detail']}",
    ]
    if kind == "SWEEP":
        lines.append("Reversal: Not confirmed yet")
    elif sig.get("lag_min") is not None:
        lines.append(f"Minutes after sweep: {int(sig['lag_min'])}")
    lines.extend(["", f"Dashboard: {dashboard}"])
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(scanner.NY)
    reference = current_reference(now)
    if reference is None:
        print("No active reference window; exiting.")
        return

    assets = scanner._selected_assets()
    signals_df, sessions_df = scanner.scan_history(assets, days=scanner.SCAN_DAYS, cfg=scanner.CFG, verbose=False)
    scanner.export_html(signals_df, sessions_df, DASHBOARD_FILE, scanner.CFG, days=scanner.SCAN_DAYS)

    if signals_df.empty:
        print("No signals found.")
        return

    state = load_state()
    cutoff = now - timedelta(days=3)
    state = {k: v for k, v in state.items() if v >= cutoff.isoformat()}
    current = signals_df[
        (signals_df["session_start"] == reference)
        & (signals_df["time"] <= now)
        & (signals_df["kind"].isin(["SWEEP", "REVERSAL"]))
    ].copy()

    if current.empty:
        print(f"No SWEEP/REVERSAL signals for {reference:%Y-%m-%d %H:%M} EST window yet.")
        save_state(state)
        return

    for _, row in current.sort_values("time").iterrows():
        sig = row.to_dict()
        key = signal_key(sig)
        if key in state:
            continue
        message = format_alert(sig)
        print("\n" + message)
        #discord_send(message)
        telegram_send(message)
        state[key] = now.isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
