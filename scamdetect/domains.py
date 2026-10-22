"""Official finance domains (allowlist) and brand-impersonation detection.

A host is trusted if it equals an official domain or is a subdomain of one
(kite.zerodha.com). A host that uses a brand name but is NOT the official
domain (zerodha-profit.in, nse-tips-vip.online) is flagged as impersonation.
Extend these lists for your region.
"""
import re

TRUSTED_DOMAINS = {
    # brokers / investment apps
    "zerodha.com", "groww.in", "upstox.com", "angelone.in", "icicidirect.com", "hdfcsec.com",
    "kotaksecurities.com", "motilaloswal.com", "sharekhan.com", "5paisa.com", "dhan.co", "paytmmoney.com",
    # crypto exchanges
    "coinswitch.co", "wazirx.com", "coindcx.com", "binance.com", "coinbase.com", "kraken.com",
    # exchanges, regulators, depositories, government
    "nseindia.com", "bseindia.com", "sebi.gov.in", "rbi.org.in", "amfiindia.com", "nsdl.co.in",
    "cdslindia.com", "npci.org.in", "incometax.gov.in", "cybercrime.gov.in", "mygov.in", "india.gov.in",
    # banks / insurance
    "hdfcbank.com", "icicibank.com", "sbi.co.in", "onlinesbi.sbi", "axisbank.com", "kotak.com",
    "pnbindia.in", "bankofbaroda.in", "yesbank.in", "indusind.com", "idfcfirstbank.com", "licindia.in",
    # mutual funds / registrars
    "hdfcfund.com", "icicipruamc.com", "sbimf.com", "nipponindiamf.com", "axismf.com", "kotakmf.com",
    "mfcentral.com", "camsonline.com", "kfintech.com",
    # financial news
    "moneycontrol.com", "livemint.com", "business-standard.com", "indiatimes.com",
    "valueresearchonline.com", "morningstar.in",
}

BRANDS = {
    "zerodha", "groww", "upstox", "angelone", "icici", "icicidirect", "hdfc", "kotak", "sbi",
    "axisbank", "sebi", "rbi", "nse", "bse", "nseindia", "bseindia", "paytm", "binance", "coinbase", "wazirx",
    "coindcx", "coinswitch", "motilal", "sharekhan", "licindia", "amfi", "nsdl", "cdsl", "npci",
    "5paisa", "moneycontrol",
}


def is_trusted(host):
    host = (host or "").lower()
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def impersonated_brand(host):
    """Brand name used in a host that is not that brand's official domain, else ''."""
    host = (host or "").lower()
    if not host or is_trusted(host):
        return ""
    tokens = set(re.split(r"[.\-_]", host))
    for b in BRANDS:
        if b in tokens or (len(b) >= 5 and b in host):
            return b
    return ""
