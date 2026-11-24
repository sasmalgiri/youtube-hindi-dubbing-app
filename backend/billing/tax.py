"""Tax rule per buyer country (worldwide release, 2026-11-25).

India: prices include 18% GST (shown on the receipt).
Elsewhere: Indian GST is not charged on an export of services (zero-rated,
supplied under a Letter of Undertaking). Many countries tax digital services
sold to consumers (EU VAT via OSS, UK VAT, Australian GST, ...) above their
thresholds; this service does not collect those yet. Every non-India rule is
marked ``needs_accountant`` until the owner's accountant confirms it
(tracker M09/L06); the receipt states plainly that no tax was charged.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

GST = (18, 100)

# Countries whose consumer digital-services tax regime we know of; listed so
# the admin view can show where registration may be needed. Not collected.
DIGITAL_TAX_REGIMES = {
    **{c: "EU VAT (OSS)" for c in ("AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
                                     "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE")},
    "GB": "UK VAT", "AU": "Australian GST", "NZ": "NZ GST", "CA": "Canadian GST/HST", "SG": "Singapore GST",
    "JP": "Japanese consumption tax", "KR": "Korean VAT", "NO": "Norwegian VAT", "CH": "Swiss VAT",
    "SA": "Saudi VAT", "AE": "UAE VAT", "ZA": "South African VAT", "MX": "Mexican VAT", "TR": "Turkish VAT",
    "ID": "Indonesian VAT", "MY": "Malaysian service tax", "TH": "Thai VAT", "BD": "Bangladesh VAT",
}


def rule(country: Optional[str]) -> Dict:
    cc = (country or "IN").upper()
    if cc == "IN":
        return {"country": "IN", "label": "GST 18% (included)", "rate": GST, "inclusive": True,
                "needs_accountant": False, "note": ""}
    regime = DIGITAL_TAX_REGIMES.get(cc)
    return {"country": cc, "label": "No GST: export of services", "rate": None, "inclusive": True,
            "needs_accountant": True,
            "note": ("Indian GST is not charged on services exported outside India."
                     + (f" {regime} is not collected by this service." if regime else ""))}


def tax_minor(total_minor: int, r: Dict) -> int:
    rate: Optional[Tuple[int, int]] = r.get("rate")
    if not rate:
        return 0
    num, den = rate
    return (total_minor * num * 2 + (den + num)) // (2 * (den + num))
