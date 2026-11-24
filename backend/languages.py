"""Target-language catalog (worldwide release, 2026-11-25).

Languages are explicit choices from this list; nothing is ever auto-detected.
Each entry knows its script (translation guard), its Chirp 3 HD locale/voices
(hosted Standard voice), whether Sarvam can be the fallback, and the Edge
voice used on this PC (local profile only).

A language is *sellable* in the hosted service only when it has
  * a translation benchmark baseline  (quality/reports/baseline_<engine>_<lang>*.json), and
  * a reviewer sign-off              (quality/signoffs.json: {"<lang>": {"by": ..., "date": ...}})
Hindi is the flagship and is signed off by the owner decision of 2026-11-20.
Everything else is "beta": usable locally, shown as coming soon in hosted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SIGNOFFS = HERE / "quality" / "signoffs.json"
REPORTS = HERE / "quality" / "reports"

CHIRP_FEMALE = ("Kore", "Aoede", "Leda", "Zephyr")
CHIRP_MALE = ("Charon", "Puck", "Fenrir", "Orus")


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str
    script: str
    chirp_locale: Optional[str]          # None: no Chirp 3 HD voice for this language
    edge_voice: str
    sarvam: bool = False                 # Sarvam Bulbul supports it (Indian languages)
    rtl: bool = False
    regions: Tuple[str, ...] = field(default_factory=tuple)

    def chirp_voice(self, name: str = "Kore") -> Optional[str]:
        return f"{self.chirp_locale}-Chirp3-HD-{name}" if self.chirp_locale else None


L = Language
LANGUAGES: Dict[str, Language] = {x.code: x for x in (
    L("hi", "Hindi", "हिन्दी", "Devanagari", "hi-IN", "hi-IN-SwaraNeural", True, regions=("IN",)),
    L("bn", "Bengali", "বাংলা", "Bengali", "bn-IN", "bn-IN-TanishaaNeural", True, regions=("IN", "BD")),
    L("ta", "Tamil", "தமிழ்", "Tamil", "ta-IN", "ta-IN-PallaviNeural", True, regions=("IN", "LK", "SG")),
    L("te", "Telugu", "తెలుగు", "Telugu", "te-IN", "te-IN-ShrutiNeural", True, regions=("IN",)),
    L("mr", "Marathi", "मराठी", "Devanagari", "mr-IN", "mr-IN-AarohiNeural", True, regions=("IN",)),
    L("gu", "Gujarati", "ગુજરાતી", "Gujarati", "gu-IN", "gu-IN-DhwaniNeural", True, regions=("IN",)),
    L("kn", "Kannada", "ಕನ್ನಡ", "Kannada", "kn-IN", "kn-IN-SapnaNeural", True, regions=("IN",)),
    L("ml", "Malayalam", "മലയാളം", "Malayalam", "ml-IN", "ml-IN-SobhanaNeural", True, regions=("IN",)),
    L("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", None, "pa-IN-OjasNeural", True, regions=("IN",)),
    L("ur", "Urdu", "اردو", "Arabic", "ur-IN", "ur-PK-UzmaNeural", rtl=True, regions=("PK", "IN")),
    L("es", "Spanish", "Español", "Latin", "es-US", "es-MX-DaliaNeural", regions=("ES", "MX", "US", "AR", "CO")),
    L("pt", "Portuguese", "Português", "Latin", "pt-BR", "pt-BR-FranciscaNeural", regions=("BR", "PT")),
    L("fr", "French", "Français", "Latin", "fr-FR", "fr-FR-DeniseNeural", regions=("FR", "CA", "BE")),
    L("de", "German", "Deutsch", "Latin", "de-DE", "de-DE-KatjaNeural", regions=("DE", "AT", "CH")),
    L("it", "Italian", "Italiano", "Latin", "it-IT", "it-IT-ElsaNeural", regions=("IT",)),
    L("id", "Indonesian", "Bahasa Indonesia", "Latin", "id-ID", "id-ID-GadisNeural", regions=("ID",)),
    L("tr", "Turkish", "Türkçe", "Latin", "tr-TR", "tr-TR-EmelNeural", regions=("TR",)),
    L("vi", "Vietnamese", "Tiếng Việt", "Latin", "vi-VN", "vi-VN-HoaiMyNeural", regions=("VN",)),
    L("pl", "Polish", "Polski", "Latin", "pl-PL", "pl-PL-ZofiaNeural", regions=("PL",)),
    L("nl", "Dutch", "Nederlands", "Latin", "nl-NL", "nl-NL-ColetteNeural", regions=("NL", "BE")),
    L("ru", "Russian", "Русский", "Cyrillic", "ru-RU", "ru-RU-SvetlanaNeural", regions=("RU",)),
    L("uk", "Ukrainian", "Українська", "Cyrillic", "uk-UA", "uk-UA-PolinaNeural", regions=("UA",)),
    L("ar", "Arabic", "العربية", "Arabic", "ar-XA", "ar-SA-ZariyahNeural", rtl=True, regions=("SA", "AE", "EG")),
    L("ja", "Japanese", "日本語", "Japanese", "ja-JP", "ja-JP-NanamiNeural", regions=("JP",)),
    L("ko", "Korean", "한국어", "Hangul", "ko-KR", "ko-KR-SunHiNeural", regions=("KR",)),
    L("zh", "Chinese (Mandarin)", "中文", "Han", "cmn-CN", "zh-CN-XiaoxiaoNeural", regions=("CN", "TW", "SG")),
    L("th", "Thai", "ไทย", "Thai", "th-TH", "th-TH-PremwadeeNeural", regions=("TH",)),
)}

FLAGSHIP = "hi"


def get(code: str) -> Optional[Language]:
    return LANGUAGES.get((code or "").split("-")[0].lower())


def _signoffs() -> Dict[str, Dict]:
    try:
        return json.loads(SIGNOFFS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def has_baseline(code: str) -> bool:
    if code == FLAGSHIP:
        return any(REPORTS.glob("baseline_*.json"))
    return any(REPORTS.glob(f"baseline_*_{code}*.json"))


def status(code: str) -> Dict:
    lang = get(code)
    if lang is None:
        return {"sellable": False, "why": "not in the catalog"}
    missing = []
    if not has_baseline(lang.code):
        missing.append("translation benchmark baseline")
    if lang.code != FLAGSHIP and lang.code not in _signoffs():
        missing.append("native reviewer sign-off")
    if lang.chirp_locale is None and not lang.sarvam:
        missing.append("a commercially cleared voice")
    return {"sellable": not missing, "why": "; ".join(missing)}


def sellable(code: str) -> bool:
    return status(code)["sellable"]


def describe(hosted: bool) -> List[Dict]:
    out = []
    for lang in LANGUAGES.values():
        st = status(lang.code)
        out.append({"code": lang.code, "name": lang.name, "native": lang.native, "script": lang.script,
                    "rtl": lang.rtl, "flagship": lang.code == FLAGSHIP, "voice": lang.chirp_voice(),
                    "fallback": "sarvam" if lang.sarvam else None,
                    "available": st["sellable"] or not hosted, "sellable": st["sellable"], "missing": st["why"]})
    return out


def refusal(code: str, hosted: bool) -> Optional[str]:
    lang = get(code)
    if lang is None:
        return f"'{code}' is not a supported target language"
    if hosted and not sellable(lang.code):
        return f"{lang.name} is not available yet ({status(lang.code)['why']})"
    return None
