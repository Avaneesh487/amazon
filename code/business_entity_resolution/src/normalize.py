"""Text normalisation for business names and addresses.

Everything here is language/country agnostic rule-based cleaning plus a
generic Indic-script -> Latin transliterator. Country is never used to pick a
code path, so unseen countries (e.g. France in the test set) get the same
treatment as the training countries.

The heavy lifting is done with polars string expressions (Rust, multi-threaded);
only records that contain Indic script go through Python.
"""
import re
import unicodedata

import polars as pl

# --------------------------------------------------------------------------
# Indic transliteration
# --------------------------------------------------------------------------
# All major Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
# Tamil, Telugu, Kannada, Malayalam) share the ISCII-derived 128-codepoint
# layout, so any of them can be folded onto Devanagari by offset and then
# transliterated with a single table.
INDIC_LO, INDIC_HI = 0x0900, 0x0D7F
INDIC_RE = re.compile("[ऀ-ൿ]")

_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e",
    "ऒ": "o", "ॠ": "ri", "ऌ": "l",
}
_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
    "ॄ": "ri",
}
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
    "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r",
    "ढ़": "rh", "फ़": "f", "य़": "y",
}
_VIRAMA, _NUKTA = "्", "़"
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}


def _fold_to_devanagari(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if INDIC_LO <= cp <= INDIC_HI:
            cp = 0x0900 + (cp & 0x7F)
        out.append(chr(cp))
    return "".join(out)


def transliterate_indic(text):
    """Rough phonetic Latin rendering of any Indic-script string."""
    s = unicodedata.normalize("NFC", _fold_to_devanagari(text))
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in _CONS:
            out.append(_CONS[ch])
            j = i + 1
            if j < n and s[j] == _NUKTA:
                j += 1
            if j < n and s[j] in _MATRAS:
                out.append(_MATRAS[s[j]])
                j += 1
            elif j < n and s[j] == _VIRAMA:
                j += 1
            else:
                # inherent schwa, dropped at the end of a word
                if j < n and (s[j] in _CONS or s[j] in _SIGNS):
                    out.append("a")
            i = j
            continue
        if ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif ch in _MATRAS or ch in (_VIRAMA, _NUKTA):
            pass
        elif 0x0900 <= ord(ch) <= 0x097F:
            pass  # other Devanagari signs (danda, etc.)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic_key(tok):
    """Collapse a Latin token to a coarse phonetic skeleton.

    Used so that a rule transliteration ("prodyusar") can meet its English
    source ("producer"): first letter + consonant classes, vowels dropped.
    """
    if not tok:
        return ""
    t = tok.lower()
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"),
                 ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "")):
        t = t.replace(a, b)
    t = re.sub(r"c(?=[eiy])", "s", t)
    t = t.replace("c", "k")
    head, rest = t[:1], re.sub(r"[aeiouh]", "", t[1:])
    return head + re.sub(r"(.)\1+", r"\1", rest)


# --------------------------------------------------------------------------
# Vocabularies (generic English/French/Indian business-text conventions)
# --------------------------------------------------------------------------
LEGAL_CANON = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd", "private": "pvt",
    "company": "co", "cos": "co", "ltda": "ltd", "pvtltd": "pvt ltd",
}
# tokens that carry (almost) no identity; removed from the "core" name
NAME_STOP = {
    "llc", "inc", "corp", "ltd", "pvt", "co", "llp", "pllc", "plc", "lp", "pc", "pa",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "ei", "snc", "scop", "gmbh", "ag",
    "the", "and", "of", "et", "de", "la", "le", "les", "du", "des", "l", "d", "a",
    "dba", "aka", "www", "com", "in", "net", "org", "fr", "public", "opc",
}
ADDR_CANON = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd",
    "drive": "dr", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "bvd": "blvd",
    "court": "ct", "place": "pl", "circle": "cir", "highway": "hwy",
    "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt",
    "suite": "ste", "floor": "fl", "building": "bldg", "mount": "mt", "fort": "ft",
    "saint": "st", "sainte": "ste", "rue": "r", "allee": "all", "chemin": "ch",
    "impasse": "imp", "route": "rte", "faubourg": "fbg", "cours": "crs",
    "nagar": "ngr", "colony": "col", "sector": "sec", "near": "nr", "opposite": "opp",
    "opp": "opp", "marg": "mg", "sadak": "rd", "chowk": "chk", "centre": "center",
}
ADDR_STOP = {
    "no", "house", "h", "hno", "door", "plot", "flat", "shop", "unit", "apt", "ste",
    "fl", "bldg", "null", "none", "nan", "n", "a", "po", "box", "bp", "the", "of",
    "and", "de", "la", "le", "les", "du", "des", "d", "l", "et", "city", "dist",
    "district", "gr", "ground", "first", "nr", "opp", "bis", "ter", "eme", "th",
    "nd", "rd_", "st_", "cedex",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "nct of delhi": "dl", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "jammu and kashmir": "jk", "jammu kashmir": "jk",
}


def _phrase_regex(mapping):
    keys = sorted(mapping, key=len, reverse=True)
    return r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"


# --------------------------------------------------------------------------
# polars expressions
# --------------------------------------------------------------------------
def _ascii_fold(expr):
    """lowercase + strip diacritics (é -> e, ç -> c)."""
    return (expr.str.normalize("NFKD")
                .str.replace_all(r"\p{Mn}", "")
                .str.to_lowercase()
                .str.replace_all("ß", "ss").str.replace_all("æ", "ae")
                .str.replace_all("œ", "oe").str.replace_all("ø", "o"))


def name_expr(col):
    e = _ascii_fold(pl.col(col).fill_null(""))
    e = (e.str.replace_all(r"https?://", " ")
          .str.replace_all(r"\bwww\.", " ")
          # domains: keep the label, drop the TLD  (jodiespub.com -> jodiespub)
          .str.replace_all(r"\.(com|co\.in|in|net|org|fr|co|biz|info|us|io)\b", " ")
          .str.replace_all(r"\+?\d[\d\- ]{7,}\d", " ")          # phone numbers
          .str.replace_all("&", " and ")
          .str.replace_all(r"(\w)\.(\w)\.?", "$1$2")             # l.l.c. -> llc
          .str.replace_all(r"(\w)\.(\w)\.?", "$1$2")
          .str.replace_all(r"['`´’]", "")
          .str.replace_all(r"[^\p{L}\p{N}]+", " ")
          .str.strip_chars())
    return e


def addr_expr(col):
    e = _ascii_fold(pl.col(col).fill_null(""))
    e = (e.str.replace_all("[ऀ-ൿ]+", " ")              # local-script state names
          .str.replace_all(r"<?\bnull\b>?", " ")
          .str.replace_all(r"\b(p\.?\s?o\.?\s?box|b\.?p\.?)\s*\d+", " ")  # PO boxes
          .str.replace_all(r"['`´’]", "")
          .str.replace_all(r"(\d)(bis|ter|eme|er|st|nd|rd|th)\b", "$1 $2")
          .str.replace_all(r"(\d)([a-z])\b", "$1 $2")          # 2031c -> 2031 c
          .str.replace_all(r"([a-z])(\d)", "$1 $2")
          .str.replace_all(r"[^\p{L}\p{N}]+", " ")
          .str.strip_chars())
    return e


def canon_tokens(expr, mapping):
    """split into tokens and map each through `mapping`."""
    return expr.str.split(" ").list.eval(
        pl.element().replace(mapping).filter(pl.element() != ""))


def strip_leading_zeros(expr):
    return expr.list.eval(pl.element().str.replace(r"^0+(\d)", "$1"))


STATE_ALL = {**US_STATES, **IN_STATES}
STATE_RE = _phrase_regex(STATE_ALL)
STATE_CODES = set(STATE_ALL.values())
