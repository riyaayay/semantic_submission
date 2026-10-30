"""
Text and Address Normalization for Business Entity Resolution.
Supports multilingual and multi-country data (US, India, France).
"""

import re
import unicodedata
from typing import Dict, Tuple, Optional, List, Set

# Legal suffix dictionary mapping variants to canonical tokens or empty
LEGAL_SUFFIXES = {
    # US / UK
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp",
    "limited": "ltd", "ltd": "ltd",
    "company": "co", "co": "co",
    "llc": "llc", "l.l.c.": "llc",
    "llp": "llp", "l.l.p.": "llp",
    "pllc": "llc",
    # India
    "private limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "pvt. ltd.": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt": "pvt", "private": "pvt",
    # France
    "sarl": "sarl", "s.a.r.l.": "sarl",
    "sas": "sas", "s.a.s.": "sas",
    "sasu": "sas", "s.a.s.u.": "sas",
    "sa": "sa", "s.a.": "sa",
    "eurl": "eurl", "snc": "snc", "sci": "sci",
}

# Regex to match legal suffixes at word boundaries (especially at the end of names)
_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(LEGAL_SUFFIXES.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

# Common street type abbreviations
STREET_MAPPINGS = {
    "street": "st", "saint": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "road": "rd",
    "drive": "dr",
    "lane": "ln",
    "terrace": "ter",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "apartment": "apt", "unit": "apt", "suite": "ste",
    # French street words
    "bd": "blvd", "av": "ave", "bvd": "blvd",
}

_STREET_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(STREET_MAPPINGS.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE
)

# Postal code regexes
PIN_INDIA_REGEX = re.compile(r"\b([1-9]\d{5})\b")
ZIP_US_REGEX = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
POSTCODE_FRANCE_REGEX = re.compile(r"\b(\d{5})\b")

# Numeric token regex (house numbers, suite numbers)
NUMBER_REGEX = re.compile(r"\b\d+[a-zA-Z]?\b")


def strip_accents(text: str) -> str:
    """Normalize unicode characters and remove diacritics / accents."""
    if not text:
        return ""
    # NFKD decomposes characters with accents into base character + accent mark
    nfkd = unicodedata.normalize("NFKD", str(text))
    # Keep only non-combining characters (strip diacritical marks)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_business_name(name: str) -> Tuple[str, str, str]:
    """Normalizes a business name string.
    
    Returns:
        (clean_name, name_without_suffix, legal_suffix)
    """
    if not name or str(name).lower() == "nan":
        return "", "", ""

    # Convert to ASCII / strip accents
    s = strip_accents(str(name)).lower()

    # Handle domain names (e.g. maurewilliamscolombier.com -> maure williams colombier)
    s = re.sub(r"\.(com|net|org|co|in|fr|io|biz|info)\b", "", s)

    # Replace '&' with 'and'
    s = s.replace("&", " and ")

    # Remove special punctuation but keep alphanumeric and whitespace
    s = re.sub(r"[^\w\s]", " ", s)

    # Extract legal suffix if present
    found_suffix = ""
    match = _LEGAL_SUFFIX_PATTERN.search(s)
    if match:
        raw_suffix = match.group(0).lower()
        found_suffix = LEGAL_SUFFIXES.get(raw_suffix, "")
        # Remove suffix for the suffix-stripped version
        name_no_suffix = _LEGAL_SUFFIX_PATTERN.sub(" ", s)
        name_no_suffix = " ".join(name_no_suffix.split())
    else:
        name_no_suffix = " ".join(s.split())

    clean_name = " ".join(s.split())
    return clean_name, name_no_suffix, found_suffix


def extract_postal_code(address: str, country: str) -> str:
    """Extracts postal code / PIN based on country."""
    if not address or str(address).lower() == "nan":
        return ""

    c = str(country).strip().upper()
    if c == "INDIA":
        m = PIN_INDIA_REGEX.search(address)
        return m.group(1) if m else ""
    elif c == "US":
        m = ZIP_US_REGEX.search(address)
        return m.group(1) if m else ""
    elif c == "FRANCE":
        m = POSTCODE_FRANCE_REGEX.search(address)
        return m.group(1) if m else ""
    else:
        # Generic 5 or 6 digit code fallback
        m = re.search(r"\b(\d{5,6})\b", address)
        return m.group(1) if m else ""


def extract_numeric_tokens(address: str) -> List[str]:
    """Extracts house numbers, street numbers, plot numbers."""
    if not address or str(address).lower() == "nan":
        return []
    s = strip_accents(str(address)).lower()
    return NUMBER_REGEX.findall(s)


def normalize_address(address: str, country: str) -> Tuple[str, str, List[str]]:
    """Normalizes address string.
    
    Returns:
        (clean_address, postal_code, numeric_tokens)
    """
    if not address or str(address).lower() == "nan":
        return "", "", []

    raw = str(address)
    postal_code = extract_postal_code(raw, country)
    numeric_tokens = extract_numeric_tokens(raw)

    s = strip_accents(raw).lower()
    s = s.replace("&", " and ")

    # Standardize street abbreviations
    s = _STREET_PATTERN.sub(lambda m: STREET_MAPPINGS.get(m.group(0).lower(), m.group(0)), s)

    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    clean_addr = " ".join(s.split())

    return clean_addr, postal_code, numeric_tokens


if __name__ == "__main__":
    # Test cases
    test_cases_name = [
        ("Payne Énterprises LLC", ("payne enterprises llc", "payne enterprises", "llc")),
        ("Lumay Bóral Inc.", ("lumay boral inc", "lumay boral", "inc")),
        ("maurewilliamscolombier.com", ("maurewilliamscolombier", "maurewilliamscolombier", "")),
        ("Raj Investments LLP", ("raj investments llp", "raj investments", "llp")),
    ]
    for raw, expected in test_cases_name:
        res = normalize_business_name(raw)
        print(f"Name: {raw} -> {res}")
        assert res[1] == expected[1], f"Expected {expected[1]}, got {res[1]}"

    test_cases_addr = [
        ("85 Wayne Avenue, Ticonderoga, NY 12883", "US"),
        ("6(29), C.I.T. Colony, Chennai, 600004", "India"),
        ("175 Boulevard du Président Franklin Roosevelt, Bordeaux 33000", "France"),
    ]
    for raw, country in test_cases_addr:
        res = normalize_address(raw, country)
        print(f"Address ({country}): {raw} -> Clean: '{res[0]}', Postcode: '{res[1]}', Numbers: {res[2]}")
    print("Normalizer tests passed!")
