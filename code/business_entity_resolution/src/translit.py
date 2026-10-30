"""
Indic scripts -> Latin transliteration, so names written in any Brahmic script
can be compared with their English spellings
("मॉडर्न फाइनेंस" -> "modarn phaainens" ~ "modern finance").

Unicode lays out the Brahmic script blocks (Devanagari, Bengali, Gurmukhi,
Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) in parallel: the same
letter sits at the same offset inside each 0x80-wide block. So every script is
first folded onto Devanagari by its offset, then romanized with one table.

A phonetic, lossy scheme tuned for matching, not a scholarly romanization:
consonants carry an inherent "a" unless followed by a vowel sign or virama,
and the inherent "a" is dropped at the end of a word (schwa deletion).
"""

import re

_DEVA = 0x0900
_INDIC_FIRST, _INDIC_LAST = 0x0980, 0x0D7F

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "zh", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
}
# Precomposed nukta letters U+0958..U+095F
_CONSONANTS.update({chr(0x0958 + i): v for i, v in enumerate(["q", "kh", "g", "z", "r", "rh", "f", "y"])})
# consonant + separate nukta sign
_NUKTA_FORMS = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ऎ": "e", "ए": "e", "ऐ": "ai", "ऒ": "o", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
}
_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "ॆ": "e", "े": "e", "ै": "ai", "ॊ": "o", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
_MODIFIERS = {"ं": "n", "ँ": "n", "ः": "h"}
_VIRAMA = "्"
_NUKTA = "़"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}

# Letters outside the shared layout, romanized directly
_SPECIAL = {
    "ৎ": "t",                                              # Bengali khanda ta
    "ੰ": "n", "ੱ": "",                                     # Gurmukhi tippi, addak
    "ൺ": "n", "ൻ": "n", "ർ": "r", "ൽ": "l", "ൾ": "l", "ൿ": "k",  # Malayalam chillus
}

INDIC = re.compile(r"[ऀ-ൿ]")


def _fold_to_devanagari(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if _INDIC_FIRST <= cp <= _INDIC_LAST and ch not in _SPECIAL:
            out.append(chr(_DEVA + (cp & 0x7F)))
        else:
            out.append(ch)
    return "".join(out)


def indic_to_latin(text: str) -> str:
    if not text or not INDIC.search(text):
        return text
    text = _fold_to_devanagari(text)
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _CONSONANTS:
            base = _CONSONANTS[ch]
            i += 1
            if i < n and text[i] == _NUKTA:
                base = _NUKTA_FORMS.get(ch, base)
                i += 1
            nxt = text[i] if i < n else ""
            if nxt in _MATRAS:
                out.append(base + _MATRAS[nxt])
                i += 1
            elif nxt == _VIRAMA:
                out.append(base)
                i += 1
            elif nxt and (nxt in _CONSONANTS or nxt in _MODIFIERS):
                out.append(base + "a")
            else:
                out.append(base)  # word-final: schwa deletion
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
            i += 1
        elif ch in _MODIFIERS:
            out.append(_MODIFIERS[ch])
            i += 1
        elif ch in _DIGITS:
            out.append(_DIGITS[ch])
            i += 1
        elif ch in _SPECIAL:
            out.append(_SPECIAL[ch])
            i += 1
        elif ch in (_VIRAMA, _NUKTA) or ch in _MATRAS or "ऀ" <= ch <= "ॿ":
            i += 1  # stray combining mark or unmapped Indic sign
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# Backwards-compatible name
devanagari_to_latin = indic_to_latin


if __name__ == "__main__":
    cases = [
        ("मॉडर्न फाइनेंस", "modarn phaainens"), ("राज ट्रेडर्स", "raaj tredars"),
        ("फ़ैशन", "faishan"), ("१२ गली", "12 galee"), ("Plain ASCII", "Plain ASCII"),
        ("হরি ফাউন্ডেশন প্রাইভেট লিমিটেড", "hari phaaundeshan praaibhet limited"),   # Bengali
        ("ગુજરાત", "gujaraat"),                                                  # Gujarati
        ("சென்னை", "chennai"),                                                   # Tamil
        ("హైదరాబాద్", "haidaraabaad"),                                           # Telugu
        ("ಬೆಂಗಳೂರು", "bengalooru"),                                              # Kannada
    ]
    for s, want in cases:
        got = indic_to_latin(s)
        print(f"{s} -> {got}")
        assert got == want, (got, want)
    print("ok")
