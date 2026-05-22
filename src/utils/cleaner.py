import re
import unicodedata
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class DatasetCleaner:
    """
    Cleans raw SMS datasets by removing obfuscation tactics.
    Crucially, it preserves the ML signal by appending explicit tags 
    (e.g., <OBF_HOMOGLYPH>) when an obfuscation tactic is detected.
    """
    
    # Common Cyrillic and Greek homoglyphs used in phishing mapping to Latin
    HOMOGLYPH_MAP = {
        # Cyrillic
        'а': 'a', 'А': 'A',
        'о': 'o', 'О': 'O',
        'е': 'e', 'Е': 'E',
        'р': 'p', 'Р': 'P',
        'с': 'c', 'С': 'C',
        'х': 'x', 'Х': 'X',
        'у': 'y', 'У': 'Y',
        'і': 'i', 'І': 'I',
        'ј': 'j', 'Ј': 'J',
        'ѕ': 's', 'Ѕ': 'S',
        # Greek
        'ο': 'o', 'Ο': 'O',
        'α': 'a', 'Α': 'A',
        'ε': 'e', 'Ε': 'E',
        'ν': 'v', 'Ν': 'N'
    }
    
    def __init__(self):
        # Regex to detect invisible formatting characters
        # \u200B-\u200F: Zero-width spaces and formatting marks
        # \u202A-\u202E: Bi-directional text overrides
        # \uFEFF: Byte order mark
        # \u00AD: Soft hyphen
        self.invisible_chars_re = re.compile(r'[\u200B-\u200F\u202A-\u202E\uFEFF\u00AD]')
        
        # Regex for consecutive newlines
        self.newlines_re = re.compile(r'[\r\n]+')
        
        # Fast homoglyph lookup regex
        self.homoglyph_re = re.compile('|'.join(self.HOMOGLYPH_MAP.keys()))
        
        self.stats = {
            "total_processed": 0,
            "obf_zwsp": 0,
            "obf_homoglyph": 0,
            "obf_newlines": 0,
            "obf_font": 0
        }

    def clean(self, text: str) -> Dict[str, Any]:
        """
        Cleans the text and returns a dictionary with the cleaned string and detected tags.
        """
        if not isinstance(text, str):
            text = str(text)
            
        self.stats["total_processed"] += 1
        tags = []
        original = text

        # 1. Detect and remove invisible characters (ZWSP)
        if self.invisible_chars_re.search(text):
            tags.append("<OBF_ZWSP>")
            self.stats["obf_zwsp"] += 1
            text = self.invisible_chars_re.sub('', text)

        # 2. Detect and replace homoglyphs
        if self.homoglyph_re.search(text):
            tags.append("<OBF_HOMOGLYPH>")
            self.stats["obf_homoglyph"] += 1
            # Translate homoglyphs to latin
            trans_table = str.maketrans(self.HOMOGLYPH_MAP)
            text = text.translate(trans_table)

        # 3. Detect excessive newlines (often used to hide phishing links out of view)
        # We consider > 1 consecutive newline as an obfuscation attempt for SMS,
        # but any newline should be collapsed for dataset cleanliness.
        newline_matches = self.newlines_re.findall(text)
        if newline_matches:
            # If any block of newlines is large, flag it
            if any(len(match) >= 2 for match in newline_matches):
                tags.append("<OBF_NEWLINE>")
                self.stats["obf_newlines"] += 1
            # Collapse all newlines and tabs to a single space
            text = re.sub(r'[\r\n\t]+', ' ', text).strip()

        # 4. Unicode Normalization (NFKC)
        # Fixes full-width chars (ｅ -> e) or math fonts (𝕮 -> C)
        normalized_text = unicodedata.normalize("NFKC", text)
        if normalized_text != text:
            # NFKC changed something, might be font obfuscation
            tags.append("<OBF_FONT>")
            self.stats["obf_font"] += 1
            text = normalized_text

        # Combine text and tags
        # Append tags to the end of the message so the ML model can use them as features
        final_text = text
        if tags:
            final_text = f"{text} {' '.join(tags)}"

        return {
            "original": original,
            "cleaned": final_text,
            "tags": tags
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    cleaner = DatasetCleaner()
    
    sample_texts = [
        # ZWSP Obfuscation
        "C\u200Bo\u200Br\u200Br\u200Be\u200Bo\u200Bs: Su paquete esta retenido.",
        # Cyrillic Homoglyphs (cоrrеоs using Cyrillic o and e)
        "cоrrеоs: Su paquete esta retenido.",
        # Font Obfuscation
        "𝕮𝖔𝖗𝖗𝖊𝖔𝖘: Su paquete esta retenido.",
        # Newline Obfuscation (pushing the link down)
        "Usted tiene un aviso importante.\n\n\n\n\n\n\n\n\n\nPulse aqui: http://phish.com",
        # Clean standard message
        "Tu paquete ha sido entregado correctamente en tu buzon."
    ]
    
    for txt in sample_texts:
        result = cleaner.clean(txt)
        print(f"Original: {repr(result['original'])}")
        print(f"Cleaned:  {repr(result['cleaned'])}")
        print(f"Tags:     {result['tags']}")
        print("-" * 80)
