import re
import unicodedata
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DatasetCleaner:
    """
    Cleans raw SMS datasets by removing obfuscation tactics.
    Crucially, it preserves the ML signal by appending explicit tags 
    (e.g., <OBF_HOMOGLYPH>) when an obfuscation tactic is detected.

    The output keeps the normalized text but preserves evidence of evasion
    so models can learn that obfuscation patterns occurred.
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
        
        # Regex for spacing obfuscation (e.g. C O R R E O S)
        self.spacing_re = re.compile(r'\b(?:[a-zA-Z]\s){3,}[a-zA-Z]\b')
        
        # Regex for punctuation obfuscation (e.g. P.A.Q.U.E.T.E or c_o_r_r_e_o_s)
        self.punct_obf_re = re.compile(r'\b(?:[a-zA-Z][.\-_]){3,}[a-zA-Z]\b')
        
        # Fast homoglyph lookup regex
        self.homoglyph_re = re.compile('|'.join(self.HOMOGLYPH_MAP.keys()))
        
        self.stats = {
            "total_processed": 0,
            "obf_zwsp": 0,
            "obf_homoglyph": 0,
            "obf_newlines": 0,
            "obf_font": 0,
            "obf_spacing": 0,
            "obf_punct": 0
        }

    def is_valid_sms(self, text: str) -> bool:
        """
        Heuristic to detect if a text is an email, newsletter, or article instead of an SMS.
        """
        if not text or not isinstance(text, str):
            return False
            
        # 1. Extremely long text is unlikely to be an SMS (even concatenated)
        if len(text) > 1600:
            return False
            
        text_lower = text.lower()
        
        # 2. Email-specific phrases
        email_indicators = [
            "this email was sent",
            "click here to unsubscribe",
            "view email in browser",
            "forwarded message",
            "to view this email as a web page",
            "copyright ©",
            "all rights reserved"
        ]
        
        if any(indicator in text_lower for indicator in email_indicators):
            return False
            
        return True

    def clean(self, text: str) -> Dict[str, Any]:
        """
        Clean a single message and return normalized text plus tags.

        The returned dict contains:
        - original: the original input text
        - cleaned: normalized text with optional obfuscation tags appended
        - tags: list of tags that describe detected obfuscation patterns
        """
        if not isinstance(text, str):
            text = str(text)
            
        self.stats["total_processed"] += 1
        tags = []
        original = text

        # Normalize escaped whitespace sequences from CSV exports.
        if "\\n" in text or "\\t" in text or "\\r" in text:
            text = text.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")

        # 0. Standardize pre-anonymized tokens from academic datasets to prevent ML data leakage.
        # Different datasets use different tokens (e.g. <URL> vs [URL] vs http://url.com)
        text = re.sub(r'(?i)<URL>|\[URL\]|http://url\.com|https://url\.com|http://link\.com|https://link\.com|<LINK>|\[LINK\]', '<URL>', text)
        text = re.sub(r'(?i)<PHONE>|\[PHONE\]|<PHONE_NUMBER>|\[PHONE_NUMBER\]|\b0000000000\b', '<PHONE>', text)
        text = re.sub(r'(?i)<EMAIL>|\[EMAIL\]', '<EMAIL>', text)
        text = re.sub(r'(?i)<USER>|\[USER\]|<USERNAME>|\[USERNAME\]|<NAME>|\[NAME\]|<NAMED_ENTITY>|\[NAMED_ENTITY\]', '<PERSON>', text)
        text = re.sub(r'(?i)<CARD>|\[CARD\]|<CREDIT_CARD>|\[CREDIT_CARD\]', '<CREDIT_CARD>', text)
        text = re.sub(r'(?i)<ACCOUNT>|\[ACCOUNT\]|<ACC>|\[ACC\]', '<ACCOUNT_NUMBER>', text)
        # Normalize date/time placeholder tokens used by Agarwal and similar datasets
        text = re.sub(r'(?i)<DATE_TIME>|\[DATE_TIME\]|<DATETIME>|\[DATETIME\]', '<DATE>', text)

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

        # 3. Detect spacing obfuscation (C O R R E O S)
        if self.spacing_re.search(text):
            tags.append("<OBF_SPACING>")
            self.stats["obf_spacing"] += 1
            # Remove spaces from the matched spaced-out words
            text = self.spacing_re.sub(lambda m: m.group(0).replace(' ', ''), text)
            
        # 4. Detect punctuation obfuscation (P.A.Q.U.E.T.E)
        if self.punct_obf_re.search(text):
            tags.append("<OBF_PUNCTUATION>")
            self.stats["obf_punct"] += 1
            # Remove punctuation from the matched obfuscated words
            text = self.punct_obf_re.sub(lambda m: re.sub(r'[.\-_]', '', m.group(0)), text)

        # 5. Detect excessive newlines (often used to hide phishing links out of view)
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

        # Collapse tabs that appear without newlines.
        if "\t" in text:
            text = re.sub(r"\t+", " ", text).strip()

        # 6. Unicode Normalization (NFKC)
        # Fixes full-width chars (ｅ -> e) or math fonts (𝕮 -> C)
        normalized_text = unicodedata.normalize("NFKC", text)
        if normalized_text != text:
            # NFKC changed something, might be font obfuscation
            tags.append("<OBF_FONT>")
            self.stats["obf_font"] += 1
            text = normalized_text

        # We no longer append tags to the text string. 
        # Deep Learning models benefit from having raw text and tags separate.
        return {
            "text_raw": original,
            "text_normalized": text,
            "tags": tags
        }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    cleaner = DatasetCleaner()
    
    sample_texts = [
        # ZWSP Obfuscation
        "P\u200Bo\u200Bs\u200Bt\u200Ba\u200Bl: Your package is held.",
        # Cyrillic Homoglyphs (pоstаl using Cyrillic o and a)
        "pоstаl: Your package is held.",
        # Spacing Obfuscation
        "Your package from A M A Z O N has arrived.",
        # Punctuation Obfuscation
        "Update your P.A.Y.P.A.L account now.",
        # Font Obfuscation
        "𝕻𝖔𝖘𝖙𝖆𝖑: Your package is held.",
        # Newline Obfuscation (pushing the link down)
        "You have an important notice.\n\n\n\n\n\n\n\n\n\nClick here: http://phish.com",
        # Clean standard message
        "Your package has been successfully delivered to your mailbox."
    ]
    
    for txt in sample_texts:
        result = cleaner.clean(txt)
        print(f"Raw:        {repr(result['text_raw'])}")
        print(f"Normalized: {repr(result['text_normalized'])}")
        print(f"Tags:       {result['tags']}")
        print("-" * 80)
