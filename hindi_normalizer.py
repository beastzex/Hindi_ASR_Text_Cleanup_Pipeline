"""
hindi_normalizer.py

Implements a rule-based approach to convert Hindi number words into digits.
Uses a dictionary mapping of Hindi number words and handles compound numbers
and idiom exclusions securely without needing machine learning.
"""
import re

class HindiNumberNormalizer:
    """
    Converts Hindi number words to digits in transcribed text.
    
    Why we need this:
    When Whisper transcribes Hindi speech, numbers come out as 
    words (दो सौ पचास) instead of digits (250). This makes the 
    text unusable for downstream tasks.
    
    Approach chosen:
    Rule-based conversion using a comprehensive word-to-value 
    mapping with context-aware idiom detection.
    """
    def __init__(self):
        # build mapping
        self.units = {
            "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, 
            "छह": 6, "छै": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
            "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
            "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20,
            "इक्कीस": 21, "बाइस": 22, "तेइस": 23, "चौबीस": 24, "पच्चीस": 25,
            "छब्बीस": 26, "सत्ताइस": 27, "अठ्ठाइस": 28, "उनतीस": 29, "तीस": 30,
            "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
            "छत्तीज़": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
            "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45,
            "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "पचास": 50,
            "इक्यावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54, "पचपन": 55,
            "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
            "इकसठ": 61, "बासठ": 62, "तिरेसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
            "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70,
            "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
            "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
            "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84, "पचासी": 85,
            "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90,
            "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
            "छियानवे": 96, "संतानवे": 97, "अट्ठानवे": 98, "निन्नानवे": 99
        }
        self.multipliers = {
            "सौ": 100, "हजार": 1000, "हज़ार": 1000, "लाख": 100000, "करोड़": 10000000, "करोड़": 10000000
        }
        self.idioms = [
            r"दो-चार", r"चार चाँद", r"नौ दो ग्यारह", r"दो टूक", r"तीन तिकड़म", 
            r"सात खून माफ", r"पाँच उँगलियाँ", r"एक न एक", r"एक दो", r"सौ बात की एक बात"
        ]
        
    def normalize(self, text: str) -> tuple[str, list[dict]]:
        """
        Parses the text to substitute numeric words with corresponding actual digits.
        Avoids specific idioms to ensure conversational intent is preserved.
        """
        changes = []
        words = text.split()
        normalized_words = []
        i = 0
        
        while i < len(words):
            word = words[i].strip('.,!?”“')
            punct_end = words[i][len(word):] if words[i].startswith(word) else ""
            
            # Check for hyphenated idioms explicitly in current or combined tokens
            idiom_found = False
            for idiom in self.idioms:
                if re.search(idiom, " ".join(words[i:i+4])): # look ahead 4 words
                    # If idiom is spanning current word, we skip converting it
                    if word in idiom:
                        idiom_found = True
                        break
            
            if "-" in word and word.split('-')[0] in self.units and word.split('-')[1] in self.units:
                idiom_found = True # e.g. "दो-चार"
                
            if idiom_found:
                normalized_words.append(words[i])
                i += 1
                changes.append({
                    "original": word,
                    "converted": word,
                    "position": i,
                    "confidence": "HIGH"
                })
                continue

            # Parsing compound numbers greedily
            val = 0
            temp_val = 0
            orig_tokens = []
            
            while i < len(words):
                w = words[i].strip('.,!?”“')
                if w in self.units:
                    temp_val += self.units[w]
                    orig_tokens.append(words[i])
                    i += 1
                elif w in self.multipliers:
                    if temp_val == 0:
                        temp_val = 1
                    val += temp_val * self.multipliers[w]
                    temp_val = 0
                    orig_tokens.append(words[i])
                    i += 1
                else:
                    break
                    
            if orig_tokens:
                total = val + temp_val
                if len(orig_tokens) == 1 and temp_val < 10 and w not in self.multipliers:
                    # just a single small unit, likely to be a number but could be conversational
                    # We will convert it
                    pass
                
                # check if there's trailing punctuation on the last word captured
                last_token = orig_tokens[-1]
                core_last = last_token.strip('.,!?”“')
                punct = last_token[len(core_last):]
                
                converted_str = str(total) + punct
                normalized_words.append(converted_str)
                orig_str = " ".join(orig_tokens)
                
                if total > 0:
                    changes.append({
                        "original": orig_str,
                        "converted": str(total),
                        "position": i,
                        "confidence": "HIGH"
                    })
            else:
                normalized_words.append(words[i])
                i += 1
                
        return " ".join(normalized_words), changes

if __name__ == "__main__":
    normalizer = HindiNumberNormalizer()
    samples = ["मुझे तीन सौ रुपये दो", "दो-चार बातें करनी हैं", "उसने पच्चीस हजार दिए"]
    for s in samples:
        res, chg = normalizer.normalize(s)
        print(f"Sample: {s}\nResult: {res}\nChanges: {chg}\n")
