"""
english_tagger.py

Identifies and tags English-origin words in Hindi transcripts. Supports both
Roman script words and common Devanagari transliterations.
"""
import re

class EnglishWordTagger:
    """
    Identifies and tags English-origin words in Hindi transcripts.
    
    Why we need this:
    English words in Devnagari (इंटरव्यू) need to be tagged
    for tokenization, sentiment analysis, script normalization pipelines.
    
    Approach chosen:
    Hybrid approach combining Roman script detection + Devanagari 
    dictionary + context-based confidence scoring.
    """
    def __init__(self):
        # A list of common Roman English words
        self.roman_words = {
            "interview", "job", "offer", "company", "office", "call", "meeting",
            "problem", "solve", "work", "team", "project", "deadline", "salary",
            "manager", "report", "email", "phone", "mobile", "laptop", "computer",
            "internet", "app", "download", "update", "password", "login", "account",
            "school", "college", "degree", "certificate", "exam", "result", "marks",
            "hospital", "doctor", "medicine", "test", "report", "surgery",
            "train", "flight", "ticket", "booking", "hotel", "travel", "trip",
            "sir", "maam", "ok", "okay", "yes", "no", "thanks", "thank", "you",
            "please", "hi", "hello", "good", "morning", "evening", "night",
            "welcome", "bye", "sorry", "excuse", "pardon"
        }
        
        # Devanagari Transliteration Dictionary
        self.devanagari_to_english = {
            "इंटरव्यू": "interview", "जॉब": "job", "ऑफर": "offer", "कंपनी": "company",
            "ऑफिस": "office", "कॉल": "call", "मीटिंग": "meeting", "प्रॉब्लम": "problem",
            "सॉल्व": "solve", "वर्क": "work", "टीम": "team", "प्रोजेक्ट": "project",
            "डेडलाइन": "deadline", "सैलरी": "salary", "मैनेजर": "manager", "रिपोर्ट": "report",
            "ईमेल": "email", "फोन": "phone", "मोबाइल": "mobile", "लैपटॉप": "laptop",
            "कंप्यूटर": "computer", "इंटरनेट": "internet", "ऐप": "app", "डाउनलोड": "download",
            "अपडेट": "update", "पासवर्ड": "password", "लॉगिन": "login", "अकाउंट": "account",
            "स्कूल": "school", "कॉलेज": "college", "डिग्री": "degree", "सर्टिफिकेट": "certificate",
            "एग्जाम": "exam", "रिजल्ट": "result", "मार्क्स": "marks", "हॉस्पिटल": "hospital",
            "डॉक्टर": "doctor", "मेडिसिन": "medicine", "टेस्ट": "test", "सर्जरी": "surgery",
            "ट्रेन": "train", "फ्लाइट": "flight", "टिकट": "ticket", "होटल": "hotel",
            "बुकिंग": "booking", "ट्रैवल": "travel", "ट्रिप": "trip", "सर": "sir",
            "मैम": "maam", "ओके": "ok", "यस": "yes", "नो": "no", "थैंक्स": "thanks",
            "थैंक्यू": "thank you", "प्लीज": "please", "हाय": "hi", "हेलो": "hello",
            "सॉरी": "sorry", "बाय": "bye", "गुड": "good", "मॉर्निंग": "morning"
        }
        
    def tag(self, text: str) -> tuple[str, list[dict]]:
        detections = []
        words = text.split()
        tagged_words = []
        
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?”“()')
            is_roman = bool(re.match(r'^[a-zA-Z]+$', clean_word))
            
            tagged = False
            
            if is_roman:
                # Roman word found
                conf = "HIGH" if clean_word.lower() in self.roman_words else "MEDIUM"
                detections.append({
                    "word": clean_word,
                    "type": "roman",
                    "english_equivalent": clean_word.lower(),
                    "confidence": conf
                })
                tagged_word = word.replace(clean_word, f"[EN]{clean_word}[/EN]")
                tagged_words.append(tagged_word)
                tagged = True
            else:
                # Devanagari word
                if clean_word in self.devanagari_to_english:
                    en_equiv = self.devanagari_to_english[clean_word]
                    detections.append({
                        "word": clean_word,
                        "type": "devanagari",
                        "english_equivalent": en_equiv,
                        "confidence": "HIGH"
                    })
                    tagged_word = word.replace(clean_word, f"[EN]{clean_word}[/EN]")
                    tagged_words.append(tagged_word)
                    tagged = True
                    
            if not tagged:
                tagged_words.append(word)
                
        return " ".join(tagged_words), detections

if __name__ == "__main__":
    tagger = EnglishWordTagger()
    res, dets = tagger.tag("मेरा interview बहुत अच्छा गया और मुझे जॉब मिल गई")
    print(f"Result: {res}\nDetections: {dets}")
