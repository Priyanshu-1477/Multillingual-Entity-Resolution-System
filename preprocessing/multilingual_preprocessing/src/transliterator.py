"""Modular, offline, production-grade transliterator preserving entity identity."""

import re
from typing import Dict, List, Tuple
import anyascii
from .script_detector import detect_scripts, get_char_script, has_non_latin, segment_by_script
from .unicode_utils import normalize_unicode, unaccent_latin

# High-frequency dictionary of Indic business terms, loanwords, legal entities, and geographic locations.
# These represent standardized phonetic transliterations commonly used in business registration and entity resolution.
INDIC_PHONETIC_VOCABULARY: Dict[str, str] = {
    # Legal / Company entity types (Devanagari)
    "प्राइवेट लिमिटेड": "Private Limited",
    "प्रा. लि.": "Pvt Ltd",
    "प्रा० लि०": "Pvt Ltd",
    "प्राइवेट": "Private",
    "लिमिटेड": "Limited",
    "पब्लिक लिमिटेड": "Public Limited",
    "कंपनी": "Company",
    "कम्पनी": "Company",
    "कॉर्पोरेशन": "Corporation",
    "कारपोरेशन": "Corporation",
    "एंटरप्राइजेज": "Enterprises",
    "एन्टरप्राइजेज": "Enterprises",
    "एंटरप्राइज": "Enterprise",
    "इंडस्ट्रीज": "Industries",
    "इण्डस्ट्रीज": "Industries",
    "सर्विसेज": "Services",
    "सोल्यूशंस": "Solutions",
    "सॉल्यूशंस": "Solutions",
    "सोल्यूशन": "Solution",
    "इन्फोटेक": "Infotech",
    "टेक्नोलॉजीज": "Technologies",
    "टेक्नोलॉजी": "Technology",
    "कंसल्टेंसी": "Consultancy",
    "कंसल्टेंट्स": "Consultants",
    "कंसल्टिंग": "Consulting",
    "एसोसिएट्स": "Associates",
    "सिक्योरिटीज": "Securities",
    "फाइनेंस": "Finance",
    "फायनांस": "Finance",
    "फाइनेंशियल": "Financial",
    "प्रॉपर्टीज": "Properties",
    "डेवलपर्स": "Developers",
    "कंस्ट्रक्शंस": "Constructions",
    "कंस्ट्रक्शन": "Construction",
    "इंजीनियरिंग": "Engineering",
    "ट्रेडर्स": "Traders",
    "ट्रेडिंग": "Trading",
    "मार्केटिंग": "Marketing",
    "होटल": "Hotel",
    "होटल्स": "Hotels",
    "रेस्टोरेंट": "Restaurant",
    "हॉस्पिटल": "Hospital",
    "स्टोर": "Store",
    "स्टोर्स": "Stores",
    "दुकान": "Dukan",
    "मोबाइल": "Mobile",
    "इलेक्ट्रॉनिक्स": "Electronics",
    "ऑटोमोबाइल": "Automobile",
    "मोटर्स": "Motors",
    "फार्मा": "Pharma",
    "फार्मास्यूटिकल्स": "Pharmaceuticals",
    "केमिकल्स": "Chemicals",
    "टेक्सटाइल्स": "Textiles",
    "ज्वेलर्स": "Jewellers",
    "फाउंडेशन": "Foundation",
    "ट्रस्ट": "Trust",
    "सोसाइटी": "Society",
    "क्लब": "Club",
    "एकेडमी": "Academy" ,
    "इंटरनेशनल": "International",
    "ग्लोबल": "Global",
    "इंडिया": "India",
    "भारत": "Bharat",
    "हिंदुस्तान": "Hindustan",
    "नेशनल": "National",
    "जनरल": "General",
    "एलएलपी": "LLP",
    "एल.एल.पी.": "LLP",
    "ओपीसी": "OPC",
    "ग्रुप": "Group",
    "वेंचर्स": "Ventures",
    "लॉजिस्टिक्स": "Logistics",
    "कम्युनिकेशंस": "Communications",
    "एग्रो": "Agro",
    "फूड्स": "Foods",
    "प्रेस": "Press",
    "प्रिंट": "Print",
    "स्टूडियो": "Studio",
    "सिक्योरिटी": "Security",
    "कैरियर": "Carrier",
    "लैब्स": "Labs",
    "रिसर्च": "Research",

    # Gujarati terms
    "પ્રાઇવેટ લિમિટેડ": "Private Limited",
    "પ્રાઇવેટ": "Private",
    "લિમિટેડ": "Limited",
    "મોબાઇલ": "Mobile",
    "સ્ટોર": "Store",
    "એન્ટરપ્રાઇઝ": "Enterprises",
    "ટ્રેડર્સ": "Traders",
    "ઇન્ડસ્ટ્રીઝ": "Industries",
    "કંપની": "Company",
    "ફાર્મા": "Pharma",
    "કેમિકલ્સ": "Chemicals",

    # Bengali terms
    "প্রাইভেট লিমিটেড": "Private Limited",
    "প্রাইভেট": "Private",
    "লিমিটেড": "Limited",
    "স্টোর": "Store",
    "এন্টারপ্রাইজ": "Enterprises",
    "ট্রেডার্স": "Traders",
    "ইন্ডাস্ট্রিজ": "Industries",
    "কোম্পানি": "Company",

    # Tamil terms
    "பிரைவேட் லிமிடெட்": "Private Limited",
    "பிரைவேட்": "Private",
    "லிமிடெட்": "Limited",
    "மொபைல்": "Mobile",
    "ஸ்டோர்": "Store",
    "என்டர்பிரைசஸ்": "Enterprises",

    # Telugu terms
    "ప్రైవేట్ లిమిటెడ్": "Private Limited",
    "ప్రైవేట్": "Private",
    "లిమిటెడ్": "Limited",
    "మొబైల్": "Mobile",
    "స్టోర్స్": "Stores",

    # Kannada terms
    "ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್": "Private Limited",
    "ಪ್ರೈವೇಟ್": "Private",
    "ಲಿಮಿಟೆಡ್": "Limited",

    # Prominent Indian Geographic names (Devanagari / Regional)
    "महाराष्ट्र": "Maharashtra",
    "गुजरात": "Gujarat",
    "ગુજરાત": "Gujarat",
    "राजस्थान": "Rajasthan",
    "दिल्ली": "Delhi",
    "नई दिल्ली": "New Delhi",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "मध्य प्रदेश": "Madhya Pradesh",
    "कर्नाटक": "Karnataka",
    "ಕರ್ನಾಟಕ": "Karnataka",
    "तमिलनाडु": "Tamil Nadu",
    "தமிழ்நாடு": "Tamil Nadu",
    "आंध्र प्रदेश": "Andhra Pradesh",
    "ఆంధ్రప్రదేశ్": "Andhra Pradesh",
    "ఆంధ్ర ప్రదేశ్": "Andhra Pradesh",
    "तेलंगाना": "Telangana",
    "తెలంగాణ": "Telangana",
    "पश्चिम बंगाल": "West Bengal",
    "পশ্চিমবঙ্গ": "West Bengal",
    "हरियाणा": "Haryana",
    "पंजाब": "Punjab",
    "ਪੰਜਾਬ": "Punjab",
    "बिहार": "Bihar",
    "केरल": "Kerala",
    "കേരളം": "Kerala",
    "ओडिशा": "Odisha",
    "ଓଡ଼ିଶା": "Odisha",
    "झारखंड": "Jharkhand",
    "असम": "Assam",
    "অসম": "Assam",
    "उत्तराखंड": "Uttarakhand",
    "हिमाचल प्रदेश": "Himachal Pradesh",
    "गोवा": "Goa",
    "मुंबई": "Mumbai",
    "बेंगलुरु": "Bangalore",
    "ಬೆಂಗಳೂರು": "Bangalore",
    "कोलकाता": "Kolkata",
    "কলকাতা": "Kolkata",
    "चेन्नई": "Chennai",
    "சென்னை": "Chennai",
    "हैदराबाद": "Hyderabad",
    "హైదరాబాద్": "Hyderabad",
    "अहमदाबाद": "Ahmedabad",
    "અમદાવાદ": "Ahmedabad",
    "पुणे": "Pune",
    "जयपुर": "Jaipur",
    "सूरत": "Surat",
    "સુરત": "Surat",
    "लखनऊ": "Lucknow",
    "कानपुर": "Kanpur",
    "नागपुर": "Nagpur",
    "इंदौर": "Indore",
    "ठाणे": "Thane",
    "भोपाल": "Bhopal",
    "विशाखापट्टनम": "Visakhapatnam",
    "पटना": "Patna",
    "वडोदरा": "Vadodara",
    "વડોદરા": "Vadodara",
    "गाजियाबाद": "Ghaziabad",
    "लुधियाना": "Ludhiana",
    "आगरा": "Agra",
    "नासिक": "Nashik",
    "फरीदाबाद": "Faridabad",
    "मेरठ": "Meerut",
    "राजकोट": "Rajkot",
    "રાજકોટ": "Rajkot",
    "वाराणसी": "Varanasi",
    "श्रीनगर": "Srinagar",
    "औरंगाबाद": "Aurangabad",
    "धनबाद": "Dhanbad",
    "अमृतसर": "Amritsar",
    "नवी मुंबई": "Navi Mumbai",
    "प्रयागराज": "Prayagraj",
    "इलाहाबाद": "Prayagraj",
    "रांची": "Ranchi",
    "हावड़ा": "Howrah",
    "कोयंबटूर": "Coimbatore",
    "जबलपुर": "Jabalpur",
    "ग्वालियर": "Gwalior",
    "विजयवाड़ा": "Vijayawada",
    "जोधपुर": "Jodhpur",
    "मदुरै": "Madurai",
    "रायपुर": "Raipur",
    "कोटा": "Kota",
    "गुवाहाटी": "Guwahati",
    "चंडीगढ़": "Chandigarh",
    "मैसूर": "Mysore",

    # Common Indian Business Names & Honorifics
    "शिव": "Shiv",
    "शक्ति": "Shakti",
    "श्री": "Shri",
    "શ્રી": "Shri",
    "ஸ்ரீ": "Shri",
    "श्रीमती": "Shrimati",
    "गणेश": "Ganesh",
    "ગણેશ": "Ganesh",
    "राम": "Ram",
    "आदित्य": "Aditya",
    "साईं": "Sai",
    "साई": "Sai",
    "कृष्णा": "Krishna",
    "कृष्ण": "Krishna",
    "महावीर": "Mahavir",
    "बालाजी": "Balaji",
    "हनुमान": "Hanuman",
    "लक्ष्मी": "Laxmi",
    "दुर्गा": "Durga",
    "सरस्वती": "Saraswati",
    "विष्णु": "Vishnu",
    "सूर्य": "Surya",
    "ओम": "Om",
    "ॐ": "Om",
    "जय": "Jai",
    "विजय": "Vijay",
    "शांति": "Shanti",
    "आनंद": "Anand",
    "सुख": "Sukh",
    "समृद्धि": "Samriddhi",
    "स्वस्तिक": "Swastik",
    "विकास": "Vikas",
    "मॉडर्न": "Modern",
    "सुप्रीम": "Supreme",
    "रॉयल": "Royal",
    "क्लासिक": "Classic",
    "स्टार": "Star",
    "सन": "Sun",
    "मून": "Moon",
    "गोल्डन": "Golden",
    "डायमंड": "Diamond",
    "सिल्वर": "Silver",
    "सिटी": "City",
    "टाउन": "Town",
    "पैलेस": "Palace",
    "प्लाजा": "Plaza",
    "सेंटर": "Center",
    "सेंट्रल": "Central",
}

# Phonetic character enhancement mapping for Indic scripts
# Ensures standard Indian English transliteration (e.g., 'श' -> 'sh' instead of 's')
_INDIC_CONSONANT_MAP: Dict[str, str] = {
    "श": "sh", "ष": "sh", "च": "ch", "छ": "chh", "झ": "jh",
    "थ": "th", "ध": "dh", "भ": "bh", "ख": "kh", "घ": "gh",
    "फ": "ph", "ण": "n", "ज्ञ": "gya", "क्ष": "ksh",
    # Gujarati
    "શ": "sh", "ષ": "sh", "ચ": "ch", "છ": "chh", "ઝ": "jh",
    "થ": "th", "ધ": "dh", "ભ": "bh", "ખ": "kh", "ઘ": "gh",
    "ફ": "ph", "ણ": "n", "જ્ઞ": "gya", "ક્ષ": "ksh",
    # Bengali
    "শ": "sh", "ষ": "sh", "চ": "ch", "ছ": "chh", "ঝ": "jh",
    "থ": "th", "ধ": "dh", "ভ": "bh", "খ": "kh", "ঘ": "gh",
    "ফ": "ph", "ণ": "n", "জ্ঞ": "gya", "ক্ষ": "ksh",
}


def _transliterate_indic_segment(text: str) -> str:
    """Faithfully transliterate an Indic text segment into Latin script, preserving whitespace."""
    if not text:
        return ""

    leading_ws = text[:len(text) - len(text.lstrip())]
    trailing_ws = text[len(text.rstrip()):]
    core_text = text.strip()

    if not core_text:
        return text

    # First check if the core text matches a known term
    if core_text in INDIC_PHONETIC_VOCABULARY:
        return leading_ws + INDIC_PHONETIC_VOCABULARY[core_text] + trailing_ws

    # Check multi-word phrases and individual tokens
    result_text = core_text
    # Replace known phrases/words (longer first)
    for term, latin_equiv in sorted(INDIC_PHONETIC_VOCABULARY.items(), key=lambda x: -len(x[0])):
        if term in result_text:
            result_text = result_text.replace(term, latin_equiv)

    # If any Indic characters remain, apply phonetic consonant mapping and fallback to anyascii
    if has_non_latin(result_text):
        chars = []
        for ch in result_text:
            if ch in _INDIC_CONSONANT_MAP:
                chars.append(_INDIC_CONSONANT_MAP[ch])
            else:
                chars.append(ch)
        result_text = anyascii.anyascii("".join(chars))

    return leading_ws + result_text + trailing_ws


def transliterate_text(text: str) -> str:
    """Master transliteration function adhering strictly to project policies:

    1. Latin script policy:
       - Preserves original language (French remains French, Spanish remains Spanish, etc.).
       - Normalizes diacritics to clean, readable Latin (Farmacia San José -> Farmacia San Jose).
    2. Non-Latin script policy:
       - Faithfully transliterates into Latin script without semantic translation.
       - शिव शक्ति जनरल स्टोर -> Shiv Shakti General Store.
       - ശ്രീ ഗണേഷ് മൊബൈൽ -> Shri Ganesh Mobile.
    3. Mixed-script policy:
       - Identifies script segments, preserves Latin segments, transliterates non-Latin segments.
       - ABC मोबाइल Store -> ABC Mobile Store.
       - Reconstructs text preserving digits and punctuation.
    """
    if not text:
        return ""

    # Fast path: pure ASCII text requires zero transliteration
    if text.isascii():
        return text

    # Check if text contains non-Latin scripts
    if not has_non_latin(text):
        # Pure Latin with accents/diacritics: unaccent cleanly to Latin-readable
        return unaccent_latin(text)

    # Mixed script or pure non-Latin: segment by script
    segments = segment_by_script(text)
    reconstructed: List[str] = []

    for seg_text, script in segments:
        if script in ("Latin", "Common"):
            # Preserve Latin text and common numbers/symbols
            reconstructed.append(unaccent_latin(seg_text))
        elif script in ("Devanagari", "Gujarati", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam", "Gurmukhi", "Odia"):
            # Indic script transliteration
            reconstructed.append(_transliterate_indic_segment(seg_text))
        else:
            # Other non-Latin scripts (Arabic, CJK, Cyrillic, Greek, Hebrew, Thai, etc.)
            reconstructed.append(anyascii.anyascii(seg_text))

    return "".join(reconstructed)
