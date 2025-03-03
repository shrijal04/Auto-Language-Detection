# to run this we must firsts install two python libraries
# pip install langdetect
# pip install pycountry

from langdetect import detect
import pycountry

def get_language_full_name(text):
    lang_code = detect(text)
    language = pycountry.languages.get(alpha_2=lang_code)
    if language:
        return language.name.split(" (")[0]  # Remove extra details like " (macrolanguage)"
    return "Unknown Language"

text = "I am working hard right now."
detected_language = get_language_full_name(text)
print(detected_language)  