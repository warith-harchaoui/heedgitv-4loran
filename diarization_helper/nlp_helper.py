"""
NLP Helper

NLP Helper provides several functions to process text, including language detection, 
text cleaning, entity extraction, and keyword detection. It uses popular NLP libraries 
such as spaCy, langdetect, and nltk for performing language-based operations and the RAKE 
algorithm for keyword extraction.

The functions in this module are designed to support multiple languages, including 
French, English, Spanish, and others.

Requirements
------------
- spaCy
- langdetect
- nltk
- keybert
- sentence_transformers
- rake_nltk

Example usage
-------------
>>> text = "Artificial Intelligence (AI) is transforming industries across the globe."
>>> detect_keywords(text, score=False, top_n=10)
['a subset of ai', 'machine learning', 'artificial intelligence', 'ai']

Authors
-------
- Warith Harchaoui, https://harchaoui.org/warith

"""
from collections import Counter
from typing import List, Tuple, Union
import os_helper as osh
import spacy
import langdetect
import string
import nltk
from nltk.util import ngrams  # For n-gram generation
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from rake_nltk import Rake
import re

import unicodedata

valid_characters = []
def get_valid_characters():
    global valid_characters
    if len(valid_characters)>0:
        return valid_characters

    # Start with digits and ASCII letters ( and apostrophe and double quote and punctuation )
    valid_chars = set(string.digits + string.ascii_letters + "'\"" + ".!,*+;?<>=")
    
    # Loop through Unicode characters to find accented letters
    for codepoint in range(0x00C0, 0x024F):  # Common accented letters range
        char = chr(codepoint)
        if unicodedata.category(char).startswith("L"):  # Check if it's a letter
            valid_chars.add(char)
    
    valid_characters = sorted(valid_chars)
    return valid_characters




NLTK_LANGUAGES = [
    ("en", "english"),
    ("fr", "french"),
    ("de", "german"),
    ("es", "spanish"),
    ("it", "italian"),
    ("nl", "dutch"),
    ("pt", "portuguese"),
    ("ru", "russian"),
]

SPACY_LANGUAGES = [
    ("en", "en_core_web_sm"),
    ("fr", "fr_core_news_sm"),
    # ("es", "es_core_news_sm"),
    # ("de", "de_core_news_sm"),
    # ("it", "it_core_news_sm"),
    # ("nl", "nl_core_news_sm"),
    # ("pt", "pt_core_news_sm"),
    # ("ru", "ru_core_news_sm"),
]

# make sure you did 
# >> python -m spacy download en_core_web_sm
# >> python -m spacy download fr_core_news_sm
# >> python -m spacy download es_core_news_sm
# ... for other languages

spacy_models = {}
def load_model(lang_code: str) -> spacy.Language:
    """
    Load the appropriate spaCy language model based on the given language code.

    Parameters
    ----------
    lang_code : str
        The language code (e.g., 'fr' for French, 'en' for English).

    Returns
    -------
    spacy.Language
        The loaded spaCy language model.

    Raises
    ------
    ValueError
        If the language code is not supported.
    """
    global spacy_models, SPACY_LANGUAGES
    if lang_code in spacy_models:
        return spacy_models[lang_code]

    short2long = dict(SPACY_LANGUAGES)
    long2short = {long: short for short, long in SPACY_LANGUAGES}
    spacy_lang = None
    if lang_code in short2long:
        spacy_lang = short2long[lang_code]
    elif lang_code in long2short:
        spacy_lang = lang_code

    osh.check(not(spacy_lang is None), f"Language code {lang_code} not supported by Spacy.")
    
    # Load and spacy model if not already cached
    if not(lang_code in spacy_models):
        try:
            res =  spacy.load(spacy_lang)
        except:
            code = f"python -m spacy download {spacy_lang}"
            osh.error(f"Spacy is not configured for your language ({lang_code} | {spacy_lang}), please do: {code}")
        spacy_models[spacy_lang] = res
        spacy_models[long2short[spacy_lang]] = res
        spacy_models[lang_code] = res

    return spacy_models[lang_code]



lang2stopwords = {}
def load_stopwords(lang_code: str) -> List[str]:
    """
    Load the appropriate stopwords list based on the given language code.

    Parameters
    ----------
    lang_code : str
        The language code (e.g., 'fr' for French, 'en' for English).

    Returns
    -------
    list of str
        A sorted list of unique stopwords for the specified language.
    """
    global NLTK_LANGUAGES, lang2stopwords
    if lang_code in lang2stopwords:
        return lang2stopwords[lang_code]

    short2long = dict(NLTK_LANGUAGES)
    long2short = {long: short for short, long in NLTK_LANGUAGES}

    nltk_lang = None
    if lang_code in short2long:
        nltk_lang = short2long[lang_code]
    elif lang_code in long2short:
        nltk_lang = lang_code

    osh.check(not(nltk_lang is None), f"Language code {lang_code} not supported by NLTK.")

    # Load and cache stopwords if not already cached
    if lang_code not in lang2stopwords:
        nltk.download("stopwords", quiet=True)  # Ensure stopwords are downloaded
        res = sorted(set(nltk.corpus.stopwords.words(nltk_lang)))
        lang2stopwords[nltk_lang] = res
        lang2stopwords[long2short[nltk_lang]] = res
        lang2stopwords[lang_code] = res

    return lang2stopwords[lang_code]

def detect_language(text: str, error: bool = True) -> str:
    """
    Detect the language of the given text.

    Parameters
    ----------
    text : str
        The text for which to detect the language.
    error : bool, optional
        Whether to raise an error if the language cannot be detected (default is True).

    Returns
    -------
    str
        The detected language code (e.g., "fr" for French, "en" for English).

    Raises
    ------
    ValueError
        If the language detection fails and `error` is set to True.
    """
    res = "en"
    try:
        res = langdetect.detect(text)
        osh.info(f"Detected language: {res}")
    except Exception as e:
        res = "unrecognized language"    
        if error:
            osh.error(
                f"Language detection failed for text:\n{text}",
            )
            
    return res


def clean_text(input_text: str) -> str:
    """
    Clean and normalize the input text by removing unwanted characters and symbols.

    Parameters
    ----------
    input_text : str
        The raw input text.

    Returns
    -------
    str
        The cleaned and normalized text.
    """
    lines = input_text.split("\n")
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    forbidden = ["@", "http", "#"]
    lines = [
            [
                w
                for w in line.split(" ")
                if len(w) > 0 and not (any([w.startswith(b) for b in forbidden]))
            ]
            for line in lines
        ]
    lines = [" ".join(words) for words in lines]
    # one-word forbidden lines
    forbidden += ["//", "http", "www", "https", "ftp", ":", "(", ")", "[", "]", "{", "}", "<", ">", "«", "»", "“", "”", "’", "‘", "—", "–", "…"]
    lines = [line for line in lines if not((len(line.split(" ")) == 1) and any([w in line for w in forbidden]))]
    text = "\n".join(lines)
    # Remove extra spaces and minuses
    change = True
    while change:
        old_text = str(text)
        text = text.replace("  ", " ")
        text = text.replace("_", "-")
        text = text.replace("--", "-")
        change = old_text != text

    # Punctuation spaces
    text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
    text = text.replace("…", "...").replace(" ...", '...')

    # Apostrophes
    text = text.replace("’", "'").replace("`", "'").replace("‘", "'").replace(" '", "'").replace("' ", "'")
    # Quotes
    text = text.replace("«", '"').replace("»", '"').replace('“', '"').replace('”', '"').replace(' "', '"').replace('" ', '"')

    valid_characters = get_valid_characters()
    # Remove invalid characters
    lines = text.split("\n")
    lines = [ell.split(" ") for ell in lines]
    lines = [[w for w in ell if all([c in valid_characters for c in w])] for ell in lines]
    lines = [" ".join(ell) for ell in lines]
    text = "\n".join(lines)
    

    return text


def filter_out_bad_keywords_list(keywords: List[str], lang:str) -> List[str]:
    """
    Filter out unwanted keywords from the provided list based on specific rules.

    Parameters
    ----------
    keywords : list of str
        List of keywords to filter.
    lang: str
        Languages (remove stop words)

    Returns
    -------
    list of str
        Cleaned list of keywords.
    """
    # Remove leading/trailing whitespace
    entities = [k.strip() for k in keywords if len(k.strip()) > 0]

    # Standardize quotation marks and remove specific leading phrases
    entities = [re.sub(r"[‘’`]", "'", e) for e in entities]
    entities = [e.replace('“', '"').replace('”', '"') for e in entities]
    entities = [e.replace("L'", "").replace("l'", "") for e in entities]
    entities = [e.replace("Le ", "").replace(" le ", "") for e in entities]

    stopwords = load_stopwords(lang)
    # remove pure stop words
    entities = list(filter(lambda e: not(e.strip().lower() in stopwords), entities))

    # remove entities just made of stop words
    entities = list(filter(lambda e: not(all([w in stopwords for w in e.strip().lower().split()])), entities))

    # Remove entities containing unwanted substrings
    unwanted_substrings = {"http", "www", "\n", "\t", "--", "//", ":"}
    entities = [e for e in entities if not any(sub in e for sub in unwanted_substrings)]

    # Remove entities in which all words are made of just numbers
    entities = [
        e for e in entities 
        if all(any(not c.isdigit() for c in word) for word in e.split())
    ]

    # Filter out any entities that became too short after processing
    entities = [e for e in entities if len(e) > 3]

    return entities


def extract_entities(input_text: str, lang: str = "fr", break_n_grams: bool = False) -> List[str]:
    """
    Extract entities of specified types from the given text using the specified language model.

    Parameters
    ----------
    input_text : str
        The text from which to extract entities.
    lang : str, optional
        The language code for the spaCy model to use (default is "fr").
    break_n_grams : bool, optional
        Whether to break n-grams into individual words (default is False).

    Returns
    -------
    list of str
        A list of extracted entities.
    """
    nlp = load_model(lang)
    text = str(input_text)
    doc = nlp(text)
    keys = ["PERSON", "ORG", "GPE", "LOC", "PER", "WORK_OF_ART", "NORP"]
    entities = [ent.text for ent in doc.ents if ent.label_ in keys]
    # entities = filter_out_bad_keywords_list(entities, lang)
    # entities = [e for e in entities if not (e.endswith("'"))]
    if break_n_grams:
        ell = []
        for e in entities:
            ell += e.split(" ")
        entities = ell
    entities = [e for e in entities if not (e.endswith("'"))]
    entities = filter_out_bad_keywords_list(entities, lang)
    return entities

rake_model = None
def detect_keywords(
    input_text: str, 
    lang: str = None, 
    top_n: int = 5, 
    ngram_max: int = 5, 
    score: bool = False, 
    check_language: bool = False, 
    safe_list: List[str] = None,
    score_theshold: float = 1.0
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Extract keywords from the given text using the RAKE algorithm.

    Parameters
    ----------
    input_text : str
        The text from which to extract keywords.
    lang : str, optional
        The language code for the text (default is None).
    top_n : int, optional
        The number of keywords to extract (default is 5).
    ngram_max : int, optional
        The maximum n-gram length to consider (default is 5).
    score : bool, optional
        Whether to return the keywords with their scores (default is False).
    check_language : bool, optional
        Whether to check the language of each keyword to filter them out (default is True).
    safe_list : list of str, optional
        The list of entities to filter out (default is None).
    score_theshold: float, optional
        Filter out bad keywords

    Returns
    -------
    list of str or list of tuple
        The extracted keywords or keywords with scores.

    Example
    -------
    >>> text = "Artificial Intelligence (AI) is transforming industries across the globe."
    >>> keywords = detect_keywords(text, lang="en", score=False, top_n=10)
    >>> print(keywords)
    ['a subset of ai', 'machine learning', 'artificial intelligence', 'ai']
    """
    global rake_model

    text = str(input_text)

    if lang is None:
        lang = detect_language(text)

    # Load appropriate stopwords based on the language
    stopwords_lang = load_stopwords(lang)

    # Initialize RAKE model if not already initialized
    if rake_model is None:
        nltk.download("punkt_tab")
        rake_model = Rake(stopwords=stopwords_lang, max_length=ngram_max)

    # Extract keywords using RAKE
    rake_model.extract_keywords_from_text(text)

    # Get ranked phrases (with scores)
    ranked_phrases_with_scores = rake_model.get_ranked_phrases_with_scores()

    kws = [r[1] for r in ranked_phrases_with_scores]
    p = filter_out_bad_keywords_list(kws, lang)

    ranked_phrases_with_scores = [(r[0], r[1]) for r in ranked_phrases_with_scores if r[1] in p]

    # Sort phrases by score
    ranked_phrases_with_scores = sorted(
        ranked_phrases_with_scores, key=lambda x: x[0], reverse=True
    )

    # Apply thresholding for RAKE (filtering out phrases with low relevance)
    filtered_phrases = [
        phrase for phrase in ranked_phrases_with_scores
        if phrase[0] > score_theshold
    ]

    if check_language:
        if safe_list is None:
            filtered_phrases = [
                phrase for phrase in filtered_phrases
                if detect_language(phrase[1], error=False) == lang
            ]
        else:
            filtered_phrases = [
                phrase for phrase in filtered_phrases
                if detect_language(phrase[1], error=False) == lang
                or any([e in phrase[1] for e in safe_list])
            ]

    # Return the top_n results
    filtered_phrases = filtered_phrases[:min(top_n, len(filtered_phrases))]

    if score:
        return filtered_phrases
    
    res = [phrase[1] for phrase in filtered_phrases]
    return res


if __name__ == "__main__":

    # Example usage of the functions with French text
    example_text_fr = "Vincent, François, Paul ... et les autres ne savent pas ce qu'ils font dans la vie."
    lang = detect_language(example_text_fr)
    osh.check(
        lang == "fr",
        msg="Language detection failed.\n\tExpected: fr\nDetected: %s" % lang,
    )
    osh.info(f"Good language detection: {lang}\nfor text:\n{example_text_fr}")
    entities = extract_entities(example_text_fr, lang=lang)
    s = "\n\t".join(entities)
    entities_gt = set(["Vincent", "François", "Paul"])
    osh.check(
        set(entities) == entities_gt,
        msg=f"Entity extraction failed for\n{example_text_fr}\n\t{s}\nExpected: {entities_gt}",
    )
    osh.info(
        f"Good entities detection for text:\n{example_text_fr}\nEntities\n\t{s}"
    )

    # Example usage with English text
    example_text_en = "In the movie Star Wars, characters like C3-PO, Lando Calrissian or Luke Skywalker used to say 'I have a bad feeling about this.'"
    lang = detect_language(example_text_en)
    osh.check(
        lang == "en",
        msg="Language detection failed.\n\tExpected: en\nDetected: %s" % lang,
    )
    osh.info(f"Good language detection: {lang}\nfor text:\n{example_text_en}")
    entities = extract_entities(example_text_en, lang=lang, break_n_grams=False)
    s = "\n\t".join(entities)
    entities_gt = set(["Star Wars", "C3-PO", "Lando Calrissian", "Luke Skywalker"])
    osh.check(
        set(entities) == entities_gt,
        msg=f"Entity extraction failed for\n{example_text_en}\n\t{s}\nExpected: {entities_gt}",
    )
    osh.info(
        f"Good entities detection for text:\n{example_text_en}\nEntities\n\t{s}"
    )

    # Example usage with keyword detection
    text = (
        "Artificial Intelligence (AI) is transforming industries across the globe. "
        "Machine learning, a subset of AI, allows computers to learn from data without being explicitly programmed. "
        "Neural networks and deep learning algorithms are enabling machines to process complex patterns and make predictions. "
        "AI applications range from natural language processing to computer vision, making it a key technology for the future."
    )
    keywords = detect_keywords(text, score=False, top_n=10)
    s = "\n\t".join(keywords)
    keywords_gt = set(["a subset of ai", "machine learning", "artificial intelligence", "ai"])
    osh.check(
        set(keywords) == keywords_gt,
        msg=f"Keyword extraction failed for\n{text}\n\t{s}\nExpected: {keywords_gt}",
    )
    osh.info(f"Good keyword extraction for text:\n{text}\nKeywords\n\t{s}")
