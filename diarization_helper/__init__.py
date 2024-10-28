"""
Diarization Helper

This module provides a set of utility functions to determine who said when in a video or audio file.

Dependencies
- os_helper (custom utility for OS tasks)
- audio_helper (custom audio processing utility)
- video_helper (custom video processing utility)
- yt_helper (custom YouTube processing utility)

Authors:
- Warith Harchaoui, https://harchaoui.org/warith
"""

from .mix_gladia_pyannote import diarization, words2breath_segments
from .pyannote_helper import credentials as pyannote_credentials
from .gladia_helper import credentials as gladia_credentials, gladia_processing
from sftp_helper import credentials as sftp_credentials
from .donut import make_donut, make_donut_image
from .angel_hair_pasta import make_angel_hair_pasta, make_angel_hair_pasta_image
from .elbow import elbow_visu
from .nlp_helper import detect_language, extract_entities, detect_keywords, clean_text

__all__ = [
    'diarization',
    "pyannote_credentials",
    "gladia_credentials",
    "sftp_credentials",
    "make_donut",
    "words2breath_segments",
    "elbow_visu",
    "make_donut",
    "make_donut_image",
    "detect_language",
    "extract_entities",
    "detect_keywords",
    "make_angel_hair_pasta",
    "make_angel_hair_pasta_image",
    "gladia_processing", 
    "clean_text",
]
