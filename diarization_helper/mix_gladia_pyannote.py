"""
Module for combining PyAnnote and Gladia outputs for speaker diarization and transcription.

This module provides a function `diarization` that processes an audio file using PyAnnote
and Gladia, aligns the speakers from both systems, and outputs the combined results in JSON format.

Functions
---------
diarization(sound_path, output_file, sftp_cred, gladia_cred, pyannote_cred, lang, custom_vocabulary=[]):
    Processes the audio file and outputs the combined diarization and transcription results.


Authors
-------
- Warith Harchaoui, https://harchaoui.org/warith
"""

from itertools import product
from .pyannote_helper import pyannote_processing
from .gladia_helper import gladia_processing

from .elbow import abascus, continuous_elbow

import os_helper as osh
import numpy as np
import json

from typing import List, Dict, Any, Optional, Tuple

def diarization(
    sound_path: str,
    output_file: Optional[str],
    sftp_cred: Optional[Dict[str, Any]],
    gladia_cred: Optional[Dict[str, Any]],
    pyannote_cred: Optional[Dict[str, Any]],
    lang: Optional[str],
    custom_vocabulary: Optional[List[str]] = None,
    num_workers: int = 1,
    overwrite: bool = True
) -> str:
    """
    Performs speaker diarization and transcription on an audio file using PyAnnote and Gladia.

    Parameters
    ----------
    sound_path : str
        Path to the input audio file.
    output_file : Optional[str]
        Path to the output JSON file. If not provided, defaults to '<basename>-diarization.json'.
    sftp_cred : Dict[str, Any]
        Credentials for SFTP access.
    gladia_cred : Optional[Dict[str, Any]]
        Credentials for Gladia API access.
    pyannote_cred : Optional[Dict[str, Any]]
        Credentials for PyAnnote API access.
    lang : Optional[str]
        Language code for transcription.
    custom_vocabulary : Optional[List[str]]
        Custom vocabulary words to enhance transcription accuracy.
    num_workers : int
        Number of parallel workers for processing.
    overwrite : bool
        Whether to overwrite the output file if it already exists.

    Returns
    -------
    str
        Path to the output JSON file containing the combined results.

    Notes
    -----
    The function processes the audio file with PyAnnote for speaker diarization and Gladia
    for transcription. It then aligns speakers between the two systems based on overlapping
    time intervals using interval trees for efficient computation. The combined results
    include sentence-level and word-level transcriptions with speaker labels.
    """
    # Set default values if necessary
    if custom_vocabulary is None:
        custom_vocabulary = []

    # Extract folder, base name, and extension
    folder, basename, ext = osh.folder_name_ext(sound_path)

    # Set default output file paths
    if osh.emptystring(output_file):
        output_file = osh.join(folder, f"{basename}-diarization.json")

    # Check if the output file already exists
    if not(overwrite) and osh.file_exists(output_file):
        osh.info(f"Output file already exists: {output_file}")
        return output_file

    osh.check(not((gladia_cred is None) and (pyannote_cred is None)), "At least one of Gladia or PyAnnote valid credentials must be provided.")

    pyannote_segments = []
    if not(pyannote_cred is None):
        osh.check(not(sftp_cred is None), "SFTP credentials must be provided for PyAnnote processing.")
        # Run PyAnnote processing
        if not(pyannote_cred is None):
            pyannote_output_file = osh.join(folder, f"{basename}-pyannote.json")
            pyannote_processing(
                sound_path,
                sftp_cred,
                pyannote_cred,
                pyannote_output_file,
                num_workers=num_workers
            )
            # Load PyAnnote output
            with open(pyannote_output_file, 'r') as f:
                pyannote_data = json.load(f)
                pyannote_segments = pyannote_data["output"]["diarization"]
                confidences = pyannote_data["output"]["confidence"]["score"]
                freq_confidence_diarization_pyannote = 1.0 / pyannote_data["output"]["confidence"]["resolution"]
            pyannote_segments = sorted(pyannote_segments, key=lambda x: x["start"])

    gladia_segments = []
    if not(gladia_cred is None):
        # Run Gladia processing
        if not(gladia_cred is None):
            gladia_output_file = osh.join(folder, f"{basename}-gladia.json")
            gladia_processing(
                sound_path,
                gladia_cred,
                gladia_output_file,
                lang,
                custom_vocabulary,
                num_workers=num_workers
            )
            # Load Gladia output
            with open(gladia_output_file, 'r') as f:
                gladia_segments = json.load(f)
            gladia_segments = sorted(gladia_segments, key=lambda x: x["start"])

    if gladia_cred is None and not(pyannote_cred is None):
        # If only PyAnnote is available, use its diarization results
        with open(output_file, 'wt') as fout:
            json.dump(pyannote_segments, fout, indent=2)
        return output_file

    if not(gladia_cred is None) and pyannote_cred is None:
        # If only Gladia is available, use its diarization results
        with open(output_file, 'wt') as fout:
            json.dump(gladia_segments, fout, indent=2)
        return output_file
    
    # At this point, both Gladia and PyAnnote are available

    # Get unique speakers from PyAnnote and Gladia
    pyannote_speakers = sorted(list(set(str(segment["speaker"]) for segment in pyannote_segments)))
    gladia_speakers = sorted(list(set(str(segment["speaker"]) for segment in gladia_segments)))

    # Initialize overlap matrix
    overlap_matrix = np.zeros((len(pyannote_speakers), len(gladia_speakers)))

    # Compute overlaps 
    for i, ps in enumerate(pyannote_speakers):
        # Get all segments for the current PyAnnote speaker
        pyannote_speaker_segments = [
            segment for segment in pyannote_segments if str(segment["speaker"]) == ps
        ]
        for j, gs in enumerate(gladia_speakers):
            gladia_speaker_segments = [ 
                seg for seg in gladia_segments if str(seg["speaker"]) == gs
            ]
            for seg_py, seg_gl in product(pyannote_speaker_segments, gladia_speaker_segments):
                start_py = seg_py["start"]
                end_py = seg_py["end"]
                start_gl = seg_gl["start"]
                end_gl = seg_gl["end"]
                overlap_start = max(start_py, start_gl)
                overlap_end = min(end_py, end_gl)
                overlap_duration = max(0, overlap_end - overlap_start)
                overlap_matrix[i, j] += overlap_duration


    # Map PyAnnote speakers to Gladia speakers based on maximum overlap
    pyannote_to_gladia = {}
    for i, pyannote_speaker in enumerate(pyannote_speakers):
        j = np.argmax(overlap_matrix[i])
        pyannote_to_gladia[pyannote_speaker] = gladia_speakers[j]

    # Rename PyAnnote speakers to match Gladia speakers
    for i, segment in enumerate(pyannote_segments):
        original_speaker = str(segment["speaker"])
        pyannote_segments[i]["speaker"] = pyannote_to_gladia[original_speaker]

    pyannote_segments = sorted(pyannote_segments, key=lambda x: x["start"])

    # Prepare the result dictionary
    result = {}

    if len(gladia_segments)>0:
        # Sentence-level transcription
        result["sentences"] = []
        for segment in gladia_segments:
            confidence = segment["confidence"]
            if confidence > 0.19:
                entry = {
                    "sentence": segment["sentence"],
                    "speaker": str(segment["speaker"]),
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": confidence,
                }
                result["sentences"].append(entry)
        # Sort sentences by start time
        result["sentences"] = sorted(result["sentences"], key=lambda x: x["start"])

        # Word-level transcription
        result["words"] = []
        for segment in gladia_segments:
            confidence = segment["confidence"]
            for word_info in segment.get("words", []):
                if confidence > 0.19:
                    entry = {
                        "word": word_info["word"],
                        "speaker": str(segment["speaker"]),
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "confidence": word_info.get("confidence", None),
                    }
                    result["words"].append(entry)
        # Sort words by start time
        result["words"] = sorted(result["words"], key=lambda x: x["start"])

    # Diarization information
    if len(pyannote_segments)>0:
        result["diarization"] = []
        for segment in pyannote_segments:
            entry = {
                "speaker": str(segment["speaker"]),
                "start": segment["start"],
                "end": segment["end"],
            }
            result["diarization"].append(entry)
        # Sort diarization by start time
        result["diarization"] = sorted(result["diarization"], key=lambda x: x["start"])

    # Diarization confidence scores
    if len(confidences)>0:
        result["diarization_confidence"] = confidences
        result["diarization_confidence_rate"] = freq_confidence_diarization_pyannote

    # Save the result to the output file
    with open(output_file, 'wt') as fout:
        json.dump(result, fout, indent=2)

    osh.info(f"Speaker diarization and transcription results saved to: {output_file}")

    return output_file



def words2breath_segments(words: List[Dict[str, Any]], max_inter_word_silence:float = 1.5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Creates breath segments from word segments based on silence intervals between words.

    Parameters
    ----------
    words : List[Dict[str, Any]]
        List of word segments with 'start', 'end', 'speaker', and 'word' keys.
    max_inter_word_silence : float
        Maximum silence interval between words to consider for breath segmentation
    
    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        Tuple of elbow analysis results and list of breath segments.
    
    Notes
    -----
    The function computes the silence intervals between words and applies an elbow analysis
    to determine the optimal threshold for grouping words into breath segments. The output
    includes the elbow analysis (with an abascus) and a list of breath segments with 'start', 'end',
    'speaker', and 'word' keys.

    """
    segments = [dict(seg) for seg in words]
    segments = sorted(segments, key=lambda x: x["start"])
    silences_between_words = [next["start"] - previous["end"] for previous, next in zip(segments[:-1], segments[1:])]
    bs = 0.02
    x = np.arange(0, max(silences_between_words) + bs, bs)
    y = np.histogram(silences_between_words, bins=x)[0]
    x2 = []
    y2 = []
    for xx, yy in zip(x, y):
        if yy > 0 and xx<max_inter_word_silence:
            x2.append(xx)
            y2.append(yy)
    x = x2
    y = y2
    x_y = np.array(list(zip(x, y)))
    res = abascus(x_y, device="cpu")
    tau = res["elbow"]
    breath_segments = []
    current_segment = None
    for i, word in enumerate(segments):
        if current_segment is None:
            current_segment = dict(word)
        else:
            if word["speaker"] == current_segment["speaker"] and word["start"] - current_segment["end"] < tau:
                current_segment["end"] = word["end"]
                current_segment["word"] += word["word"]
            else:
                breath_segments.append(current_segment)
                current_segment = word

    breath_segments.append(current_segment)

    res["x"] = [xx for xx in x]
    res["y"] = [yy for yy in y]

    return res, breath_segments

