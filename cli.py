"""
Command-line interface for the diarization module.

This script allows users to perform speaker diarization and transcription on an audio file
using PyAnnote and Gladia from the command line.

Usage:
    python cli.py --sound_path /path/to/audio.wav --output_file /path/to/output.json \
                  --sftp_cred /path/to/sftp_cred.json \
                  --gladia_cred /path/to/gladia_cred.json \
                  --pyannote_cred /path/to/pyannote_cred.json \
                  --lang en \
                  --custom_vocabulary word1 word2 word3
"""

import argparse
import json

from diarization_helper.donut import make_donut_image
import os_helper as osh
import audio_helper as ah
import video_helper as vh
import yt_helper as yth

import diarization_helper

import numpy as np

import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speaker Diarization and Transcription CLI")
    parser.add_argument('--input', "-i", type=str, required=True, help='Path to the input audio/video/url file.')
    parser.add_argument('--output_file', "-o", type=str, help='Path to the output JSON file.', default="")
    parser.add_argument('--sftp_cred', type=str, help='Path to the SFTP credentials JSON file.', default="config/sftp")
    parser.add_argument('--gladia_cred', type=str, help='Path to the Gladia credentials JSON file.', default="config/gladia")
    parser.add_argument('--pyannote_cred', type=str, help='Path to the PyAnnote credentials JSON file.', default="config/pyannote")
    parser.add_argument('--custom_vocabulary', nargs='*', default=[], help='List of custom vocabulary words.')
    parser.add_argument('--disable_sftp', action='store_true', help='Disable SFTP access.')
    parser.add_argument('--disable_gladia', action='store_true', help='Disable Gladia access.')
    parser.add_argument('--disable_pyannote', action='store_true', help='Disable PyAnnote access.')
    parser.add_argument('--disable_source_separation', action='store_true', help='Disable source separation.')
    parser.add_argument('--device', type=str, default="cpu", help='Device to use for source separation. (it can be cuda for example)')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level. 0: silent, 1: normal, 2: verbose')

    args = parser.parse_args()
    lang = []

    input_audio_video_url = args.input
    input_sound_path = None
    if osh.is_working_url(input_audio_video_url):
        # URL
        if yth.is_valid_video_url(input_audio_video_url):
            metadata = yth.video_url_meta_data(input_audio_video_url)
            title = metadata["title"]
            description = metadata["description"]
            text = f"{title}\n\n{description}"
            lang = diarization_helper.detect_language(text)
            osh.info(f"Detected language: {lang}")
            basename = osh.asciistring(title)
            name = title
            osh.make_directory(basename)
            input_sound_path = f"audio.mp3"
            input_sound_path = osh.join(basename, input_sound_path)
            yth.download_audio(input_audio_video_url, input_sound_path)
        else:
            ext = input_audio_video_url.split(".")[-1].lower()
            basename = input_audio_video_url.split("/")[-1].split(".")[0]
            name = basename
            basename = osh.asciistring(basename)
            osh.make_directory(basename)
            input_file_path = f"file.{ext}"
            input_file_path = osh.join(basename, input_file_path)
            osh.download_file(input_audio_video_url, input_file_path)
            input_sound_path = f"audio.mp3"
            input_sound_path = osh.join(basename, input_sound_path)
            ah.sound_converter(input_file_path, input_sound_path)
    else:
        # File
        osh.checkfile(input_audio_video_url, f"Invalid input: {input_audio_video_url}")
        folder, base, ext = osh.folder_name_ext(input_audio_video_url)
        name = base
        folder = osh.join(folder, base)
        osh.make_directory(folder)
        input_sound_path = f"audio.mp3"
        input_sound_path = osh.join(folder, input_sound_path)
        if vh.is_valid_video_file(input_audio_video_url):
            ah.sound_converter(input_audio_video_url, input_sound_path)
        elif ah.is_valid_audio_file(input_audio_video_url):
            ah.sound_converter(input_audio_video_url, input_sound_path)
        else:
            osh.error(f"Invalid input: {input_audio_video_url}")

    osh.check(ah.is_valid_audio_file(input_sound_path), f"Invalid sound file:\n\t{input_sound_path}")
    folder, base, ext = osh.folder_name_ext(input_sound_path)

    output_file = args.output_file
    if osh.emptystring(output_file):
        output_file = f"{folder}/{base}-diarization.json"


    # Load credentials from JSON files if provided
    sftp_cred = None
    if not(args.disable_sftp):
        sftp_cred = diarization_helper.sftp_credentials(args.sftp_cred)

    gladia_cred = None
    if not(args.disable_gladia):
        gladia_cred = diarization_helper.gladia_credentials(args.gladia_cred)

    pyannote_cred = None
    if not(args.disable_pyannote):
        pyannote_cred = diarization_helper.pyannote_credentials(args.pyannote_cred)

    device = args.device

    if not(args.disable_source_separation):
        folder_sources = osh.join(folder, "sources")
        osh.make_directory(folder_sources)
        sources = ah.separate_sources(
            input_sound_path,
            output_folder=folder_sources,
            device = device, # or "cuda" if GPU or nothing to let it decide
            nb_workers = 4, # ignored if not cpu
            output_format = "mp3",
        )
        input_sound_path = sources["vocals"]




    output_file = diarization_helper.diarization(
        sound_path=input_sound_path,
        output_file=output_file,
        sftp_cred=sftp_cred,
        gladia_cred=gladia_cred,
        pyannote_cred=pyannote_cred,
        lang=lang,
        custom_vocabulary=args.custom_vocabulary
    )

    with open(output_file, "rt") as fin:
        data = json.load(fin)
        words = data["words"]
        sentences = data["sentences"]
        pure_diarization = data["diarization"]

    
    colors = ["#FF6961", "#FFB340", "#02D46A", "#0C817B", "#007AFF", "#5856D6", "#BF40BF", "#FFB6C1"]

    ascii_name = osh.asciistring(name)
    o = osh.join(folder, f"{ascii_name}-words.json")
    with open(o, "wt") as fout:
        words = sorted(words, key=lambda x: x["start"])
        json.dump(words, fout, indent=2)

    o = osh.join(folder, f"{ascii_name}-sentences.json")
    with open(o, "wt") as fout:
        sentences = sorted(sentences, key=lambda x: x["start"])
        json.dump(sentences, fout, indent=2)

    o = osh.join(folder, f"{ascii_name}-diarization.json")
    with open(o, "wt") as fout:
        pure_diarization = sorted(pure_diarization, key=lambda x: x["start"])
        json.dump(pure_diarization, fout, indent=2)

    # Donut breaths
    res, breath_segments = diarization_helper.words2breath_segments(words)
    o = osh.join(folder, f"{ascii_name}-breath-segments.json")
    with open(o, "wt") as fout:
        breath_segments = sorted(breath_segments, key=lambda x: x["start"])
        json.dump(breath_segments, fout, indent=2)

    diarization_helper.elbow_visu(res["x"], res["y"], res, base)


    speakers_order = list(set([d["speaker"] for d in words]))
    times = {s: sum(seg["end"] - seg["start"] for seg in words if seg["speaker"] == s) for s in speakers_order}
    speakers_order = sorted(speakers_order, key=lambda x: times[x], reverse=True)

    angel_hair = []

    donuts = []

    f = folder.split(os.sep)[-1]

    # Donut words
    i = osh.join(folder, f"{ascii_name}-words.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "words"
    o = f"{f}-{base}-donut-{title}.json"
    o = osh.join(folder, o)
    o, url_words = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
    donuts.append(url_words)

    # Donut breath_segments
    i = osh.join(folder, f"{ascii_name}-breath-segments.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "elbow"
    o = f"{f}-{base}-donut-{title}.json"
    o = osh.join(folder, o)
    o, url_elbow = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
    donuts.append(url_elbow)

    # Donut sentences
    i = osh.join(folder, f"{ascii_name}-sentences.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "sentences"
    o = f"{f}-{base}-donut-{title}.json"
    o = osh.join(folder, o)
    o, url_sentences = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
    donuts.append(url_sentences)

    # Donut diariation
    i = osh.join(folder, f"{ascii_name}-diarization.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "pyannote"
    o = f"{f}-{base}-donut-{title}.json"
    o = osh.join(folder, o)
    o, url_pure_diarization = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
    donuts.append(url_pure_diarization)



    # Angel Hair
    i = osh.join(folder, f"{ascii_name}-words.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "words"
    o = f"{f}-{base}-angel-hair-{title}.json"
    o = osh.join(folder, o)
    o, url_words = diarization_helper.make_angel_hair_pasta_image(
        source,
        speakers_order,
        o=o,
        title = f"{title} for {name}",
        sftp_cred=sftp_cred
    )
    angel_hair.append(url_words)

    i = osh.join(folder, f"{ascii_name}-breath-segments.json")
    with open(i, "rt") as fin:
        source = json.load(fin)
    title = "elbow"
    o = f"{f}-{base}-angel-hair-{title}.json"
    o = osh.join(folder, o)
    o, url_breath_segments = diarization_helper.make_angel_hair_pasta_image(
        source,
        speakers_order,
        o=o,
        title = f"{title} for {name}",
        sftp_cred=sftp_cred
    )
    angel_hair.append(url_breath_segments)

    source = sentences
    title = "sentences"
    o = f"{f}-{base}-angel-hair-{title}.json"
    o = osh.join(folder, o)
    o, url_breath_segments = diarization_helper.make_angel_hair_pasta_image(
        source,
        speakers_order,
        o=o,
        title = f"{title} for {name}",
        sftp_cred=sftp_cred
    )
    angel_hair.append(url_breath_segments)

    source = pure_diarization
    title = "pyannote"
    o = f"{f}-{base}-angel-hair-{title}.json"
    o = osh.join(folder, o)
    o, url_pure_diarization = diarization_helper.make_angel_hair_pasta_image(
        source,
        speakers_order,
        o=o,
        title = f"{title} for {name}",
        sftp_cred=sftp_cred
    )
    angel_hair.append(url_pure_diarization)

    for d in donuts:
        osh.info(d)

    for a in angel_hair:
        osh.info(a)




