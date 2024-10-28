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
import os
from diarization_helper import detect_language, extract_entities, detect_keywords, clean_text
from functools import partial

from diarization_helper.donut import make_donut_image
import os_helper as osh
import audio_helper as ah
import diarization_helper
import streamlit as st
import yt_helper as yth

import os_helper as osh

import numpy as np
import sftp_helper as sftph

import re

def extract_video_id(youtube_url):
    # Regular expression to match different YouTube URL patterns
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|.+\?v=)|youtu\.be\/)([^&\n?#]+)"
    match = re.search(pattern, youtube_url)
    
    # Return the video ID if found, else None
    return match.group(1) if match else None

# Function to embed YouTube with JavaScript to control start time
def display_youtube_video(place_holder, youtube_url, time, height=400):
    video_id = extract_video_id(youtube_url)
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    start_time = round(time)
    place_holder.empty()

    # HTML code with JavaScript to set the video start time dynamically
    html_code = f"""
    <iframe id="ytplayer" type="text/html" width="700" height="400"
    src="{embed_url}?start={start_time}&autoplay=1&enablejsapi=1"
    frameborder="0"></iframe>
    <script>
    var player;
    function onYouTubeIframeAPIReady() {{
        player = new YT.Player('ytplayer', {{
            events: {{
                'onReady': function(event) {{
                    event.target.seekTo({start_time}, true);
                    event.target.playVideo();
                }}
            }}
        }});
    }}
    </script>
    <script src="https://www.youtube.com/iframe_api"></script>
    """
    # Display the HTML in Streamlit
    with place_holder:
        st.components.v1.html(html_code, height=height)

if __name__ == '__main__':


    st.set_page_config(layout="wide", page_title="heedgiTV", page_icon="logo.png")

    title = "heedgiTV"

    col1, col2 = st.columns([1,3])
    with col1:
        st.image("logo.png", width=200)
    with col2:
        title_component = st.title(title)

    verbosity = 2
    osh.verbosity(verbosity)

    # Choose video url
    video_url = st.text_input("Enter Youtube URL:", "")

    st.info("TODO: thread for progress bar (stqdm animation)")
    st.info("TODO: integrate faces for named fingerprints")
    st.info("TODO: finish MFCC approach for overlaps between speakers")

    if not("cred" in st.session_state) or not st.session_state["cred"]:
        sftp_cred = sftph.credentials("config/sftp")
        gladia_cred = diarization_helper.gladia_credentials("config/gladia")
        pyannote_cred = diarization_helper.pyannote_credentials("config/pyannote")
        st.session_state["cred"] = True
        st.session_state["sftp_cred"] = sftp_cred
        st.session_state["gladia_cred"] = gladia_cred
        st.session_state["pyannote_cred"] = pyannote_cred
    else:
        sftp_cred = st.session_state["sftp_cred"]
        gladia_cred = st.session_state["gladia_cred"]
        pyannote_cred = st.session_state["pyannote_cred"]

    archive_path = "archives"
    folder = archive_path
    osh.make_directory(folder)

    source_separation = False



    if "processing" not in st.session_state:
        st.session_state["processing"] = False

    if "rename" not in st.session_state:
        st.session_state["rename"] = {}

    if st.button("Start Processing"):
        st.session_state["processing"] = True


    if st.session_state["processing"]:
        # check if the video url is valid
        if not yth.is_valid_video_url(video_url):
            st.error("Please enter a valid video URL.")
        else:
            if not("meta" in st.session_state) or not st.session_state["meta"]:
                metadata = yth.video_url_meta_data(video_url)
                title = metadata["title"]
                name = title
                description = metadata["description"]
                text = f"{title}\n\n{description}"
                lang = detect_language(text)
                basename = osh.asciistring(title)
                folder = osh.join(archive_path, basename)
                osh.make_directory(folder)
                st.session_state["meta"] = True
                st.session_state["metadata"] = metadata
                st.session_state["lang"] = lang
                st.session_state["folder"] = folder
                st.session_state["basename"] = basename
                st.session_state["name"] = name
                st.session_state["title"] = title
                st.session_state["description"] = description
            else:
                metadata = st.session_state["metadata"]
                lang = st.session_state["lang"]
                folder = st.session_state["folder"]
                basename = st.session_state["basename"]
                name = st.session_state["name"]
                title = st.session_state["title"]
                description = st.session_state["description"]


            if not("vocab" in st.session_state) or not st.session_state["vocab"]:
                # custom vocabulary
                custom_vocabulary = []
                text = clean_text(text)
                custom_vocabulary_unbroken_list = extract_entities(text, lang=lang, break_n_grams=False)
                custom_vocabulary += custom_vocabulary_unbroken_list
                keywords = detect_keywords(text, lang=lang, score=False, top_n=50)
                custom_vocabulary += keywords
                custom_vocabulary = sorted(list(set(custom_vocabulary)))
                st.session_state["vocab"] = True
                st.session_state["custom_vocabulary"] = custom_vocabulary
            else:
                custom_vocabulary = st.session_state["custom_vocabulary"]
                
            

            if "start_time" not in st.session_state:
                st.session_state["start_time"] = 0

            video_placeholder = st.empty()

            # Display initial video
            start_time = 0
            if "start_time" in st.session_state:
                start_time = st.session_state["start_time"]

            display_youtube_video(video_placeholder, video_url, start_time)
                

            # download thumbnail
            thumb_path = osh.join(folder, "thumb.png")
            if not osh.file_exists(thumb_path):
                yth.download_thumbnail(video_url, thumb_path)

            # download audio
            sound_path = osh.join(folder, "audio.mp3")
            if not osh.file_exists(sound_path):
                yth.download_audio(video_url, sound_path)
                osh.check(ah.get_audio_duration(sound_path), f"Bad download\n\t{video_url}\n\t{sound_path}")
                if source_separation:
                    chrono_separation = osh.tic()
                    f = osh.join(folder, "sources")
                    osh.make_directory(f)
                    sources = ah.separate_sources(sound_path, f)
                    sound_ext = "mp3"
                    # task = partial(ah.separate_sources, input_audio_file = sound_path, output_folder = f, output_format = sound_ext)
                    # estimated_duration = 1.0 * ah.get_audio_duration(sound_path) * 0.14
                    # progress_task = osh.ProgressTask(task=task, estimated_duration=estimated_duration, total_steps=100)
                    # progress_task.run_with_both()
                    i = sound_path
                    o = osh.join(folder, "original_audio.mp3")
                    osh.copyfile(i, o)
                    v = osh.join(f, f"vocals.{sound_ext}")
                    osh.copyfile(v, i)
                    chrono_separation = osh.toc(chrono_separation)
                    tt = osh.time2str(chrono_separation)
                    ss= osh.time2str(ah.get_audio_duration(sound_path))
                    st.info(f"Source Separation took {tt} for sound duration {ss}")


            num_workers = osh.get_nb_workers() - 1
            # num_workers = 1
            num_workers = max(1, num_workers)

            output_file = osh.join(folder, "diarization.json")
            output_file = diarization_helper.diarization(
                sound_path=sound_path,
                output_file=output_file,
                sftp_cred=sftp_cred,
                gladia_cred=gladia_cred,
                pyannote_cred=pyannote_cred,
                lang=lang,
                custom_vocabulary=custom_vocabulary,
                num_workers = num_workers,
                overwrite=False,
            )


            with open(output_file, "rt") as fin:
                data = json.load(fin)
                words = data["words"]
                sentences = data["sentences"]
                pure_diarization = data["diarization"]

            
            colors = ["#FF6961", "#FFB340", "#02D46A", "#0C817B", "#007AFF", "#5856D6", "#BF40BF", "#FFB6C1"]

            ascii_name = osh.asciistring(name)


            o = osh.join(folder, f"{ascii_name}-words.json")
            if not osh.file_exists(o):
                with open(o, "wt") as fout:
                    words = sorted(words, key=lambda x: x["start"])
                    json.dump(words, fout, indent=2)

            o = osh.join(folder, f"{ascii_name}-sentences.json")
            if not osh.file_exists(o):
                with open(o, "wt") as fout:
                    sentences = sorted(sentences, key=lambda x: x["start"])
                    json.dump(sentences, fout, indent=2)

            o = osh.join(folder, f"{ascii_name}-diarization.json")
            if not osh.file_exists(o):
                with open(o, "wt") as fout:
                    pure_diarization = sorted(pure_diarization, key=lambda x: x["start"])
                    json.dump(pure_diarization, fout, indent=2)

            res, breath_segments = diarization_helper.words2breath_segments(words)
            o = osh.join(folder, f"{ascii_name}-breath-segments.json")
            if not osh.file_exists(o):
                with open(o, "wt") as fout:
                    breath_segments = sorted(breath_segments, key=lambda x: x["start"])
                    json.dump(breath_segments, fout, indent=2)

            o = osh.join(folder, f"{ascii_name}-elbow.png")
            if not osh.file_exists(o):
                diarization_helper.elbow_visu(res["x"], res["y"], res, output_image=o)

            st.image(o)

            if ("anonymous" not in st.session_state) or (st.session_state["anonymous"]):
                st.session_state["anonymous"] = True
                speakers_order = sorted(list(set([d["speaker"] for d in breath_segments])))
                speaker2timestamp = {}
                for s in speakers_order:
                    bits = [seg for seg in breath_segments if seg["speaker"] == s]
                    biggest_bit = max(bits, key=lambda x: x["end"] - x["start"])
                    middle = (biggest_bit["end"] + biggest_bit["start"]) / 2
                    t = max(biggest_bit["start"], middle)
                    speaker2timestamp[s] = t

            f = basename

            st.header("📇 Rename Speakers")
            rename_buttons = []

            # Render text inputs for each speaker inside the form
            col_name_1, col_name_2 = st.columns(2)
            for i, speaker in enumerate(speakers_order):
                c = col_name_1 if i % 2 == 0 else col_name_2
                with c:
                    t = speaker2timestamp[speaker]
                    tt = osh.time2str(t)
                    rename_buttons.append(st.text_input(f"Rename speaker {str(i)}", str(i)))
                    if st.button(f"⏯️  Speaker {str(i)}: {tt}"):
                        st.session_state["start_time"] = t

            if st.button(f"Submit Names"):
                if any([osh.emptystring(rename) for rename in rename_buttons]):
                    st.error("Please enter a name for each speaker.")
                st.session_state["rename"] = {
                    str(i): rename_buttons[i] for i, speaker in enumerate(speakers_order)
                }
                st.session_state["anonymous"] = False

            if not(st.session_state["anonymous"]):
                # Render the speakers with their new names
                if "rename" in st.session_state and len(st.session_state["rename"]) == len(speakers_order):
                    old_names = sorted(list(set([d["speaker"] for d in breath_segments])))
                    new_names = [st.session_state["rename"][old_name] for old_name in old_names]
                    conversion = dict(zip(old_names, new_names))
                    conversion.update(dict(zip(new_names, new_names)))
                    for j in range(len(words)):
                        words[j]["speaker"] = conversion[words[j]["speaker"]]
                    for j in range(len(sentences)):
                        sentences[j]["speaker"] = conversion[sentences[j]["speaker"]]
                    for j in range(len(pure_diarization)):
                        pure_diarization[j]["speaker"] = conversion[pure_diarization[j]["speaker"]]
                    for j in range(len(breath_segments)):
                        breath_segments[j]["speaker"] = conversion[breath_segments[j]["speaker"]]

                    o = osh.join(folder, f"{ascii_name}-words-renamed.json")
                    with open(o, "wt") as fout:
                        words = sorted(words, key=lambda x: x["start"])
                        json.dump(words, fout, indent=2)

                    o = osh.join(folder, f"{ascii_name}-sentences-renamed.json")
                    with open(o, "wt") as fout:
                        sentences = sorted(sentences, key=lambda x: x["start"])
                        json.dump(sentences, fout, indent=2)

                    o = osh.join(folder, f"{ascii_name}-diarization-renamed.json")
                    with open(o, "wt") as fout:
                        pure_diarization = sorted(pure_diarization, key=lambda x: x["start"])
                        json.dump(pure_diarization, fout, indent=2)

                    o = osh.join(folder, f"{ascii_name}-breath-segments-renamed.json")
                    with open(o, "wt") as fout:
                        breath_segments = sorted(breath_segments, key=lambda x: x["start"])
                        json.dump(breath_segments, fout, indent=2)


                    speakers_order = list(set([d["speaker"] for d in sentences]))
                    times = {s: sum(seg["end"] - seg["start"] for seg in sentences if seg["speaker"] == s) for s in speakers_order}
                    speakers_order = sorted(speakers_order, key=lambda x: times[x], reverse=True)


                    donuts = []
                    angel_hair = []
                    donut_name = []

                    # Donut words
                    i = osh.join(folder, f"{ascii_name}-words-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "words"
                    o = f"{ascii_name}-donut-{title}.json"
                    o = osh.join(folder, o)
                    o, url_words = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
                    donuts.append(url_words)
                    donut_name.append(title)

                    # Donut breath_segments
                    i = osh.join(folder, f"{ascii_name}-breath-segments-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "elbow"
                    o = f"{ascii_name}-donut-{title}.json"
                    o = osh.join(folder, o)
                    o, url_elbow = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
                    donuts.append(url_elbow)
                    donut_name.append(title)

                    # Donut sentences
                    i = osh.join(folder, f"{ascii_name}-sentences-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "sentences"
                    o = f"{ascii_name}-donut-{title}.json"
                    o = osh.join(folder, o)
                    o, url_sentences = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
                    donuts.append(url_sentences)
                    donut_name.append(title)

                    # Donut diariation
                    i = osh.join(folder, f"{ascii_name}-diarization-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "pyannote"
                    o = f"{ascii_name}-donut-{title}.json"
                    o = osh.join(folder, o)
                    o, url_pure_diarization = make_donut_image(source, o=o, title = f"{title} for {name}", sftp_cred=sftp_cred, speakers=speakers_order)
                    donuts.append(url_pure_diarization)
                    donut_name.append(title)





                    # Angel Hair
                    i = osh.join(folder, f"{ascii_name}-words-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "words"
                    o = f"{ascii_name}-angel-hair-{title}.json"
                    o = osh.join(folder, o)
                    o, url_words = diarization_helper.make_angel_hair_pasta_image(
                        source,
                        speakers_order,
                        o=o,
                        title = f"{title} for {name}",
                        sftp_cred=sftp_cred
                    )
                    angel_hair.append(url_words)

                    i = osh.join(folder, f"{ascii_name}-breath-segments-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "elbow"
                    o = f"{ascii_name}-angel-hair-{title}.json"
                    o = osh.join(folder, o)
                    o, url_breath_segments = diarization_helper.make_angel_hair_pasta_image(
                        source,
                        speakers_order,
                        o=o,
                        title = f"{title} for {name}",
                        sftp_cred=sftp_cred
                    )
                    angel_hair.append(url_breath_segments)

                    i = osh.join(folder, f"{ascii_name}-sentences-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "sentences"
                    o = f"{ascii_name}-angel-hair-{title}.json"
                    o = osh.join(folder, o)
                    o, url_breath_segments = diarization_helper.make_angel_hair_pasta_image(
                        source,
                        speakers_order,
                        o=o,
                        title = f"{title} for {name}",
                        sftp_cred=sftp_cred
                    )
                    angel_hair.append(url_breath_segments)

                    i = osh.join(folder, f"{ascii_name}-diarization-renamed.json")
                    with open(i, "rt") as fin:
                        source = json.load(fin)
                    title = "pyannote"
                    o = f"{ascii_name}-angel-hair-{title}.json"
                    o = osh.join(folder, o)
                    o, url_pure_diarization = diarization_helper.make_angel_hair_pasta_image(
                        source,
                        speakers_order,
                        o=o,
                        title = f"{title} for {name}",
                        sftp_cred=sftp_cred
                    )
                    angel_hair.append(url_pure_diarization)            

                    for a,d,n in zip(angel_hair, donuts, donut_name):
                        st.markdown(f"[🔗  Donut {n}]({d})")
                        st.components.v1.iframe(d, width=2*800, height=600 , scrolling=True)
                        st.markdown(f"[🔗  Angel Hair Pasta {n}]({a})")
                        st.components.v1.iframe(a, width=3*800, height=600 // 2, scrolling=True)


                    # Example: Create a ZIP file of all the results
                    zip_file = folder.rstrip(os.sep) + ".zip"
                    result_folder = folder
                    osh.zip_folder(result_folder, zip_file)

                    zip_folder, zip_basename, zip_ext = osh.folder_name_ext(zip_file)
                    # Upload the ZIP file
                    remote_path = '%s/%s' % (sftp_cred["sftp_destination_path"], f"{zip_basename}.zip")
                    sftph.upload(zip_file, sftp_cred, remote_path)
                    url = f"{sftp_cred['sftp_https']}/{zip_basename}.zip"

                    st.header("🗜️ Download results")
                    st.markdown(f"[🔗 🗜️ All results]({url})")


                    # After generating the donuts and angel_hair lists

                    # Define HTML content for index.html
                    html_content = """
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>{title}</title>
                    </head>
                    <body>
                        <h1>{title}</h1>
                        {iframes}
                    </body>
                    </html>
                    """

                    iframes_html = ""
                    for d, a, n in zip(donuts, angel_hair, donut_name):
                        iframes_html += f"""
                        <h2>Donut {n}</h2>
                        <iframe src="{d}" width="1600" height="600" scrolling="yes"></iframe>
                        <h2>Angel Hair Pasta {n}</h2>
                        <iframe src="{a}" width="1600" height="300" scrolling="yes"></iframe>
                        <hr>
                        """

                    # Complete HTML with title and iframe content
                    html_content = html_content.format(title=name, iframes=iframes_html)

                    # Define file path for index.html
                    index_html_path = osh.join(folder, "index.html")

                    # Write HTML content to file
                    with open(index_html_path, "w") as f:
                        f.write(html_content)

                    # Upload the HTML file
                    remote_folder = "%s/%s" % (sftp_cred["sftp_destination_path"], ascii_name)
                    sftph.make_remote_directory(remote_folder, sftp_cred)
                    remote_path = '%s/%s' % (remote_folder, "index.html")
                    sftph.upload(index_html_path, sftp_cred, remote_path)
                    html_url = "/".join([sftp_cred['sftp_https'], ascii_name, "index.html"])

                    # Display the link to the user in Streamlit
                    st.header("🔗 View Results")
                    st.markdown(f"[View consolidated HTML report]({html_url})")
