"""
Gladia Helper

This code handles the process of uploading an audio file to Gladia for transcription and retrieving
the transcription result through asynchronous API calls. It supports diarization, named entity recognition,
and structured data extraction, among other features.

Dependencies:
- `os_helper` (osh): Helper functions for file operations.
- `audio_helper` (ah): Helper functions for audio processing.
- `requests`: For handling HTTP requests to Gladia's API.
- `json`: For parsing and handling JSON responses.
- `time`: For implementing delays during the transcription polling process.
 
Ensure that the modules `os_helper` and `audio_helper` are available in your Python environment.

Authors:
- Warith Harchaoui, https://harchaoui.org/warith
"""

from collections import OrderedDict
import os_helper as osh
import audio_helper as ah
import requests
import json
import time
from typing import List, Optional, Union, Dict
import numpy as np
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed


def credentials(config_path: Optional[str] = None) -> Dict[str, Union[str, int, List[str]]]:
    """
    Retrieve Gladia credentials from a configuration file or environment variables.

    This function loads Gladia credentials from a given config path (file or folder).
    It expects certain mandatory keys in the configuration file.

    Parameters
    ----------
    config_path : Optional[str]
        Path to the configuration file or folder containing the Gladia credentials.
        If None, it checks environment variables with keys in capitals.

    Returns
    -------
    Dict[str, Union[str, int, List[str]]]
        A dictionary containing the Gladia credentials and configurations.
    """
    keys = [
        "gladia_key",
        "gladia_url",
        "gladia_upload_url",
        "custom_vocabulary",
        "callback_url",
        "max_number_of_speakers",
        "custom_spelling",
        "speed_factor",
        "min_waiting_time_per_iteration",
    ]
    return osh.get_config(keys, "gladia", config_path)


def gladia_processing(
    sound_path: str,
    gladia_cred: Dict[str, Union[str, int, List[str]]],
    output_file: Optional[str] = None,
    lang: Union[str, List[str]] = "",
    custom_vocabulary: Optional[List[str]] = None,
    num_workers: int = 1,
    overwrite: bool = True,
    
) -> Optional[str]:
    """
    Handles the complete transcription process for an audio file using the Gladia API.

    Parameters
    ----------
    sound_path : str
        Path to the audio file to be transcribed.
    gladia_cred : Dict[str, Union[str, int, List[str]]]
        Gladia credentials for API access.
    output_file : Optional[str], default None
        Path to save the transcription result as a JSON file.
    lang : Union[str, List[str]], optional
        Language code(s) for the transcription. Can be a single language code or a list for code-switching.
    custom_vocabulary : Optional[List[str]], default None
        List of custom words to be recognized by the transcription engine.
    num_workers : int, default 1
        Number of parallel workers to use for processing.
    overwrite : bool, default True
        Overwrite the output file if it already exists.

    Returns
    -------
    Optional[str]
        Path to the transcription result JSON file, or None if an error occurred.
    """
    if custom_vocabulary is None:
        custom_vocabulary = []

    # Check if the audio file exists
    osh.checkfile(sound_path, msg=f"Audio file not found at {sound_path}")

    # Determine output file path
    if osh.emptystring(output_file):
        folder, _, _ = osh.folder_name_ext(sound_path)
        output_file = f"{folder}/gladia-result.json"
    else:
        folder, _, _ = osh.folder_name_ext(output_file)

    # Avoid reprocessing if output file already exists
    if not(overwrite) and osh.file_exists(output_file):
        osh.info(f"Gladia processing already done for {sound_path}:\n\t{output_file}")
        return output_file

    # Get duration of the audio file
    duration = ah.get_audio_duration(sound_path)
    split_time = gladia_cred.get("split_time", 30*60)  # Default split time is 30 minutes if not provided

    if duration > split_time:
        # If the audio is longer than split_time, split it into chunks
        temporary_chunks_folder = f"{folder}/gladia-chunks"
        osh.make_directory(temporary_chunks_folder)
        audio_chunks = ah.split_audio_regularly(sound_path, temporary_chunks_folder, split_time * 0.5)

        s = "\n\t".join(audio_chunks)
        osh.info(f"Audio file split into chunks:\n\t{sound_path} to\n\t{s}")
        analysis_chunks = []

        w = min(num_workers, len(audio_chunks))
        if w<=1:
            for i, a_chunk in enumerate(audio_chunks):
                o = f"{temporary_chunks_folder}/gladia-result-chunk-{i:04d}.json"
                # Recursively process each chunk
                gladia_processing(a_chunk, gladia_cred, output_file=o, lang=lang, custom_vocabulary=custom_vocabulary, num_workers=1)
                analysis_chunks.append(o)
        else:
            # Parallelize processing of each chunk
            osh.info(f"Parallelization: {w} workers")
            with ThreadPoolExecutor(max_workers=w) as executor:
                future_to_chunk = {
                    executor.submit(gladia_processing, a_chunk, gladia_cred, f"{temporary_chunks_folder}/gladia-result-chunk-{i:04d}.json", lang, custom_vocabulary, num_workers=1): a_chunk
                    for i, a_chunk in enumerate(audio_chunks)
                }
                
                for future in as_completed(future_to_chunk):
                    try:
                        result_file = future.result()
                        analysis_chunks.append(result_file)
                    except Exception as exc:
                        osh.error(f"Error processing chunk {future_to_chunk[future]}: {exc}")


        # Merge the results from the chunks
        return _merge_chunk_results(output_file, audio_chunks, analysis_chunks, gladia_cred, lang, custom_vocabulary)

    else:
        # Upload audio to Gladia server
        try:
            with open(sound_path, "rb") as fin:
                files = {"audio": (sound_path, fin, "audio/mpeg")}
                response = requests.post(
                    gladia_cred["gladia_upload_url"],
                    headers={"x-gladia-key": gladia_cred["gladia_key"]},
                    files=files,
                )
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            s  = pprint(gladia_cred)
            osh.error(f"Network error during audio upload: {e}\n\t{s}")

        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            osh.error(f"Error decoding JSON response during audio upload: {e}")

        # Retrieve the uploaded audio URL
        gladia_audio_upload_url = response_json.get("audio_url")
        if gladia_audio_upload_url is None:
            osh.error(f"Failed to upload audio to Gladia: {response_json}")

        osh.info(f"Audio uploaded to Gladia: {gladia_audio_upload_url}")

        # Prepare payload for transcription request

        # 2. Request transcription from Gladia
        payload = {
            "context_prompt": gladia_cred["context_prompt"],
            "subtitles": True,
            "subtitles_config": {"formats": ["srt"]},
            "diarization": True,
            "diarization_config": {
                "min_speakers": 1,
                "max_speakers": gladia_cred["max_number_of_speakers"],
            },
            "named_entity_recognition": True,
            "chapterization": True,
            "name_consistency": True,
            "structured_data_extraction": True,
            "structured_data_extraction_config": {"classes": ["person", "organization"]},
            "sentiment_analysis": True,
            "sentences": True,
            "display_mode": True,
            "audio_url": gladia_audio_upload_url,
        }

        # Handle custom vocabulary
        custom_vocabulary = list(custom_vocabulary) + gladia_cred["custom_vocabulary"]
        # custom_vocabulary = nlp_helper.filter_out_bad_keywords_list(custom_vocabulary)
        # custom_vocabulary = sorted(list(set(custom_vocabulary)))
        # if len(custom_vocabulary) > 0:
        #     payload["custom_vocabulary"] = custom_vocabulary
        #     s = "\n\t".join(custom_vocabulary)
        #     os_helper.info(f"Custom vocabulary added to Gladia transcription:\n\t{s}")

        code_switching = gladia_cred.get("code_switching", False)

        # Handle language detection with multiple languages switching scenarios
        payload["detect_language"] = code_switching # seems always useful
        payload["enable_code_switching"] = code_switching # same
        if not(osh.emptystring(lang)):
            main_language = None
            other_languages = []
            if isinstance(lang, list) and len(lang) > 0:
                main_language = lang[0]
                if len(lang) > 1:
                    other_languages = lang[1:]
                    other_languages = [l for l in other_languages if l != main_language]
            elif isinstance(lang, str):
                main_language = lang

            osh.check(
                main_language is not None,
                msg=f"Language parameter should be a string or a list of strings: {lang}",
            )
            payload["language"] = main_language
            osh.info(f"Language set to {main_language}")

            language_details = {
                "language": main_language
            }
            osh.info(f"Main language: {main_language}")
            if len(other_languages) > 0:
                language_details["code_switching_config"]={
                    "languages": other_languages
                }
                osh.info(f"Code switching enabled with languages: {other_languages}")

            payload.update(language_details)

            
        # Handle custom spelling
        if "custom_spelling" in gladia_cred and len(gladia_cred["custom_spelling"]) > 0:
            spelling_dictionary = gladia_cred["custom_spelling"]
            payload["custom_spelling"] = True
            payload["custom_spelling_config"] = {
                "spelling_dictionary": spelling_dictionary
            }
        else:
            payload["custom_spelling"] = False

        headers = {"Content-Type": "application/json",
                "x-gladia-key": gladia_cred["gladia_key"]}

        # Request transcription from Gladia
        try:
            response = requests.post(gladia_cred["gladia_url"], json=payload, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            s = pprint(payload)
            osh.error(f"Network error during transcription request: {e}\n\t{s}")

        try:
            transcription_json = response.json()
        except json.JSONDecodeError as e:
            osh.error(f"Error decoding JSON response during transcription request: {e}")

        # Retrieve result URL and transcription ID
        gladia_audio_result_url = transcription_json.get("result_url")
        if gladia_audio_result_url is None:
            osh.error(f"Failed to request transcription from Gladia: {transcription_json}")

        transcription_id = transcription_json.get("id")
        if transcription_id is None:
            osh.error(f"Transcription ID not found in response: {transcription_json}")

        # Polling for the transcription result
        res = _poll_for_gladia_result(gladia_audio_result_url, gladia_cred, transcription_id, sound_path)
        if res is None:
            osh.error("Failed to get transcription result from Gladia.")

        # Extract sentences from the result
        sentences = res.get("result", {}).get("transcription", {}).get("sentences", [])

        # Save final transcription result to output file
        with open(output_file, "wt", encoding="utf8") as fout:
            json.dump(sentences, fout, indent=2)

    return output_file


def _poll_for_gladia_result(
    gladia_audio_result_url: str,
    gladia_cred: Dict[str, Union[str, int, List[str]]],
    transcription_id: str,
    sound_path: str,
) -> Optional[dict]:
    """
    Polls Gladia for transcription result and logs statuses during the wait.

    Parameters
    ----------
    gladia_audio_result_url : str
        URL to the Gladia transcription result.
    gladia_cred : Dict[str, Union[str, int, List[str]]]
        Gladia credentials used for API requests.
    transcription_id : str
        Unique ID of the transcription job.
    sound_path : str
        Path to the audio file that was transcribed.

    Returns
    -------
    Optional[dict]
        Transcription result from Gladia, or None if an error occurred.
    """
    duration = ah.get_audio_duration(sound_path)
    estimated_gladia_time = duration * gladia_cred["speed_factor"]
    status = "processing"
    waiting_time = gladia_cred["min_waiting_time_per_iteration"]
    estimated_iterations = 1 + round(estimated_gladia_time / waiting_time)
    counter = 0
    res = None
    while status != "done":
        counter += 1
        try:
            response = requests.get(
                f"{gladia_audio_result_url}?id={transcription_id}",
                headers={
                    "Content-Type": "application/json",
                    "x-gladia-key": gladia_cred["gladia_key"]
                },
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            osh.error(f"Network error during polling: {e}")

        try:
            res = response.json()
        except json.JSONDecodeError as e:
            osh.error(f"Error decoding JSON response during polling: {e}")

        status = res.get("status", "")
        osh.info(f"Gladia processing status: {status}, iteration {counter}/{estimated_iterations}")
        if status != "done":
            time.sleep(waiting_time)  # Wait before checking again

    # Return the result after processing is done
    return res


def _merge_chunk_results(
    output_file: str,
    audio_chunks: List[str],
    analysis_chunks: List[str],
    gladia_cred: Dict[str, Union[str, int, List[str]]],
    lang: Union[str, List[str]],
    custom_vocabulary: List[str],
    overwrite: bool = True,
) -> str:
    """
    Merge the results of multiple chunks into a single output file with consistent speaker labels.
    This approach concatenates the longest speaker segments across chunks and creates synthetic audio.
    The diarization problem is thus simplified to a single and shorter audio file with consistent speaker labels.
    Indeed, this adds robustness for long audio especially when somebody speaks at the very beginning, is silent in the middle, and finally speaks at the end.

    Parameters
    ----------
    output_file : str
        Output file for the merged results (same format as gladia_processing).
    audio_chunks : List[str]
        List of audio chunks (mp3 files) to merge.
    analysis_chunks : List[str]
        List of analysis chunks (json files) to merge (from gladia_processing).
    gladia_cred : Dict[str, Union[str, int, List[str]]]
        Credentials for the Gladia API (see credentials function in gladia_helper)
    lang : Union[str, List[str]]
        Language code(s) for the audio file(s).
    custom_vocabulary : List[str]
        Custom vocabulary for the Gladia API.
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default True
    
    Returns
    -------
    str
        Path to the output file with merged results.

    """
    if not(overwrite) and osh.exists(output_file):
        return output_file
    
    folder, sound_basename, _ = osh.folder_name_ext(output_file)
    time_cursor = 0
    silent_time = 3  # Seconds of silence between segments in synthetic audio

    # Step 1: Gather speaker segments across chunks (max 6 seconds)
    max_segment_duration = 6.0
    speaker_segments = OrderedDict()  # Holds the longest segment for each speaker

    for i, (audio_chunk, analysis_chunk) in enumerate(zip(audio_chunks, analysis_chunks)):
        with open(analysis_chunk, "rt", encoding="utf8") as fin:
            chunk_results = json.load(fin)
            for sentence in chunk_results:
                name = f"{i}_{sentence['speaker']}"
                start, end = sentence["start"], sentence["end"]
                duration = end - start

                # Take the heart of the segment and limit its duration at max_segment_duration
                middle = 0.5 * (start + end) 
                start = max(start, middle - 0.5 * max_segment_duration)
                end = min(end, middle + 0.5 * max_segment_duration)
                duration = end - start

                # Track the longest segment per speaker
                if name in speaker_segments:
                    if duration > speaker_segments[name][1] - speaker_segments[name][0]:
                        speaker_segments[name] = (start, end, audio_chunk)
                else:
                    speaker_segments[name] = (start, end, audio_chunk)

    # Step 2: Create synthetic audio with representative segments for each speaker
    synthetic_audio_segments = []
    synthetic_diarization = []
    time_cursor = 0
    for i, name in enumerate(speaker_segments):
        start, end, audio_chunk = speaker_segments[name]
        segment_path = f"{folder}/synthetic_segment_{i}.mp3"
        ah.extract_audio_chunk(audio_chunk, start, end, segment_path, overwrite=True)
        synthetic_audio_segments.append(segment_path)
        d = {
            "start": time_cursor,
            "end": time_cursor + end - start,
            "speaker": name,
            "word": segment_path
        }
        synthetic_diarization.append(d)

        # Add silent segment for separation
        silence_path = f"{folder}/silence_{i}.mp3"
        ah.generate_silent_audio(silent_time, output_audio_filename=silence_path)
        synthetic_audio_segments.append(silence_path)

        time_cursor += end - start + silent_time

    synthetic_diarization = sorted(synthetic_diarization, key=lambda x: x["start"])

    # Concatenate all segments into one synthetic audio file
    synthetic_audio = f"{folder}/synthetic_audio.mp3"
    ah.audio_concatenation(synthetic_audio_segments, synthetic_audio, overwrite=True)

    # Step 3: Process the synthetic audio to get consistent speaker labels in order to reconcile with the original diarization
    synthetic_output_file = f"{folder}/synthetic_output.json"
    gladia_processing(synthetic_audio, gladia_cred, output_file=synthetic_output_file, lang=lang, custom_vocabulary=custom_vocabulary)

    with open(synthetic_output_file, "rt", encoding="utf8") as fin:
        synthetic_results = json.load(fin)
        for s in synthetic_results:
            s["speaker"] = str(s["speaker"])
        synthetic_results = sorted(synthetic_results, key=lambda x: x["start"])
    
    # Map old speaker names to new ones from synthetic results
    initial_names = sorted({str(sentence["speaker"]) for sentence in synthetic_diarization})
    synthetic_speakers = sorted({str(sentence["speaker"]) for sentence in synthetic_results})

    R, C = len(initial_names), len(synthetic_speakers)
    overlap_matrix = np.zeros((R, C))

    initial_names2id = {name: i for i, name in enumerate(initial_names)}    
    synthetic_speakers2id = {name: i for i, name in enumerate(synthetic_speakers)}

    for sd, sr in product(synthetic_diarization, synthetic_results):
        i = initial_names2id[sd["speaker"]]
        j = synthetic_speakers2id[sr["speaker"]]
        si, ei = sd["start"], sd["end"]
        start, end = sr["start"], sr["end"]
        intersection = max(0, min(ei, end) - max(si, start))
        overlap_matrix[i, j] += intersection

    ok = all([np.sum(overlap_matrix[:,j])>0 for j in range(C)])
    ok = ok and all([np.sum(overlap_matrix[i,:])>0 for i in range(R)])
    osh.check(ok, "Bad overlap")
            
    oa = osh.join(folder, sound_basename)
    osh.make_directory(oa)
    oa = osh.join(folder, sound_basename, "M.txt")
    np.savetxt(oa, overlap_matrix)
    ob = osh.join(folder, sound_basename, "initial_names.txt")
    np.savetxt(ob, np.array(initial_names, dtype=str), fmt="%s")
    oc = osh.join(folder, sound_basename, "synthetic_speakers.txt")
    np.savetxt(oc, np.array(synthetic_speakers, dtype=str), fmt="%s")
    

    # Assign new names based on the largest overlap
    initial_to_synthetic_names = {}
    for i in range(R):
        j = np.argmax(overlap_matrix[i, :])
        initial_to_synthetic_names[initial_names[i]] = synthetic_speakers[j]

    # Step 4: Adjust all original chunks with new speaker mappings and save merged output
    res = []
    time_cursor = 0
    for i, (audio_chunk, analysis_chunk) in enumerate(zip(audio_chunks, analysis_chunks)):
        duration = ah.get_audio_duration(audio_chunk)
        with open(analysis_chunk, "rt", encoding="utf8") as fin:
            chunk_results = json.load(fin)
            for sentence in chunk_results:
                original_name = f"{i}_{sentence['speaker']}"
                osh.check(original_name in initial_to_synthetic_names, f"Missing speaker {original_name}")
                sentence["speaker"] = initial_to_synthetic_names.get(original_name, sentence["speaker"])
                # Adjust timings for the overall timeline
                sentence["start"] += time_cursor
                sentence["end"] += time_cursor
                for word in sentence.get("words", []):
                    word["start"] += time_cursor
                    word["end"] += time_cursor
                    word["speaker"] = sentence["speaker"]
                res.append(sentence)
        time_cursor += duration + silent_time
    res = sorted(res, key=lambda x: x["start"])

    # Save final merged results
    with open(output_file, "wt", encoding="utf8") as fout:
        json.dump(res, fout, indent=2)
    osh.info(f"Saved merged results to:\n\t{output_file}")

    return output_file


if __name__ == "__main__":
    """
    Example usage of the gladia_processing function.
    Ensure that you have the necessary credentials set up in your configuration.
    """

    # Example audio file path
    sound_path = "path/to/your/audiofile.mp3"

    # Load Gladia credentials
    gladia_cred = credentials()

    # Optional: specify output file, language, and custom vocabulary
    output_file = "path/to/output/gladia-result.json"
    lang = "en"
    custom_vocabulary = ["CustomWord1", "CustomWord2"]

    # Process the audio file
    result = gladia_processing(
        sound_path,
        gladia_cred,
        output_file=output_file,
        lang=lang,
        custom_vocabulary=custom_vocabulary
    )

    if result:
        print(f"Transcription result saved to {result}")
    else:
        print("Transcription failed.")
