"""
Pyannote Helper

This code handles the process of uploading an audio file to the Pyannote API for diarization,
retrieving the diarization result asynchronously, and processing large audio files by splitting them into chunks.

Features include:
- Handling audio file uploads to a remote server for processing.
- Support for diarization (speaker identification).
- Merging diarization results from split audio chunks for large files.
- Polling the Pyannote API for diarization results with progress updates.

Dependencies:
- `osh`: Helper functions for file operations, logging, and utilities.
- `audio_helper`: Helper functions for audio processing such as splitting and concatenation.
- `requests`: For handling HTTP requests to the Pyannote API.
- `json`: For parsing and handling JSON responses.
- `time`: For implementing delays during the result polling process.
- `sftp_helper`: For handling file upload/download to and from an SFTP server.

Authors:
- Warith Harchaoui, https://harchaoui.org/warith
"""


import os_helper as osh
from typing import Any, Dict, Optional
import audio_helper as ah
import requests
import sftp_helper as sftph
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def credentials(config_path: str = None) -> dict:
    """
    Retrieve pyannote credentials from a configuration file or environment variables.

    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file or folder (or None to load from environment variables).

    Returns
    -------
    dict
        A dictionary containing pyannote credentials.
    """
    keys = [
        "pyannote_key",
        "pyannote_key_name",
        "pyannote_url",
        "webhook_url",
        "speed_factor",
        "max_number_of_speakers",
        "approximated_number_iterations",
        "split_time",
        "min_waiting_time_per_iteration",
    ]
    return osh.get_config(keys, "pyannote", config_path)


def pyannote_processing(
    sound_path: str,
    sftp_cred: Dict[str, str],
    pyannote_cred: Dict[str, Any],
    output_file: Optional[str] = None,
    num_workers: int = 1
) -> str:
    """
    Process transcription with Pyannote API and return the output file path.

    Parameters
    ----------
    sound_path : str
        Path to the audio file to be processed.
    sftp_cred : dict
        SFTP credentials for uploading and downloading files.
    pyannote_cred : dict
        Pyannote credentials for API access.
    output_file : str, optional
        Path to save the transcription result as a JSON file. Defaults to 'pyannote-result.json'.
    num_workers : int, optional
        Number of workers to use for parallel processing of audio chunks. Defaults to 1.

    Returns
    -------
    str
        Path to the final output file.
    """
    osh.info(f"Pyannote processing started for {sound_path}")

    # Check the audio file validity
    osh.checkfile(sound_path, msg=f"Audio file not found at {sound_path}")
    osh.check(ah.is_valid_audio_file(sound_path), msg=f"Invalid audio file at {sound_path}")
    folder, basename, ext = osh.folder_name_ext(sound_path)

    # Set default output file if not provided
    if osh.emptystring(output_file):
        output_file = f"{folder}/{basename}-pyannote-result.json"

    # Create a hash for the sftp upload
    hash_basename = osh.hashfile(sound_path) + "." + ext
    remote_file = sftp_cred["sftp_destination_path"] + "/" + hash_basename

    # Info logging about audio file duration
    duration = ah.get_audio_duration(sound_path)
    osh.info(f"Audio duration is {osh.time2str(duration)}")

    # Check if output already exists to avoid reprocessing
    if osh.file_exists(output_file):
        osh.info(f"Pyannote processing already done for {sound_path}. Result: {output_file}")
        return output_file  # Return if the file already exists

    # Handle large files by splitting into chunks
    if duration > pyannote_cred["split_time"]:
        temporary_chunks_folder = f"{folder}/pyannote-chunks"
        osh.make_directory(temporary_chunks_folder)
        audio_chunks = ah.split_audio_regularly(sound_path, temporary_chunks_folder, pyannote_cred["split_time"] * 0.5)
        osh.info(f"Audio file split into chunks: {', '.join(audio_chunks)}")
        analysis_chunks = []

        w = min(num_workers, len(audio_chunks))
        if w <=1:
            for i, chunk_file in enumerate(audio_chunks):
                o = f"{temporary_chunks_folder}/pyannote-result-chunk-{i:04d}.json"
                pyannote_processing(chunk_file, sftp_cred, pyannote_cred, output_file=o, num_workers=1)
                analysis_chunks.append(o)
        else:
            # Parallelize processing of each chunk
            osh.info(f"Parallelization: {w} workers")
            with ThreadPoolExecutor(max_workers=w) as executor:
                future_to_chunk = {
                    executor.submit(pyannote_processing, chunk_file, sftp_cred, pyannote_cred, output_file=f"{temporary_chunks_folder}/pyannote-result-chunk-{i:04d}.json", num_workers = 1): chunk_file
                    for i, chunk_file in enumerate(audio_chunks)
                }

                for future in as_completed(future_to_chunk):
                    try:
                        result_file = future.result()
                        analysis_chunks.append(result_file)
                    except Exception as exc:
                        osh.error(f"Error processing chunk {future_to_chunk[future]}: {exc}")


        return _merge_chunk_results(output_file, audio_chunks, analysis_chunks, sftp_cred, pyannote_cred)

    else:
        if not sftph.remote_file_exists(remote_file, sftp_cred):
            osh.info(f"Uploading {sound_path} to {remote_file}")
            sftph.upload(sound_path, sftp_cred, sftp_address=remote_file)

        # Process the file using Pyannote API
        file_url = sftp_cred["sftp_https"] + "/" + hash_basename
        url, webhook_url, pyannote_key = (
            pyannote_cred["pyannote_url"],
            pyannote_cred["webhook_url"],
            pyannote_cred["pyannote_key"],
        )
        headers = {"Authorization": f"Bearer {pyannote_key}"}
        data = {"webhook": webhook_url, "url": file_url, "confidence": True}

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            job_id = response.json().get("jobId")
            if not job_id:
                osh.error("Job ID not found in the response. (pyannote)")
        else:
            osh.error(f"API request failed with status code {response.status_code}: {response.text}")

        remote_json = f"{sftp_cred['sftp_destination_path']}/{job_id}.txt"

        # Polling for the result file to be generated
        _poll_for_json_final(remote_json, sftp_cred, duration, pyannote_cred)

        # Download the final result
        sftph.download(remote_json, sftp_cred, output_file)

        return output_file


def _poll_for_json_final(
    remote_json: str,
    sftp_cred: Dict[str, str],
    duration: float,
    pyannote_cred: Dict[str, Any]
) -> None:
    """
    Polls the SFTP server for the remote JSON file and logs statuses during the wait.

    Parameters
    ----------
    remote_json : str
        Path to the remote JSON file.
    sftp_cred : dict
        SFTP credentials for file operations.
    duration : float
        Duration of the audio file.
    pyannote_cred : dict
        Pyannote credentials for API access.
    """
    waiting_time_per_iteration = max(
        duration * pyannote_cred["speed_factor"] / pyannote_cred["approximated_number_iterations"],
        pyannote_cred["min_waiting_time_per_iteration"],
    )
    approximated_number_iterations = round(1 + duration * pyannote_cred["speed_factor"] / waiting_time_per_iteration)
    counter, total_wait = 0, osh.tic()
    approximated_waiting_time = osh.time2str(round(1 + approximated_number_iterations * waiting_time_per_iteration))
    dd = osh.time2str(duration)
    
    while not sftph.remote_file_exists(remote_json, sftp_cred):
        counter += 1
        tt = osh.time2str(osh.toc(total_wait))
        status = f"Pyannote working (time = {tt} / {approximated_waiting_time}) for a sound duration of {dd}. Iteration {counter}/{approximated_number_iterations}."
        osh.info(status)
        osh.info(f"Waiting for {waiting_time_per_iteration} seconds")
        time.sleep(waiting_time_per_iteration)

    tt = osh.time2str(osh.toc(total_wait))
    dd = osh.time2str(duration)
    osh.info(f"Remote JSON file available after {tt} (audio duration = {dd})")


def _merge_chunk_results(
    merged_result_file: str,
    audio_chunks: list,
    analysis_chunks: list,
    sftp_cred: Dict[str, str],
    pyannote_cred: Dict[str, Any]
) -> str:
    """
    Merges diarization and confidence score data from multiple chunk results into a single file.

    Parameters
    ----------
    merged_result_file : str
        Path to the final merged result file.
    audio_chunks : list
        List of audio chunk file paths.
    analysis_chunks : list
        List of JSON files containing analysis results for the audio chunks.
    sftp_cred : dict
        SFTP credentials used for handling files.
    pyannote_cred : dict
        Pyannote credentials used for API access.

    Returns
    -------
    str
        Path to the merged result file.
    """
    confidences = []
    for i, analysis in enumerate(analysis_chunks):
        with open(analysis, "rt", encoding="utf8") as fin:
            data = json.load(fin)
            confidences += data["output"]["confidence"]["score"]

    original_spk2bit = {}
    for i, (analysis, audio) in enumerate(zip(analysis_chunks, audio_chunks)):
        _, b, _ = osh.folder_name_ext(audio)
        with open(analysis, "rt", encoding="utf8") as fin:
            diarization_data = json.load(fin)["output"]["diarization"]

        local_speakers = sorted(list(set(d["speaker"] for d in diarization_data)))

        for s in local_speakers:
            bits = [d for d in diarization_data if d["speaker"] == s]
            biggest_bit = max(bits, key=lambda x: x["end"] - x["start"])
            middle = 0.5 * (biggest_bit["start"] + biggest_bit["end"])
            biggest_bit["start"] = max(biggest_bit["start"], middle - 3)
            biggest_bit["end"] = min(biggest_bit["end"], middle + 3)
            speaker_name = f"{b}_{s}"
            biggest_bit["origin"] = audio
            biggest_bit["speaker"] = speaker_name
            original_spk2bit[speaker_name] = biggest_bit

    time_cursor = 0
    list_of_audio_bits = []
    f, _, _ = osh.folder_name_ext(audio_chunks[0])
    concatenated_audio = osh.join(f, "concatenated.mp3")
    bit_counter = 0
    synthetic_diarization = []

    for i, speaker_name in enumerate(original_spk2bit):
        bit = original_spk2bit[speaker_name]
        new_audio_chunk = f"{f}/bit_{bit_counter:04d}.mp3"
        audio_bit = ah.extract_audio_chunk(
            bit["origin"], bit["start"], bit["end"], new_audio_chunk, overwrite=True
        )
        list_of_audio_bits.append(audio_bit)
        duration = bit["end"] - bit["start"]
        bit["start"] = time_cursor
        bit["end"] = time_cursor + duration
        synthetic_diarization.append(bit)
        time_cursor += duration
        bit_counter += 1

        silence_chunk = f"{f}/bit_{bit_counter:04d}.mp3"
        silence_duration = 1.0
        silence = ah.generate_silent_audio(silence_duration, silence_chunk, overwrite=True)
        list_of_audio_bits.append(silence)
        time_cursor += silence_duration
        bit_counter += 1

    synthetic_audio = ah.audio_concatenation(list_of_audio_bits, concatenated_audio, overwrite=True)

    synthetic_audio_duration = ah.get_audio_duration(synthetic_audio)
    if synthetic_audio_duration > 2 * 60 * 60:
        tt = osh.time2str(synthetic_audio_duration)
        osh.error(f"Synthetic audio exceeds the 2-hour processing limit for Pyannote {tt}:\n\t{synthetic_audio}")

    output_file = pyannote_processing(synthetic_audio, sftp_cred, pyannote_cred)

    with open(output_file, "rt", encoding="utf8") as fin:
        synthetic_analysis = json.load(fin)

    synthetic_speakers = sorted(set(d["speaker"] for d in synthetic_analysis["output"]["diarization"]))
    original_speakers = sorted(original_spk2bit.keys())

    R, C = len(original_speakers), len(synthetic_speakers)
    M = np.zeros((R, C))

    for i, s_original in enumerate(original_speakers):
        original_bit = original_spk2bit[s_original]
        for j, s_synthetic in enumerate(synthetic_speakers):
            synthetic_bits = [d for d in synthetic_analysis["output"]["diarization"] if d["speaker"] == s_synthetic]
            for sb in synthetic_bits:
                intersection = max(0, min(sb["end"], original_bit["end"]) - max(sb["start"], original_bit["start"]))
                M[i, j] += intersection / (original_bit["end"] - original_bit["start"])

    osh.check(np.sum(M.ravel()) > 0, "No overlap found between original and synthetic speakers")

    original2synthetic = {}
    for i, s_original in enumerate(original_speakers):
        j = np.argmax(M[i, :])
        original2synthetic[s_original] = synthetic_speakers[j]

    time_cursor = 0
    merged_diarization = []
    resolution = None
    for i, (analysis, audio) in enumerate(zip(analysis_chunks, audio_chunks)):
        _, b, _ = osh.folder_name_ext(audio)
        with open(analysis, "rt", encoding="utf8") as fin:
            d = json.load(fin)
            chunk_diarization = d["output"]["diarization"]
            resolution = d["output"]["confidence"]["resolution"]

        for entry in chunk_diarization:
            original_speaker = f"{b}_{entry['speaker']}"
            entry["speaker"] = original2synthetic.get(original_speaker, original_speaker)
            entry["start"] += time_cursor
            entry["end"] += time_cursor

        merged_diarization.extend(chunk_diarization)

        time_cursor += ah.get_audio_duration(audio)

    final_result = {"output": {"diarization": merged_diarization, "confidence": {"score": confidences, "resolution": resolution}}}
    with open(merged_result_file, "wt", encoding="utf8") as fout:
        json.dump(final_result, fout, indent=2)

    return merged_result_file
