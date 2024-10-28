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


from itertools import product
from collections import OrderedDict  # Corrected import
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
        "max_wait_time",
    ]
    return osh.get_config(keys, "pyannote", config_path)


def pyannote_processing(
    sound_path: str,
    sftp_cred: Dict[str, str],
    pyannote_cred: Dict[str, Any],
    output_file: Optional[str] = None,
    num_workers: int = 1,
    overwrite: bool = True
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
        Path to save the transcription result as a JSON file.
    num_workers : int, optional
        Number of workers to use for parallel processing of audio chunks. Defaults to 1.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Defaults to True.

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
    if osh.file_exists(output_file) and not overwrite:
        osh.info(f"Pyannote processing already done for {sound_path}. Result: {output_file}")
        return output_file  # Return if the file already exists and overwrite is False

    # Determine if we need to split the audio
    allow_splitting = duration > pyannote_cred["split_time"]

    # Handle large files by splitting into chunks
    if allow_splitting:
        temporary_chunks_folder = f"{folder}/pyannote-chunks"
        osh.make_directory(temporary_chunks_folder)
        # Use half of the split_time for splitting duration
        split_duration = pyannote_cred["split_time"] * 0.5
        audio_chunks = ah.split_audio_regularly(sound_path, temporary_chunks_folder, split_duration)
        osh.info(f"Audio file split into chunks: {', '.join(audio_chunks)}")
        analysis_chunks = []


        w = min(num_workers, len(audio_chunks))
        analysis_chunks = []

        # Prepare output file paths
        output_files = [f"{temporary_chunks_folder}/pyannote-result-chunk-{i:04d}.json" for i in range(len(audio_chunks))]

        if w <= 1:
            # Sequential processing
            for chunk_file, output_file in zip(audio_chunks, output_files):
                pyannote_processing(
                    chunk_file,
                    sftp_cred,
                    pyannote_cred,
                    output_file=output_file,
                    num_workers=1,
                    overwrite=overwrite
                )
                analysis_chunks.append(output_file)
        else:
            # Parallel processing
            osh.info(f"Parallelization: {w} workers")
            with ThreadPoolExecutor(max_workers=w) as executor:
                futures = [
                    executor.submit(
                        pyannote_processing,
                        chunk_file,
                        sftp_cred,
                        pyannote_cred,
                        output_file=output_file,
                        num_workers=1,
                        overwrite=overwrite
                    )
                    for chunk_file, output_file in zip(audio_chunks, output_files)
                ]

                for future, output_file in zip(as_completed(futures), output_files):
                    try:
                        future.result()
                        analysis_chunks.append(output_file)
                    except Exception as exc:
                        osh.error(f"Error processing chunk {output_file}: {exc}")


        return _merge_chunk_results(
            output_file,
            audio_chunks,
            analysis_chunks,
            sftp_cred,
            pyannote_cred,
            overwrite=overwrite,
            num_workers=num_workers
        )
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
    approximated_number_iterations = round(
        1 + duration * pyannote_cred["speed_factor"] / waiting_time_per_iteration
    )
    counter, total_wait = 0, osh.tic()
    approximated_waiting_time = osh.time2str(
        round(1 + approximated_number_iterations * waiting_time_per_iteration)
    )
    dd = osh.time2str(duration)

    max_wait_time = pyannote_cred.get('max_wait_time', 3600)  # Maximum wait time in seconds
    while not sftph.remote_file_exists(remote_json, sftp_cred):
        counter += 1
        tt = osh.time2str(osh.toc(total_wait))
        if osh.toc(total_wait) > max_wait_time:
            osh.error(f"Maximum wait time exceeded while waiting for {remote_json}")
        status = (
            f"Pyannote working (time = {tt} / {approximated_waiting_time}) "
            f"for a sound duration of {dd}. Iteration {counter}/{approximated_number_iterations}."
        )
        osh.info(status)
        osh.info(f"Waiting for {waiting_time_per_iteration} seconds")
        time.sleep(waiting_time_per_iteration)

    tt = osh.time2str(osh.toc(total_wait))
    dd = osh.time2str(duration)
    osh.info(f"Remote JSON file available after {tt} (audio duration = {dd})")


def _merge_chunk_results(
    output_file: str,
    audio_chunks: list,
    analysis_chunks: list,
    sftp_cred: Dict[str, str],
    pyannote_cred: Dict[str, Any],
    overwrite: bool = True,
    num_workers: int = 1,
    silent_time: float = 3.0,            # Configurable silent time
    max_segment_duration: float = 6.0,   # Configurable max segment duration
    n_segments_per_speaker: int = 3      # Number of segments per speaker
) -> str:
    """
    Merges diarization and confidence score data from multiple chunk results into a single file.

    Parameters
    ----------
    output_file : str
        Path to the final merged result file.
    audio_chunks : list
        List of audio chunk file paths.
    analysis_chunks : list
        List of JSON files containing analysis results for the audio chunks.
    sftp_cred : dict
        SFTP credentials used for handling files.
    pyannote_cred : dict
        Pyannote credentials used for API access.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Defaults to True.
    num_workers : int, optional
        Number of workers to use for parallel processing of audio chunks. Defaults to 1.
    silent_time : float, optional
        Duration of silence between segments in synthetic audio. Defaults to 3.0 seconds.
    max_segment_duration : float, optional
        Maximum duration for a speaker segment in synthetic audio. Defaults to 6.0 seconds.
    n_segments_per_speaker : int, optional
        Number of segments to collect per speaker for robustness. Defaults to 3.

    Returns
    -------
    str
        Path to the merged result file.
    """
    # Collect confidences from all chunks
    confidences = []
    for analysis in analysis_chunks:
        with open(analysis, "rt", encoding="utf8") as fin:
            data = json.load(fin)
            confidences += data["output"]["confidence"]["score"]

    if osh.file_exists(output_file) and not overwrite:
        osh.info(f"File {output_file} already exists. Skipping merge.")
        return output_file

    folder, sound_basename, _ = osh.folder_name_ext(output_file)
    synthetic_folder = osh.join(folder, sound_basename, "pyannote")
    osh.make_directory(synthetic_folder)

    # Step 1: Collect multiple segments per speaker across chunks
    speaker_segments = {}
    for i, (audio_c, analysis_c) in enumerate(zip(audio_chunks, analysis_chunks)):
        with open(analysis_c, "rt", encoding="utf8") as fin:
            chunk_results = json.load(fin)
            chunk_results = chunk_results["output"]["diarization"]
            for chunk_segment in chunk_results:
                name = f"{i}_{chunk_segment['speaker']}"
                start, end = chunk_segment["start"], chunk_segment["end"]
                duration = end - start

                # Adjust segment to be within max_segment_duration
                middle = 0.5 * (start + end)
                start = max(start, middle - 0.5 * max_segment_duration)
                end = min(end, middle + 0.5 * max_segment_duration)
                duration = end - start

                # Store segments per speaker
                if name not in speaker_segments:
                    speaker_segments[name] = []
                speaker_segments[name].append((duration, start, end, audio_c))

    # Keep top n_segments_per_speaker segments per speaker
    for name in speaker_segments:
        segments = speaker_segments[name]
        # Sort segments by duration in descending order
        segments.sort(key=lambda seg: seg[0], reverse=True)
        # Keep only the top N segments
        num_segments = min(n_segments_per_speaker, len(segments))
        speaker_segments[name] = segments[:num_segments]

    # Step 2: Create synthetic audio with multiple segments per speaker
    synthetic_audio_segments = []
    synthetic_diarization = []
    time_cursor = 0.0
    for i, (name, segments) in enumerate(speaker_segments.items()):
        for j, (duration, start, end, audio_c) in enumerate(segments):
            segment_path = f"{synthetic_folder}/synthetic_segment_{i}_{j}.mp3"
            ah.extract_audio_chunk(audio_c, start, end, segment_path, overwrite=True)
            synthetic_audio_segments.append(segment_path)
            synthetic_diarization.append({
                "start": time_cursor,
                "end": time_cursor + (end - start),
                "speaker": name,
                "word": segment_path
            })
            time_cursor += (end - start)

            # Add silent segment for separation
            silence_path = f"{synthetic_folder}/silence_{i}_{j}.mp3"
            ah.generate_silent_audio(silent_time, output_audio_filename=silence_path)
            synthetic_audio_segments.append(silence_path)
            time_cursor += silent_time

    synthetic_diarization = sorted(synthetic_diarization, key=lambda x: x["start"])

    # Concatenate all segments into one synthetic audio file
    synthetic_audio = f"{synthetic_folder}/synthetic_audio.mp3"
    osh.info(f"Concatenating synthetic audio segments to {synthetic_audio}")
    ah.audio_concatenation(synthetic_audio_segments, synthetic_audio, overwrite=True)

    # Step 3: Process the synthetic audio to get consistent speaker labels
    synthetic_output_file = f"{synthetic_folder}/synthetic_output.json"
    pyannote_processing(
        synthetic_audio,
        sftp_cred,
        pyannote_cred,
        output_file=synthetic_output_file,
        num_workers=1,
        overwrite=overwrite
    )

    with open(synthetic_output_file, "rt", encoding="utf8") as fin:
        synthetic_results = json.load(fin)
        synthetic_results = synthetic_results["output"]["diarization"]
        for s in synthetic_results:
            s["speaker"] = str(s["speaker"])
        synthetic_results = sorted(synthetic_results, key=lambda x: x["start"])

    # Map initial speaker names to synthetic speakers based on maximum overlap
    initial_to_synthetic_names = {}
    for sd in synthetic_diarization:
        sd_start, sd_end = sd["start"], sd["end"]
        overlaps = {}
        for sr in synthetic_results:
            sr_start, sr_end = sr["start"], sr["end"]
            intersection = max(0, min(sd_end, sr_end) - max(sd_start, sr_start))
            if intersection > 0:
                overlaps[sr["speaker"]] = overlaps.get(sr["speaker"], 0) + intersection

        if overlaps:
            # Assign to the synthetic speaker with the maximum overlap
            best_match = max(overlaps, key=overlaps.get)
            initial_to_synthetic_names[sd["speaker"]] = best_match
        else:
            osh.warning(f"No overlap found for speaker {sd['speaker']}")
            initial_to_synthetic_names[sd["speaker"]] = sd["speaker"]  # Keep original

    # Step 4: Adjust original chunks with new speaker mappings and save merged output
    res = []
    time_cursor = 0.0
    for i, (audio_chunk, analysis_chunk) in enumerate(zip(audio_chunks, analysis_chunks)):
        duration = ah.get_audio_duration(audio_chunk)
        with open(analysis_chunk, "rt", encoding="utf8") as fin:
            chunk_results = json.load(fin)
            chunk_diarization = chunk_results["output"]["diarization"]
            for sentence in chunk_diarization:
                original_name = f"{i}_{sentence['speaker']}"
                if original_name in initial_to_synthetic_names:
                    sentence["speaker"] = initial_to_synthetic_names[original_name]
                else:
                    osh.warning(f"Speaker {original_name} not found in mapping.")
                    sentence["speaker"] = sentence["speaker"]  # Keep original
                # Adjust timings for the overall timeline
                sentence["start"] += time_cursor
                sentence["end"] += time_cursor
                res.append(sentence)
        time_cursor += duration

    res = sorted(res, key=lambda x: x["start"])

    # Adjust confidence scores length to match diarization results
    if len(confidences) != len(res):
        osh.warning("Mismatch between number of confidence scores and diarization segments.")
        # Adjust accordingly, e.g., truncate or pad with average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if len(confidences) > len(res):
            confidences = confidences[:len(res)]
        else:
            confidences += [avg_confidence] * (len(res) - len(confidences))

    # Use template from the first chunk
    with open(analysis_chunks[0], "rt", encoding="utf8") as fin:
        template = json.load(fin)

    final_output = {
        "jobId": template.get("jobId", ""),
        "status": template.get("status", "done"),
        "output": {
            "diarization": res,
            "confidence": {
                "score": confidences,
                "resolution": template["output"]["confidence"].get("resolution", 0.02)
            }
        }
    }

    # Save final merged results
    with open(output_file, "wt", encoding="utf8") as fout:
        json.dump(final_output, fout, indent=2)
    osh.info(f"Saved merged results to:\n\t{output_file}")

    return output_file
