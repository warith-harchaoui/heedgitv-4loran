"""
This angel_hair_pasta.py module provides a set of utility functions to create an angel hair pasta visualization in vega json format (not vega lite).

Dependencies
- os_helper (custom utility for OS tasks)
- sftp_helper (custom utility for SFTP tasks)

Authors:
- Warith Harchaoui, https://harchaoui.org/warith
"""


import json
from typing import List, Dict
import os_helper as osh
import sftp_helper as sftph
import numpy as np

time2str = osh.time2str

def make_angel_hair_pasta(
    data: List[Dict],
    speakers_order: List[str],
    output_json: str,
    name2color: Dict[str, str],
    title: str,
    cuts: List[Dict]=None,
    color_to_image_map: Dict[str, str]=None,
):
    # Sort data by speaker for timeline stacking
    speakers = speakers_order

    # Mapping speakers to Y-axis, adjusted by scale_factor
    speaker_y_mapping = {
        speaker: i * 30 * 2 for i, speaker in enumerate(speakers)
    }  # Adjust spacing between speakers

    if cuts is None:
        cuts = []
        
    cuts = [c for c in cuts if c["victim"] in speaker_y_mapping]

    # Determine the time domain (min and max time values)
    time_domain = [
        min([item["start"] for item in data]),
        max([item["end"] for item in data] + [cut["end"] for cut in cuts]),
    ]

    data = [
        {
            **item,
            "y": speaker_y_mapping[item["speaker"]],
            "start_formatted": time2str(item["start"]),
            "end_formatted": time2str(item["end"]),
        }
        for item in data
    ]

    if color_to_image_map is None:
        color_to_image_map = {}
    
    cuts = [
        {
            **cut,
            "y": speaker_y_mapping[cut["victim"]],
            "start_formatted": time2str(cut["start"]),
            "end_formatted": time2str(cut["end"]),
            "color": cut["executioner_color"],
            "image_url": color_to_image_map[cut["executioner_color"]],
        }
        for cut in cuts
    ]

    # List of possible time steps
    possible_time_steps = [
        5,
        10,
        30,
        60,
        5 * 60,
        10 * 60,
        15 * 60,
        20 * 60,
        30 * 60,
        60 * 60,
    ]
    T = time_domain[1]
    possible_steps = [T / t for t in possible_time_steps]
    ideal_step = 10
    t = np.argmin(np.abs(np.array(possible_steps) - ideal_step))
    best_step = possible_time_steps[t]

    # Create the time axis based on best step and adjust the time domain max
    time_axis = list(np.arange(0, T + best_step, best_step))
    time_domain[1] = max(time_axis)

    # Apply time2str to create human-readable tick labels
    time_ticks = time_axis  # The values for the ticks
    time_ticks_label = [time2str(time) for time in time_axis]  # Human-readable labels

    "datum.value === 0 ? 'a' : datum.value === 2 ? 'b' : datum.value === 4 ? 'c' : datum.value === 5 ? 'd' : datum.value === 19 ? 'e' : 'f'"
    # Create a signal for the time_ticks_label
    time_ticks_label_signal = [
        f"datum.value === {time} ? '{time2str(time)}' :"
        for time in time_ticks
    ]
    time_ticks_label_signal[-1] = f"'{time2str(time_axis[-1])}'"
    time_ticks_label_signal = " ".join(time_ticks_label_signal)

    image_size = 40  # Adjusted image size by scale_factor
    bigger_image_size = 45  # Adjusted image size by scale_factor

    off = int(-image_size)
    off_bigger = int(-bigger_image_size)



    # Update Vega spec axes with time_ticks and time_ticks_label
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "title": {
            "text": f"🍝 {title}",
            "font": "Lato",
            "fontWeight": {"value": 400},
            "fontSize": {"value": 24},
            "anchor": "start",  # This aligns the title to the left
            "dy": 0,  # Adjust the vertical position of the title
        },
        "width": 700 * 2,
        "height": (30 * len(speakers)) * 2,
        "padding": 5 * 2,
        "scales": [
            {
                "name": "xscale",
                "type": "linear",
                "domain": time_domain,
                "range": "width",
                # "nice": True,  # Ensures the time scale is more readable
            },
            {
                "name": "color",
                "type": "ordinal",
                "domain": speakers,
                "range": [name2color[spk] for spk in speakers],
            },
        ],
        "axes": [
            {
                "orient": "bottom",
                "scale": "xscale",
                "title": "Time",
                "values": time_ticks,  # Apply the ticks from time_axis
                # "labels": time_ticks_label,  # Apply the human-readable labels
                "encode": {
                    "labels": {
                        "update": {
                            "text": {
                                "signal": time_ticks_label_signal
                            }
                        }
                    }
                },
                "font": "Lato",
                "fontWeight": {"value": 200},
                "fontSize": {"value": 16},
            }
        ],
        "legends": [
            {
                "fill": "color",
                "title": "Speakers",
                "orient": "right",
                "encode": {
                    "symbols": {
                        "update": {
                            "fillOpacity": {"value": 1},
                            "strokeWidth": {"value": 2},
                        }
                    },
                    "labels": {
                        "update": {
                            "font": {"value": "Lato"},
                            "fontWeight": {"value": 400},
                            "fontSize": {"value": 16},
                        }
                    },
                    "title": {
                        "update": {
                            "font": {"value": "Lato"},
                            "fontWeight": {"value": 400},
                            "fontSize": {"value": 18},
                        }
                    },
                },
            }
        ],
        "marks": [
            # Rectangles representing the word intervals
            {
                "type": "rect",
                "from": {"data": "timeline_data"},
                "encode": {
                    "enter": {
                        "x": {"scale": "xscale", "field": "start"},
                        "x2": {"scale": "xscale", "field": "end"},
                        "y": {"field": "y"},  # Use the mapped Y position directly
                        "height": {"value": 20 * 2},  # Adjusted height by scale_factor
                        "fill": {"scale": "color", "field": "speaker"},
                    },
                    "update": {
                        "tooltip": {
                            "signal": "{'Speaker': datum.speaker, 'Words': datum.word, 'Start': datum.start_formatted, 'End': datum.end_formatted}"
                        },
                        "fillOpacity": {"value": 1},  # Normal opacity
                    },
                    "hover": {
                        "fillOpacity": {"value": 0.8},  # Slight transparency on hover
                    },
                },
            },
            # Image for cuts with hover animation
            {
                "type": "image",  # Use image for cuts
                "from": {"data": "cuts_data"},
                "encode": {
                    "enter": {
                        "x": {"scale": "xscale", "field": "start", "offset": off},
                        # "x": {
                        #     "signal": f"datum.start - {image_size}"
                        # },
                        "y": {
                            "signal": f"datum.y + 5"
                        },  
                        "width": {
                            "value": image_size
                        },  # Adjusted width of the image by scale_factor
                        "height": {
                            "value": image_size
                        },  # Adjusted height of the image by scale_factor
                        "url": {"field": "image_url"},  # Image URL based on color
                    },
                    "update": {
                        "tooltip": {
                            "signal": "{'Executioner': datum.executioner, 'Victim': datum.victim, 'Time': datum.start_formatted}"
                        },
                        "width": {"value": 20 * 2},  # Default size
                        "height": {"value": 20 * 2},  # Default size
                        "opacity": {"value": 1},  # Normal opacity
                    },
                    "hover": {
                        "x": {"scale": "xscale", "field": "start", "offset": off_bigger},
                        "y": {
                            "signal": f"datum.y + 5"
                        },  
                        "width": {"value": bigger_image_size},  # Enlarged width on hover
                        "height": {"value": bigger_image_size},  # Enlarged height on hover
                        "opacity": {"value": 1},  # Maintain full opacity on hover
                    },
                },
            },
        ],
        "data": [
            {
                "name": "timeline_data",
                "values": data,
            },
            {
                "name": "cuts_data",
                "values": cuts,
            },
        ],
    }

    # Write Vega spec to a JSON file
    with open(output_json, "wt", encoding="utf-8") as f:
        json.dump(vega_spec, f, indent=2)

    return output_json




def make_angel_hair_pasta_image(source, speakers_order, o=None, title = None, sftp_cred=None):
    if title is None:
        s = json.dumps(source, indent=2)
        h = str(hash(s))
        title = h

    colors = ["#FF6961", "#FFB340", "#02D46A", "#0C817B", "#007AFF", "#5856D6", "#BF40BF", "#FFB6C1"]

    total_duration = sum(seg["end"] - seg["start"] for seg in source)
    speakers = speakers_order
    speakers = [str(s) for s in speakers]
    name2color = {s: colors[i % len(colors)] for i, s in enumerate(speakers)}
    data = []
    for seg in source:
        speaker = str(seg["speaker"])
        w = ""
        if "word" in seg:
            w = seg["word"]
        elif "sentence" in seg:
            w = seg["sentence"]
        else:
            w = str(speaker)
        d = {
            "speaker": str(speaker),
            "start": seg["start"],
            "end": seg["end"],
            "word": w,
            "color": name2color[speaker],
        }
        data.append(d)

    data = sorted(data, key=lambda x: x["start"])
    if o is None:
        b = f"{title}.json"
        o = b

    _,b, _ = osh.folder_name_ext(o)
    t = f"Angel Hair Pasta of Speakers: {title}"
    make_angel_hair_pasta(
        data,
        speakers_order,
        o,
        name2color,
        f"{t}",
        cuts=None,
        color_to_image_map=None,
    )

    if not(sftp_cred is None):
        remote_file = sftp_cred["sftp_destination_path"] +"/"+ b + ".json"
        sftph.upload(o, sftp_cred, sftp_address=remote_file)
        url = remote_file.replace(sftp_cred["sftp_destination_path"], sftp_cred["sftp_https"])
        url2 = f"https://deraison.ai/clients/vega/vega.php?json={url}"
        osh.info(f"Render: {url2}")
        return o, url2
    return o