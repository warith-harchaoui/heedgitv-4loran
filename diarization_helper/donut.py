"""
This donut.py module provides a set of utility functions to generate donut charts in vega json format (not vega lite).

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

def make_donut(data: List[Dict], output_json: str, title: str):
    """
    make_donut generates a Vega-Lite specification for a donut chart.

    Parameters:
    - data: a list of dictionaries with the following keys:
        - label: the name of the slice
        - seconds: the time in seconds of the slice
        - string_time: the time in string format of the slice
        - percent: the percentage of the slice
        - color: the color of the slice in hex format
    - output_json: the path to save the Vega-Lite specification as a JSON file.

    Returns:
    - None

    Example:
    data = [
        {"label": "Slice 1", "seconds": 100, "string_time": "00:01:40", "percent": 20, "color": "#ff0000"},
        {"label": "Slice 2", "seconds": 200, "string_time": "00:03:20", "percent": 40, "color": "#00ff00"},
        {"label": "Slice 3", "seconds": 300, "string_time": "00:05:00", "percent": 60, "color": "#0000ff"},
    ]
    make_donut(data, "donut_chart.json")

    The above example will generate a donut chart with three slices:
    - Slice 1: 20% of the chart, colored red, with a time of 00:01:40
    - Slice 2: 40% of the chart, colored green, with a time of 00:03:20
    - Slice 3: 60% of the chart, colored blue, with a time of 00:05:00
    The donut chart will be saved in donut_chart.json.

    """

    d = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 450,
        "height": 450,
        "title": {
            "text": f"🍩 {title}",  # Add title here
            "font": "Lato",
            "fontWeight": {"value": 400},
            "fontSize": {"value": 24},
            "anchor": "start",  # This aligns the title to the left
            "dy": 0,  # Adjust the vertical position of the title
        },
        "data": [
            {
                "name": "table",
                "values": data,
                "transform": [
                    {
                        "type": "window",
                        "ops": ["sum"],
                        "fields": ["percent"],
                        "as": ["cumulativePercent"],
                    },
                    {
                        "type": "formula",
                        "expr": "datum.cumulativePercent - datum.percent",
                        "as": "cumulativePercentStart",
                    },
                    {
                        "type": "formula",
                        "expr": "(datum.cumulativePercentStart + datum.cumulativePercent) / 2",
                        "as": "meanCumulativePercent",
                    },
                    {
                        "type": "formula",
                        "expr": "datum.meanCumulativePercent <= 50 ? 'left' : 'right' ",
                        "as": "alignment",
                    },
                    {"type": "pie", "field": "percent"},
                ],
            }
        ],
        "scales": [
            {
                "name": "color",
                "type": "ordinal",
                "domain": {"data": "table", "field": "label"},
                "range": {"data": "table", "field": "color"},
            }
        ],
        "marks": [
            {
                "type": "arc",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "fill": {"scale": "color", "field": "label"},
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                    },
                    "update": {
                        "startAngle": {"field": "startAngle"},
                        "endAngle": {"field": "endAngle"},
                        "padAngle": {"value": 0.02},
                        "innerRadius": {"signal": "100"},
                        "outerRadius": {"signal": "200"},
                        "cornerRadius": {"value": 5},
                    },
                },
            },
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2 + 20 - 5"},
                        "radius": {"signal": "150"},
                        "theta": {"signal": "(datum.startAngle + datum.endAngle) / 2"},
                        "fill": {"value": "#ffffff"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "text": {"signal": "datum.percent + '%'"},
                        "font": {"value": "Lato"},
                        "fontWeight": {"value": 400},
                        "fontSize": {"value": 16},
                    }
                },
            },
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2 - 5"},
                        "radius": {"signal": "150"},
                        "theta": {"signal": "(datum.startAngle + datum.endAngle) / 2"},
                        "fill": {"value": "#ffffff"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "text": {"signal": "datum.string_time"},
                        "font": {"value": "Lato"},
                        "fontWeight": {"value": 400},
                        "fontSize": {"value": 14},
                    }
                },
            },
            {
                "type": "text",
                "from": {"data": "table"},
                "encode": {
                    "enter": {
                        "x": {"signal": "width / 2"},
                        "y": {"signal": "height / 2"},
                        "radius": {"signal": "230"},
                        "theta": {"signal": "(datum.startAngle + datum.endAngle) / 2"},
                        "fill": {"value": "black"},
                        "align": {"field": "alignment"},
                        "baseline": {"value": "middle"},
                        "text": {"signal": "datum.label"},
                        "font": {"value": "Lato"},
                        "fontWeight": {"value": 400},
                        "fontSize": {"value": 16},
                    }
                },
            },
        ],
    }

    with open(output_json, "wt", encoding="utf-8") as f:
        json.dump(d, f, indent=4)

    osh.info(f"Vega-Lite donut chart saved in {output_json}")




def make_donut_image(source, o=None, title = None, sftp_cred=None, speakers=None):
    if title is None:
        s = json.dumps(source, indent=2)
        h = str(hash(s))
        title = h

    colors = ["#FF6961", "#FFB340", "#02D46A", "#0C817B", "#007AFF", "#5856D6", "#BF40BF", "#FFB6C1"]

    total_duration = sum(seg["end"] - seg["start"] for seg in source)
    if speakers is None:
        speakers = sorted(list(set(str(seg["speaker"]) for seg in source)))
        times = {s: sum(seg["end"] - seg["start"] for seg in source if str(seg["speaker"]) == s) for s in speakers}
        speakers = sorted(speakers, key=lambda x: times[x], reverse=True)

    c = {s: colors[i % len(colors)] for i, s in enumerate(speakers)}
    donut_data = []
    for s in speakers:
        segs = [seg for seg in source if str(seg["speaker"]) == s]
        duration = sum(seg["end"] - seg["start"] for seg in segs)
        d={
            "label": s,
            "seconds": duration,
            "string_time": osh.time2str(duration),
            "percent": round(100 * duration / total_duration),
            "color": c[s]
        }
        donut_data.append(d)

    donut_data = sorted(donut_data, key=lambda x: x["seconds"], reverse=True)
    if o is None:
        b = f"{title}.json"
        o = b
    _,b, _ = osh.folder_name_ext(o)
    t = f"Donut of Speakers: {title}"
    make_donut(
        donut_data,
        o,
        f"{t}"
    )
    if not(sftp_cred is None):
        remote_file = sftp_cred["sftp_destination_path"] +"/"+ b + ".json"
        sftph.upload(o, sftp_cred, sftp_address=remote_file)
        url = remote_file.replace(sftp_cred["sftp_destination_path"], sftp_cred["sftp_https"])
        url2 = f"https://deraison.ai/clients/vega/vega.php?json={url}"
        osh.info(f"Render: {url2}")
        return o, url2
    return o