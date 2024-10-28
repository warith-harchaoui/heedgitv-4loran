# heedgitv-4loran
This Diariazation engine combines gladia and pyannote to perform state of the art diarization based on sound.

[![logo](logo.png)](https://harchaoui.org/warith/ai-helpers)


Done: Sound analysis
Doing: Active Speaker Detection uses attention mechanisms to combine images and sound to recognize the talking head. This would name the speakers thanks to a database of faces

This project uses the _AI Helpers_

[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)

# Installation

## Install Package

We recommend using Python environments. Check this link if you're unfamiliar with setting one up:

[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)

### Install `yt-dlp` and `ffmpeg`

To install YT Helper, you must install the following dependencies:

- For macOS 🍎
  
Get [brew](https://brew.sh) and install the necessary packages:
```bash
brew install yt-dlp
brew install ffmpeg
```

- For Ubuntu 🐧
```bash
sudo apt install yt-dlp
sudo apt install ffmpeg
```

- For Windows 🪟
  - `yt-dlp`: Download [yt-dlp from its repository](https://github.com/yt-dlp/yt-dlp) and follow the instructions for your system.

  - `ffmpeg`: Go to the [FFmpeg website](https://ffmpeg.org/download.html) and follow the instructions for downloading FFmpeg. You'll need to manually add FFmpeg to your system PATH.

## Install the diarization engine in your environment:
```bash
# activate your python environment
conda activate env4dzh
pip install --force-reinstall --no-cache-dir git+https://github.com/warith-harchaoui/heedgitv-4loran.git@main
```

# Usage
```bash
# activate your python environment 
# and
conda activate env4dzh
streamlit run gui.py
```


# Authors
 - [Warith Harchaoui](https://harchaoui.org/warith)
 - [Laurent Pantanacce](https://www.linkedin.com/in/pantanacce/)

