# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['diarization_helper']

package_data = \
{'': ['*']}

install_requires = \
['audio-helper @ git+https://github.com/warith-harchaoui/audio-helper.git@main',
 'keybert>=0.8.5,<0.9.0',
 'langdetect>=1.0.9,<2.0.0',
 'matplotlib>=3.9.2,<4.0.0',
 'nltk>=3.9.1,<4.0.0',
 'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@main',
 'rake-nltk>=1.0.6,<2.0.0',
 'sftp-helper @ git+https://github.com/warith-harchaoui/sftp-helper.git@main',
 'spacy>=3.8.2,<4.0.0',
 'streamlit>=1.39.0,<2.0.0',
 'video-helper @ git+https://github.com/warith-harchaoui/video-helper.git@main',
 'yt-helper @ git+https://github.com/warith-harchaoui/yt-helper.git@main']

setup_kwargs = {
    'name': 'diarization-helper',
    'version': '0.1.0',
    'description': 'This Diarization Helper provides a set of utility functions to determine who said when in a video or audio file.',
    'long_description': "# Diarization Engine\n\nThe Diariazation engine combines gladia and pyannote to perform state of the art diarization based on sound.\n\n[![logo](logo.png)](https://harchaoui.org/warith/ai-helpers)\n\n\nDone: Sound analysis\nDoing: Active Speaker Detection uses attention mechanisms to combine images and sound to recognize the talking head. This would name the speakers thanks to a database of faces\n\nThis project uses the _AI Helpers_\n\n[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)\n\n# Installation\n\n## Install Package\n\nWe recommend using Python environments. Check this link if you're unfamiliar with setting one up:\n\n[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)\n\n### Install `yt-dlp` and `ffmpeg`\n\nTo install YT Helper, you must install the following dependencies:\n\n- For macOS 🍎\n  \nGet [brew](https://brew.sh) and install the necessary packages:\n```bash\nbrew install yt-dlp\nbrew install ffmpeg\n```\n\n- For Ubuntu 🐧\n```bash\nsudo apt install yt-dlp\nsudo apt install ffmpeg\n```\n\n- For Windows 🪟\n  - `yt-dlp`: Download [yt-dlp from its repository](https://github.com/yt-dlp/yt-dlp) and follow the instructions for your system.\n\n  - `ffmpeg`: Go to the [FFmpeg website](https://ffmpeg.org/download.html) and follow the instructions for downloading FFmpeg. You'll need to manually add FFmpeg to your system PATH.\n\n## Install the diarization engine in your environment:\n```bash\n# activate your python environment\nconda activate env4dzh\npip install --force-reinstall --no-cache-dir git+https://github.com/warith-harchaoui/diarization-helper.git@main\n```\n\n# Usage\n```bash\n# activate your python environment \n# and\nconda activate env4dzh\nstreamlit run gui.py\n```\n\n\n# Authors\n - [Warith Harchaoui](https://harchaoui.org/warith)\n - [Laurent Pantanacce](https://www.linkedin.com/in/pantanacce/)\n\n",
    'author': 'Warith Harchaoui',
    'author_email': 'warith.harchaoui@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)

