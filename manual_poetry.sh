#!/bin/zsh

# This is an attempt to automate the process of setting up a new Python project using Poetry.
# DO NOT RUN IT LIKE A SCRIPT; copy and paste the commands one by one in your terminal.

# Enable debug and error checking
set -x  # Print each command before executing it
set -e  # Exit immediately on any command failure

# Configurations
PROJECT_NAME="diarization-helper"
PYTHON_VERSION="3.10"  # Using Python 3.10 for compatibility
ENV="env4dzh"


DEPENDENCIES=(
  nltk
  rake_nltk
  keybert
  langdetect
  spacy
  matplotlib
  streamlit
  git+https://github.com/warith-harchaoui/os-helper.git@main
  git+https://github.com/warith-harchaoui/sftp-helper.git@main
  git+https://github.com/warith-harchaoui/audio-helper.git@main
  git+https://github.com/warith-harchaoui/video-helper.git@main
  git+https://github.com/warith-harchaoui/yt-helper.git@main 
)

DESCRIPTION="This Diarization Helper provides a set of utility functions to determine who said when in a video or audio file."
AUTHORS="Warith Harchaoui <warith@heedgi.com>"

# Initialize conda
conda init
source ~/.zshrc

# Remove existing environment if it exists
if conda info --envs | grep -q "^${ENV}[[:space:]]"; then
    echo "Environment $ENV already exists, removing it..."
    conda deactivate 2>/dev/null || true  # Attempt to deactivate, ignore errors if already inactive
    conda deactivate 2>/dev/null || true  # Attempt to deactivate, ignore errors if already inactive
    conda remove --name $ENV --all -y
else
    echo "Environment $ENV does not exist, skipping removal."
fi

# Create and activate the new environment
echo "Creating environment $ENV..."
conda create -y -n $ENV python=$PYTHON_VERSION
conda activate $ENV

# Install pip
conda install -y pip

# Install Poetry and poetry2setup
pip install --upgrade poetry poetry2setup

# Initialize a new Poetry project with Python version constraints
rm -f pyproject.toml poetry.lock
poetry init --name $PROJECT_NAME --description "$DESCRIPTION" --author "$AUTHORS" --python ">=${PYTHON_VERSION},<4.0" -n

# Convert the dependencies string into an array
DEPENDENCIES_ARRAY=(${=DEPENDENCIES})

# Add dependencies using Poetry
for dep in "${DEPENDENCIES[@]}"; do
    echo "Adding $dep..."
    poetry add "$dep"
done

# Install spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm

# Install project dependencies
poetry install

# Generate setup.py and export requirements.txt
poetry2setup > setup.py
poetry export --without-hashes --format=requirements.txt --output=requirements.txt

# Update author and description in setup.py
sed -i '' "s/author=.*/author='$AUTHORS',/" setup.py
sed -i '' "s/description=.*/description='$DESCRIPTION',/" setup.py

# Create environment.yml for conda users
cat <<EOL > environment.yml
name: $ENV
channels:
  - defaults
dependencies:
  - python=$PYTHON_VERSION
  - pip
  - pip:
$(sed 's/^/    - /' requirements.txt)
EOL

echo "Project setup completed successfully!"
