#!/bin/sh

sudo sed -i '/Components:/ s/main/main contrib non-free/' /etc/apt/sources.list.d/debian.sources
sudo echo 'APT::Get::Update::SourceListWarnings::NonFreeFirmware "false";' > /etc/apt/apt.conf.d/no-bookworm-firmware.conf
sudo apt-get update
DEBIAN_FRONTEND=noninteractive \
sudo apt-get install -y --no-install-recommends festival mbrola espeak-ng festvox-en1 festlex-poslex festlex-cmu

python -m pip install --user -U pip
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user -r requirements.txt -r requirements-dev.txt -r requirements-opt.txt
