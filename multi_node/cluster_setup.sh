#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3-venv
python3.7 -m venv /shared/venv/
source /shared/venv/bin/activate
pip install wheel
echo 'source /shared/venv/bin/activate' >> ~/.bashrc

cd /shared
git clone https://github.com/suraj813/distributed-pytorch.git
cd distributed-pytorch/multi_node
python3 -m pip install setuptools==59.5.0
pip install -r requirements.txt
