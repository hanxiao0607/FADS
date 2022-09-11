# FADS: Few-shot Anomaly Detection and Classification Through Reinforced Data Selection
A Pytorch implementation of [FADS]().

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.9
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions

Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/FADS.git my-project
        cd my-project

Run and test:

        python3 main_CERT.py
        or
        python3 main_IDS.py
        or
        python3 main_UNSW.py

## Citation
