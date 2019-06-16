This repository contains replication code for the CCES simulations in the paper [Active Matrix Factorization for Surveys](https://arxiv.org/abs/1902.07634).

Simulations are written in Python. We highly recommend using a virtual environment.

To set up python virtualenv and install packages:

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

Anytime a new shell is started, make sure to run `source venv/bin/activate` before any scripts.

To run CCES simulations:

    ./simulate-all-cces.sh

To run notebooks:

    venv/bin/jupyter notebook
