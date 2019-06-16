# Getting started

Set up python virtualenv and install packages:

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

Anytime a new shell is started, make sure to run `source venv/bin/activate` before any scripts.

To run CCES simulations:

    ./simulate-all-cces.sh

To run notebooks:

    venv/bin/jupyter notebook
