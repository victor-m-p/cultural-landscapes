#!/usr/bin/env bash

VENVNAME=glassenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
pip --version

pip install graphviz
pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt
pip install pygraphviz

deactivate
echo "build $VENVNAME"
