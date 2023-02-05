#!/usr/bin/env bash

VENVNAME=glassenv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
pip --version

sudo apt-get -y install python3-dev graphviz libgraphviz-dev pkg-config
sudo apt-get -y install python3-graph-tool

pip install graphviz
pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt
pip install pygraphviz

deactivate
echo "build $VENVNAME"
