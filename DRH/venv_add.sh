#!/usr/bin/env bash
VENVNAME=glassenv
source $VENVNAME/bin/activate
python -m ipykernel install --user --name $VENVNAME --display-name "$VENVNAME"
