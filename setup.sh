#!/bin/bash
set -e

python3.11 -m venv venv
venv/bin/pip install --prefer-binary -r requirements.txt
echo "Done. Run with: venv/bin/python split_gig.py path/to/gig.mov"
