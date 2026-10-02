#!/bin/bash

tmux new -s training
uv run sonar.py

# detach with `Ctrl + b`; `d`
# attach with
# tmux attach -t training
