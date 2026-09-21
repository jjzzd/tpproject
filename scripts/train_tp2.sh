#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc-per-node=2 -m mmtrain.train --config configs/tp2.yaml

