#!/usr/bin/env python3
"""Prepare/train native-intensity seeds using the shared seed-study workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_inferred_seeds as shared


def build_configs(protocol, root=shared.ROOT):
    return shared.build_configs(protocol, root=root, variant='native')


if __name__ == '__main__':
    shared.main(config_builder=build_configs, description=__doc__)
