"""Seed the existing evaluator after importing it (imports reset some RNGs)."""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seed_support import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--script', required=True)
    args, remaining = parser.parse_known_args()
    spec = importlib.util.spec_from_file_location('seeded_benchmark', args.script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    seed_everything(args.seed)
    sys.argv = [args.script] + (remaining[1:] if remaining[:1] == ['--'] else remaining)
    module.main()


if __name__ == '__main__':
    main()
