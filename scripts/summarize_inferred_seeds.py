"""Aggregate one completed checkpoint per seed, keeping seeds as replicates."""
import argparse
import json
from pathlib import Path
import statistics


def aggregate(records, expected_seeds):
    seeds = [r['seed'] for r in records]
    if len(seeds) != len(set(seeds)) or set(seeds) != set(expected_seeds):
        raise ValueError('Require exactly one summary per expected seed')
    keys = set(records[0]['metrics'])
    if any(set(r['metrics']) != keys for r in records):
        raise ValueError('Seeds have different metrics/conditions')
    return {key: {'n_seeds': len(records),
                  'mean': statistics.mean(r['metrics'][key] for r in records),
                  'sd': statistics.stdev(r['metrics'][key] for r in records) if len(records) > 1 else None,
                  'by_seed': {str(r['seed']): r['metrics'][key] for r in records}}
            for key in sorted(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    records = [json.loads(Path(p).read_text()) for p in args.summaries]
    result = {'checkpoints': [{k: r[k] for k in ('seed', 'epoch', 'checkpoint', 'checkpoint_sha256')}
                              for r in records], 'metrics': aggregate(records, args.seeds)}
    with open(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
