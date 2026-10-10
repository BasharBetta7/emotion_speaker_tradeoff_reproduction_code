"""Create portable training configs from recorded run configs; never train."""
from pathlib import Path
import argparse,hashlib,json,os,sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['ievc','nevc','nevc-sc'],required=True)
    parser.add_argument('--dataset',choices=['esd','crema_d'],required=True)
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    if not args.run_name.replace('-','').replace('_','').isalnum():parser.error('Use a simple run name')
    os.chdir(ROOT)
    from seed_support import check_splits,sha256
    for seed in [17,43,123]:
        source=ROOT/'seeded/configs-recorded'/f'{args.model}-{args.dataset}-{seed}.yml'
        cfg=yaml.safe_load(source.read_text())
        assert cfg['seed']==seed and not cfg.get('pretrained_model')
        cfg['log_dir']=str(ROOT/'Models/seed_reproduction'/args.run_name/f'seed_{seed}')
        cfg['wandb']={'mode':'disabled'}
        for key in ['input_sha256','code_sha256','split_audit','_resolved_config']:cfg.pop(key,None)
        cfg['recorded_config_sha256']=sha256(source)
        cfg['split_audit']=check_splits(cfg)
        paths=[cfg['train_data'],cfg['val_data']]+[v for field in ['validation_manifests','test_manifests'] for v in cfg['seed_protocol'][field].values()]
        cfg['input_sha256']={path:sha256(Path(path)) for path in paths}
        cfg['code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'train.py',ROOT/'trainer.py',ROOT/'seed_support.py']}
        out=ROOT/'Configs'/f'reproduction_{args.run_name}_{seed}.yml'
        if Path(cfg['log_dir']).exists():raise FileExistsError(cfg['log_dir'])
        text=yaml.safe_dump(cfg,sort_keys=False)
        if out.exists() and out.read_text()!=text:raise FileExistsError(out)
        out.write_text(text);print(out.relative_to(ROOT))
if __name__=='__main__':main()
