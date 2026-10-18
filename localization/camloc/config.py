"""YAML config with paths resolved relative to the localization/ folder."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _merge(base, over):
    for k, v in over.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load(path=None):
    """Load a config; `extends: other.yaml` (relative to the file) is loaded first and deep-merged."""
    path = path or os.path.join(ROOT, 'configs', 'iter1.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'extends' in cfg:
        base = load(os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop('extends')))
        cfg = _merge(base, cfg)
    for k in ('bag', 'official_eval', 'artifacts', 'results'):
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(ROOT, cfg[k]))
    os.makedirs(cfg['artifacts'], exist_ok=True)
    os.makedirs(cfg['results'], exist_ok=True)
    return cfg


def apply_overrides(cfg, overrides):
    """In-place 'a.b.c=value' overrides (value cast to the type of the existing entry)."""
    for kv in overrides or []:
        key, value = kv.split('=', 1)
        d = cfg
        *path, last = key.split('.')
        for p in path:
            d = d[p]
        old = d[last]
        if isinstance(old, bool):
            d[last] = value.lower() in ('1', 'true', 'yes')
        elif isinstance(old, list):
            d[last] = yaml.safe_load(value)
        else:
            d[last] = type(old)(value)
    return cfg


def artifact(cfg, *parts):
    p = os.path.join(cfg['artifacts'], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
