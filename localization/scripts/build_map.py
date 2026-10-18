#!/usr/bin/env python3
"""Build the relocalization map from laps of the training bag (uses GT; D-009, D-012).

    python scripts/build_map.py                               # cameras and laps from the config
    python scripts/build_map.py --cams fl r rl --laps L3 L6 --name diag --spacing 4
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camloc import config  # noqa: E402
from camloc.reloc import mapbuild  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--cams', nargs='+')
    ap.add_argument('--laps', nargs='+', help='default: split.train')
    ap.add_argument('--name', default='map')
    ap.add_argument('--spacing', type=float)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--set', nargs='*', default=[], help='config overrides, e.g. reloc.map.min_parallax_deg=0.5')
    args = ap.parse_args()
    cfg = config.apply_overrides(config.load(args.config), args.set)
    cams = args.cams or cfg['reloc']['cameras']
    lap_names = args.laps or cfg['split']['train']
    laps = {lap['name']: lap for lap in json.load(open(config.artifact(cfg, 'laps.json')))['laps']}
    assert not set(lap_names) & set(cfg['split']['val']), 'validation laps must not be in the map'
    windows = [(laps[n]['start_ns'], laps[n]['end_ns']) for n in lap_names]
    out = config.artifact(cfg, args.name, '{cam}.npz')
    t = time.time()
    print(f'building map "{args.name}" for {cams} from {lap_names}')
    mapbuild.build(cfg, cams, windows, args.spacing or cfg['reloc']['map']['spacing_m'], out, args.workers)
    with open(config.artifact(cfg, args.name, 'meta.json'), 'w') as f:
        json.dump({'cameras': cams, 'laps': lap_names, 'spacing_m': args.spacing or cfg['reloc']['map']['spacing_m'],
                   'map_cfg': cfg['reloc']['map'], 'features': cfg['reloc']['features']}, f, indent=1)
    print(f'done in {time.time() - t:.0f} s')


if __name__ == '__main__':
    main()
