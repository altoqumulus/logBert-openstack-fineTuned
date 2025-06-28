"""Generate a Drain3 state (.bin) from normal log **file OR directory**.
Usage:
  python build_drain_state.py --log_file normal.log --out drain_state.bin
  python build_drain_state.py --log_dir  /logs/normal/ --out drain_state.bin
"""
import os, gzip, argparse
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

def _iter_lines(path: str):
    if os.path.isfile(path):
        op = gzip.open if path.endswith('.gz') else open
        with op(path, errors='ignore') as f:
            for ln in f: yield ln.strip()
    else:
        for root, _, files in os.walk(path):
            for fn in files:
                full = os.path.join(root, fn)
                op = gzip.open if fn.endswith('.gz') else open
                with op(full, errors='ignore') as f:
                    for ln in f: yield ln.strip()

def build_state(src: str, out_bin: str):
    cfg = TemplateMinerConfig()
    pers = FilePersistence(out_bin)
    miner = TemplateMiner(config=cfg, persistence_handler=pers)
    for line in _iter_lines(src):
        miner.add_log_message(line)
    miner.save_state(snapshot_reason="manual_save")
    print("Drain state saved to", out_bin)

if __name__ == '__main__' and os.path.basename(__file__) == 'build_drain_state.py':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_file', help='Single normal log file')
    ap.add_argument('--log_dir',  help='Directory of normal logs')
    ap.add_argument('--out',      required=True, help='Output drain_state.bin')
    arg = ap.parse_args()
    src = arg.log_file or arg.log_dir
    if not src: ap.error('need --log_file or --log_dir')
    build_state(src, arg.out)
