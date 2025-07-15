"""Memory‑safe GPU inference with Drain3 mapping + LogBERT‑VHM distance.
Run:
  python inference_vhm.py --log_file anomaly.log --drain_state drain_state.bin \
      --window 50 --stride 10 --batch_size 512 [--threshold 3.0]
Outputs: inference_results.csv, anomalous_log_lines.txt
"""
import argparse, csv, torch, os
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig
from transformers import BertTokenizerFast
from logbert_vhm_model import LogBERT_VHM

# ---- helper ----

def load_drain(state_path):
    cfg = TemplateMinerConfig()
    return TemplateMiner(config=cfg, persistence_handler=FilePersistence(state_path))

def build_sequences(eids, window, stride):
    seqs, mapping = [], []
    for start in range(0, len(eids) - window + 1, stride):
        chunk = eids[start:start+window]
        toks = []
        for i, t in enumerate(chunk):
            toks.append(t)
            if i < len(chunk)-1:
                toks.append('[DIST]')
        seqs.append(toks)
        mapping.append((start, start+window))
    return seqs, mapping

# ---- main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_file', required=True)
    p.add_argument('--drain_state', required=True)
    p.add_argument('--window', type=int, default=50)
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--threshold', type=float)
    a = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    miner = load_drain(a.drain_state)
    with open(a.log_file) as f:
        raw = [l.rstrip() for l in f if l.strip()]
    eids = [str(miner.add_log_message(l)['cluster_id']) for l in raw]

    window = min(a.window, len(eids))
    seqs, idx = build_sequences(eids, window, a.stride)

    tok = BertTokenizerFast.from_pretrained('./logbert-vhm-finetuned')
    model = LogBERT_VHM('./logbert-vhm-finetuned').to(device)
    model.center.data = torch.load('logbert-vhm-finetuned/vhm_center.pt', map_location=device)['center']
    model.bert.resize_token_embeddings(len(tok))
    model.eval()

    batch, dists_all = a.batch_size, []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            sub = seqs[i:i+batch]
            inp = tok(sub, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
            inp = {k: v.to(device) for k, v in inp.items()}
            emb = model(inp['input_ids'], inp['attention_mask'])
            d = torch.sum((emb - model.center)**2, dim=1).cpu()
            dists_all.append(d)
    dists = torch.cat(dists_all)

    thr = a.threshold or torch.quantile(dists, 0.95).item()
    preds = (dists > thr).long().tolist()
    print(f'Threshold: {thr:.4f} | flagged {sum(preds)} / {len(preds)} sequences')

    with open('inference_results.csv','w', newline='') as f:
        csv.writer(f).writerows([["seq_id","anom","dist"]] + list(zip(range(len(preds)), preds, map(float, dists))))

    with open('anomalous_log_lines.txt','w') as f:
        for (s,e), p_ in zip(idx, preds):
            if p_:
                f.write('\n' + '='*80 + f"\n# Seq {s}-{e-1} ANOMALY\n")
                f.writelines(raw[s:e])

if __name__ == '__main__' and os.path.basename(__file__) == 'inference_vhm.py':
    main()
