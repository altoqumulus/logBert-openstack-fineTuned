from data_loader import load_structured_logs, group_by_session, split_data
from finetune_vhm import finetune_vhm

STRUCTURED_LOG_PATH = "OpenStack_2k.log_structured.csv"
#STRUCTURED_LOG_PATH = "OpenStack_2k.log_templates.csv"
PRETRAINED_MODEL_PATH = "./logbert-pretrained"

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    STRUCTURED_LOG_PATH = "OpenStack_2k.log_structured.csv"
  
    WINDOW = 5
    df = load_structured_logs(STRUCTURED_LOG_PATH)

    # Build sequences and aligned labels
    def build_sequences_labels(df, window_size=5):
        seqs, lbls = [], []
        buf_events, buf_lbls = [], []
        for _, row in df.iterrows():
            buf_events.append(row['EventId'])
            buf_lbls.append(row.get('Label', 0))
            if len(buf_events) >= window_size:
                seq = []
                for i, tok in enumerate(buf_events):
                    seq.append(tok)
                    if i < len(buf_events) - 1:
                        seq.append('[DIST]')
                seqs.append(seq)
                lbls.append(int(max(buf_lbls)))
                buf_events, buf_lbls = [], []
        return seqs, lbls

    sequences, labels = build_sequences_labels(df, WINDOW)

    train_seqs, val_seqs, train_labels, val_labels = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        stratify=labels if sum(labels) else None,
        random_state=42,
    )

    vocab = sorted({tok for seq in sequences for tok in seq})
   
    finetune_vhm(
        train_sequences=train_seqs,
        train_labels=train_labels,
        val_sequences=val_seqs,
        val_labels=val_labels,
        model_path="./logbert-pretrained",
        vocab=vocab,
        batch_size=16,
        epochs=50,
    )
