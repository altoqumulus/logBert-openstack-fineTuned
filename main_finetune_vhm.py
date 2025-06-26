from data_loader import load_structured_logs, group_by_session, split_data
from finetune_vhm import finetune_vhm

STRUCTURED_LOG_PATH = "OpenStack_2k.log_structured.csv"
PRETRAINED_MODEL_PATH = "./logbert-pretrained"

if __name__ == '__main__':
    df = load_structured_logs(STRUCTURED_LOG_PATH)
    sequences = group_by_session(df, window_size=20)
    labels = []
    buffer = []
    session_labels = []
    for i in range(len(sequences)):
        start = i * 20
        end = start + 20
        if end <= len(df):
            session_df = df.iloc[start:end]
            label = int(session_df['Label'].max()) if 'Label' in session_df.columns else 0
            session_labels.append(label)
    labels = session_labels

    train_seqs, val_seqs, train_labels, val_labels = split_data(sequences, labels)
    vocab = sorted(set(token for seq in sequences for token in seq))
    finetune_vhm(train_seqs, train_labels, val_seqs, val_labels, PRETRAINED_MODEL_PATH, vocab)