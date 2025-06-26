import pandas as pd
from sklearn.model_selection import train_test_split

def load_structured_logs(structured_path):
    df = pd.read_csv(structured_path)
    return df

def insert_dist_token(sequence, dist_token='[DIST]'):
    new_seq = []
    for i, token in enumerate(sequence):
        new_seq.append(token)
        if i < len(sequence) - 1:
            new_seq.append(dist_token)
    return new_seq

def group_by_session(df, window_size=20):
    sequences = []
    buffer = []
    for _, row in df.iterrows():
        buffer.append(row['EventId'])
        if len(buffer) >= window_size:
            sequences.append(insert_dist_token(buffer[:]))
            buffer = []
    return sequences

def split_data(sequences, labels, test_size=0.2):
    return train_test_split(sequences, labels, test_size=test_size, random_state=42)

