# logbert_openstack_pipeline: Extended with VHM Fine-tuning + Pretraining + GPU Support

# === main_pretrain.py ===
from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pandas as pd
from data_loader import load_structured_logs, group_by_session, split_data, insert_dist_token

class MLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        inputs = tokenizer(sequences, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
        self.inputs = inputs
        self.labels = inputs.input_ids.clone()

        rand = torch.rand(self.labels.shape)
        mask_arr = (rand < self.mask_prob) * (self.labels != tokenizer.pad_token_id) * \
                   (self.labels != tokenizer.cls_token_id) * (self.labels != tokenizer.sep_token_id)

        for i in range(self.labels.shape[0]):
            for j in range(self.labels.shape[1]):
                if mask_arr[i, j]:
                    self.inputs.input_ids[i, j] = tokenizer.mask_token_id

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs.input_ids[idx],
            'attention_mask': self.inputs.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    STRUCTURED_LOG_PATH = "OpenStack_2k.log_structured.csv"
    df = load_structured_logs(STRUCTURED_LOG_PATH)
    sequences = group_by_session(df, window_size=20)
    labels = [0] * len(sequences)  # dummy labels since MLM doesn't need true labels
    train_seqs, _, _, _ = split_data(sequences, labels, test_size=0.2)
    vocab = sorted(set(token for seq in train_seqs for token in seq))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.add_tokens(vocab + ['[DIST]'])

    train_dataset = MLMDataset(train_seqs, tokenizer)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./logbert-pretrained",
        overwrite_output_dir=True,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=1,
        prediction_loss_only=True,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./logbert-pretrained")
    tokenizer.save_pretrained("./logbert-pretrained")
