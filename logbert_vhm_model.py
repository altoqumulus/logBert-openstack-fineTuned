import torch
import torch.nn as nn
from transformers import BertModel

class LogBERT_VHM(nn.Module):
    def __init__(self, model_name, embedding_size=768):
        super(LogBERT_VHM, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_size = embedding_size
        self.center = nn.Parameter(torch.randn(embedding_size), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        return cls_output

    def compute_vhm_loss(self, cls_embeddings):
        return torch.mean(torch.sum((cls_embeddings - self.center) ** 2, dim=1))

    def set_center(self, dataloader, device):
        self.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                cls_emb = self.forward(input_ids, attention_mask)
                all_embeddings.append(cls_emb)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        self.center.data = torch.mean(all_embeddings, dim=0).to(device)