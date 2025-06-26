from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch
from logbert_vhm_model import LogBERT_VHM
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class VHMLogDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.encodings = tokenizer(sequences, is_split_into_words=True, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def finetune_vhm(
    train_sequences,
    train_labels,
    val_sequences,
    val_labels,
    model_path,
    vocab,
    batch_size: int = 16,
    epochs: int = 50,
):
    """Fine‑tune LogBERT with the VHM objective **and** plot:
    • Training‑loss curve
    • Precision / Recall / F1 per epoch
    Saved figures: `vhm_train_loss.png`, `vhm_metrics.png`.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- Data & tokenizer -----------------
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    tokenizer.add_tokens(['[DIST]'])
    train_dataset = VHMLogDataset(train_sequences, train_labels, tokenizer)
    val_dataset = VHMLogDataset(val_sequences, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---------------- Model -----------------
    model = LogBERT_VHM(model_path).to(device)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.set_center(train_loader, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # ---------------- Tracking lists -----------------
    loss_curve, prec_curve, rec_curve, f1_curve = [], [], [], []

    # ---------------- Training loop -----------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask)
            loss = model.compute_vhm_loss(embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_curve.append(avg_loss)

        # ----------- Validation per epoch -----------
        model.eval()
        all_preds, all_labels_epoch = [], []
        with torch.no_grad():
            # compute robust threshold once per epoch using training data distances
            # (mean + 3*std)
            train_dists = []
            for b in train_loader:
                d = torch.sum(
                    (model(b['input_ids'].to(device), b['attention_mask'].to(device)) - model.center) ** 2,
                    dim=1,
                )
                train_dists.extend(d.cpu())
            train_dists_tensor = torch.stack(train_dists)
            threshold = torch.mean(train_dists_tensor) + 3 * torch.std(train_dists_tensor)

            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                cls_embeddings = model(input_ids, attention_mask)
                distances = torch.sum((cls_embeddings - model.center) ** 2, dim=1)
                preds = (distances > threshold).long()

                all_preds.extend(preds.cpu().tolist())
                all_labels_epoch.extend(labels.cpu().tolist())

        # Compute metrics for this epoch
        precision = precision_score(all_labels_epoch, all_preds, zero_division=0)
        recall = recall_score(all_labels_epoch, all_preds, zero_division=0)
        f1 = f1_score(all_labels_epoch, all_preds, zero_division=0)
        prec_curve.append(precision)
        rec_curve.append(recall)
        f1_curve.append(f1)

        print(
            f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} | "
            f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}"
        )

    # ---------------- Save model -----------------
    out_dir = "./logbert-vhm-finetuned"
    # Save the underlying BERT weights so they can be re‑loaded with from_pretrained
    model.bert.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Persist the learned VHM center so it can be re‑attached after loading
    torch.save({"center": model.center.detach().cpu()}, f"{out_dir}/vhm_center.pt")

    # ---------------- Plotting -----------------
    # 1️⃣  Loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_curve, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("VHM Loss")
    plt.title("Training Loss Curve (VHM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vhm_train_loss.png")

    # 2️⃣  Precision / Recall / F1 curves
    plt.figure()
    plt.plot(range(1, epochs + 1), prec_curve, marker="o", label="Precision")
    plt.plot(range(1, epochs + 1), rec_curve, marker="o", label="Recall")
    plt.plot(range(1, epochs + 1), f1_curve, marker="o", label="F1‑Score")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics (VHM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vhm_metrics.png")

    print("Plots saved as vhm_train_loss.png and vhm_metrics.png")
