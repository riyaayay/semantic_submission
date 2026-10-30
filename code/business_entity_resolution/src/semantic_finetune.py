"""
Guard-railed fine-tuning of the multilingual semantic encoder (v10).

Guard rails (all enforced here):
1. The original model stays FROZEN and its features are kept; the fine-tuned
   model only adds extra features.
2. Light fine-tuning: LoRA adapters (rank 8) on the attention query/value
   projections with a low learning rate; the base weights never change.
3. Distillation anchor: every step also penalises drift of the student's
   embeddings away from the frozen teacher's on multilingual text, so the
   50+ language alignment is preserved.
4. Gate: a French<->English (and Hindi<->English) business-term test is run
   before and after; the fine-tuned adapter is REJECTED if cross-language
   retrieval accuracy or mean pair similarity drops.

Training signal: in-batch contrastive loss (InfoNCE) on true-match name pairs
from the labelled training countries.
"""

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gpu import require_cuda

# Cross-language gate: (French or Hindi, English) business terms with the same meaning.
CROSS_LANGUAGE_GATE: List[Tuple[str, str]] = [
    ("Pharmacie Centrale", "Central Pharmacy"), ("Boulangerie Dupont", "Dupont Bakery"),
    ("Garage Martin", "Martin Garage"), ("Cabinet Dentaire", "Dental Practice"),
    ("École Primaire Jules Ferry", "Jules Ferry Primary School"), ("Librairie du Centre", "Downtown Bookshop"),
    ("Boucherie Moderne", "Modern Butcher Shop"), ("Fleuriste des Lilas", "Lilac Florist"),
    ("Transports Rapides", "Fast Transport"), ("Agence Immobilière", "Real Estate Agency"),
    ("Clinique Vétérinaire", "Veterinary Clinic"), ("Salon de Coiffure", "Hair Salon"),
    ("Cave à Vins", "Wine Cellar"), ("Société Générale de Construction", "General Construction Company"),
    ("Laboratoire d'Analyses Médicales", "Medical Testing Laboratory"), ("Pâtisserie Royale", "Royal Pastry Shop"),
    ("Menuiserie Bois", "Wood Carpentry"), ("Plomberie Chauffage", "Plumbing and Heating"),
    ("Auto-École Liberté", "Liberty Driving School"), ("Hôtel de la Gare", "Station Hotel"),
    ("Cabinet d'Avocats", "Law Firm"), ("Bureau d'Études Techniques", "Engineering Consultancy"),
    ("Association Sportive", "Sports Association"), ("Comité des Fêtes", "Festival Committee"),
    ("Banque Populaire", "People's Bank"), ("Assurances Mutuelles", "Mutual Insurance"),
    ("Imprimerie Nouvelle", "New Printing House"), ("Nettoyage Industriel", "Industrial Cleaning"),
    ("Électricité Générale", "General Electrical Services"), ("Maison de Retraite", "Retirement Home"),
    ("मॉडर्न फाइनेंस", "Modern Finance"), ("राज ट्रेडर्स", "Raj Traders"),
    ("श्री गणेश इलेक्ट्रिकल्स", "Shri Ganesh Electricals"), ("भारत स्टील", "Bharat Steel"),
    ("सिटी हॉस्पिटल", "City Hospital"), ("ग्लोबल एक्सपोर्ट्स", "Global Exports"),
]


@torch.no_grad()
def gate_metrics(model, pairs=CROSS_LANGUAGE_GATE) -> dict:
    """Cross-language retrieval accuracy (each source term -> nearest English term) and mean pair cosine."""
    src = model.encode([a for a, _ in pairs], convert_to_tensor=True, normalize_embeddings=True).float()
    tgt = model.encode([b for _, b in pairs], convert_to_tensor=True, normalize_embeddings=True).float()
    sim = src @ tgt.T
    acc = (sim.argmax(1) == torch.arange(len(pairs), device=sim.device)).float().mean().item()
    return {"retrieval_acc": acc, "mean_pair_cos": sim.diag().mean().item()}


def finetune(train_pairs: List[Tuple[str, str]], anchor_texts: List[str], out_dir: str,
             base_dir: str = None, epochs: int = 1, batch: int = 256, lr: float = 1e-4,
             lora_r: int = 8, distill_weight: float = 1.0, temperature: float = 0.05,
             log=print) -> dict:
    """LoRA contrastive fine-tuning with a distillation anchor; saves the adapter only if the gate passes."""
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer
    from semantic import DEFAULT_MODEL_DIR

    dev = require_cuda()
    base_dir = os.path.abspath(base_dir or DEFAULT_MODEL_DIR)
    teacher = SentenceTransformer(base_dir, device=str(dev)).eval()       # frozen original
    for p in teacher.parameters():
        p.requires_grad_(False)
    before = gate_metrics(teacher)
    log(f"  gate before fine-tuning: {before}")

    student = SentenceTransformer(base_dir, device=str(dev))
    lora = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, target_modules=["query", "value"])
    # get_peft_model injects LoRA layers into the module in place; keep the wrapper to save the adapter only
    peft_model = get_peft_model(student[0].auto_model, lora)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    log(f"  LoRA: training {trainable:,} of {total:,} parameters ({trainable / total:.2%})")
    student.max_seq_length = teacher.max_seq_length = 64
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr)

    def emb(model, texts):
        feats = model.tokenize(texts)
        feats = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in feats.items()}
        return F.normalize(model(feats)["sentence_embedding"], dim=-1)

    rng = np.random.default_rng(42)
    student.train()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(train_pairs))
        t0 = time.time()
        for lo in range(0, len(order) - batch + 1, batch):
            b = [train_pairs[i] for i in order[lo:lo + batch]]
            anchors = [anchor_texts[i] for i in rng.integers(0, len(anchor_texts), batch // 2)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qa = emb(student, [x for x, _ in b])
                qb = emb(student, [y for _, y in b])
                logits = (qa @ qb.T).float() / temperature
                labels = torch.arange(len(b), device=dev)
                contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                with torch.no_grad():
                    ta = emb(teacher, anchors)
                sa = emb(student, anchors)
                distill = (1 - (sa.float() * ta.float()).sum(-1)).mean()     # cosine drift from teacher
                loss = contrastive + distill_weight * distill * 100
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                log(f"    step {step}: contrastive {contrastive.item():.4f}  drift {distill.item():.5f}  "
                    f"({time.time() - t0:.0f}s)")
    student.eval()
    after = gate_metrics(student)
    passed = (after["retrieval_acc"] >= before["retrieval_acc"]) and (after["mean_pair_cos"] >= before["mean_pair_cos"] - 0.01)
    log(f"  gate after fine-tuning:  {after}  ->  {'PASS' if passed else 'REJECTED'}")
    reloaded = None
    if passed:
        os.makedirs(out_dir, exist_ok=True)
        peft_model.save_pretrained(out_dir)                  # LoRA adapter weights only (small)
        reloaded = gate_metrics(load_finetuned(out_dir, base_dir))
        log(f"  gate after save + reload: {reloaded}")
        if abs(reloaded["mean_pair_cos"] - after["mean_pair_cos"]) > 0.005:
            raise RuntimeError("Reloaded adapter does not reproduce the fine-tuned embeddings")
    return {"before": before, "after": after, "reloaded": reloaded, "passed": passed, "trainable_params": trainable}


def load_finetuned(adapter_dir: str, base_dir: str = None):
    """Base model + saved LoRA adapter (merged for fast inference)."""
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from semantic import DEFAULT_MODEL_DIR
    dev = require_cuda()
    model = SentenceTransformer(os.path.abspath(base_dir or DEFAULT_MODEL_DIR), device=str(dev))
    model[0].auto_model = PeftModel.from_pretrained(model[0].auto_model, adapter_dir).merge_and_unload()
    model.half()
    model.max_seq_length = 64
    return model.eval()
