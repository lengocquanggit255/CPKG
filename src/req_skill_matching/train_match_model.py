import warnings
from transformers import logging
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import torch


# ============================================================
# 1. SETUP
# ============================================================
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned")

set_seed(42)  # random seed

# Login
login(token="YOUR_HF_TOKEN")
wandb.login(key="YOUR_WANDB_KEY")
wandb.init(project="YOUR_PROJECT_NAME")


# ============================================================
# 2. LOAD DATA
# ============================================================

raw_data = load_dataset(
    "json",
    data_files="CPKG/data/req_skill_matching.jsonl",
)

# Chia train/val/test 80/10/10
split_data = raw_data["train"].train_test_split(test_size=0.2, seed=42)
temp = split_data["test"].train_test_split(test_size=0.5, seed=42)

data = DatasetDict({
    "train": split_data["train"],
    "val": temp["train"],
    "test": temp["test"]
})

print("Sample skill:", data['train']['skill'][0])
print("Sample req:", data['train']['req'][0])
print("Sample label:", data['train']['label'][0])


# ============================================================
# 3. TOKENIZER
# ============================================================
checkpoint = "vinai/phobert-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)


# ============================================================
# 4. PREPROCESSING
# ============================================================
def preprocess_function(examples):
    # concatenation skill + req
    # Format: [CLS] req [SEP] skill [SEP]
    model_inputs = tokenizer(
        examples["req"],
        examples["skill"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = [int(l) for l in examples["label"]]
    return model_inputs


tokenized_data = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_data = tokenized_data.select_columns([
    "input_ids", "attention_mask", "labels"
])


# ============================================================
# 5. METRICS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# 6. MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)


# ============================================================
# 7. TRAINING ARGS (CHUẨN THEO THAM SỐ BẠN ĐƯA)
# ============================================================
training_args = TrainingArguments(
    output_dir="phobert-matching-model",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    per_device_train_batch_size=4,     
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,     

    learning_rate=2e-5,                
    weight_decay=0.01,
    warmup_steps=500,                  
    lr_scheduler_type="linear",        

    num_train_epochs=8,                
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,

    fp16=torch.cuda.is_available(),
    seed=42,

    logging_dir="./logs",
    logging_steps=20,
    report_to=["wandb"],

    push_to_hub=True,
    hub_model_id="YOUR_HF_USERNAME/req-skill-matching-phobert-large",
)


# ============================================================
# 8. TRAINER WITH EARLY STOPPING
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # REQUIRED
)


# ============================================================
# 9. TRAIN
# ============================================================
trainer.train()


# ============================================================
# 10. TEST
# ============================================================
test_results = trainer.evaluate(eval_dataset=tokenized_data["test"])
print("Final Test Results:", test_results)


# ============================================================
# 11. PUSH MODEL
# ============================================================
trainer.push_to_hub()
tokenizer.push_to_hub("YOUR_HF_USERNAME/req-skill-matching-phobert-large")

print("🎉 DONE! Model pushed to HuggingFace Hub.")
