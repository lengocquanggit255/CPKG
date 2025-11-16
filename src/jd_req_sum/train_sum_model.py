import os
import random
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AdamW,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    set_seed,
)

# ==========================
# Hyperparameters (theo LaTeX bạn đưa)
# ==========================
MODEL_CHECKPOINT = "google/t5-large"
MAX_INPUT_LENGTH = 512          
MAX_TARGET_LENGTH = 64          
LEARNING_RATE = 2e-4            
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4                  
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 8
WARMUP_STEPS = 500
SEED = 42
EARLY_STOPPING_PATIENCE = 3     

TASK_PREFIX = "summarize skill: "  


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_file",
        type=str,
        default="CPKG/data/req_sum.jsonl",
        help="Đường dẫn tới JSONL, mỗi dòng: {\"input_text\": ..., \"target_text\": ...}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./t5_req_sum_checkpoints",
        help="Thư mục lưu checkpoint và final model",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ validation split (mặc định 0.1 = 10%)",
    )

    return parser.parse_args()


def set_all_seeds(seed: int = 42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_all_seeds(SEED)

    # ==========================
    # 1. Load dataset từ JSONL
    # ==========================
    # File req_sum.jsonl dạng:
    # {"input_text": "...", "target_text": "..."}
    raw_datasets = load_dataset(
        "json",
        data_files={"data": args.data_file},
    )

    dataset_split = raw_datasets["data"].train_test_split(
        test_size=args.val_ratio,
        seed=SEED,
    )
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(eval_dataset))

    # ==========================
    # 2. Load tokenizer & model T5-Large
    # ==========================
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    # ==========================
    # 3. Preprocess function
    # ==========================
    def preprocess_function(examples):
        # Thêm prefix để giúp model biết đây là task gì (optional)
        inputs = [TASK_PREFIX + x for x in examples["input_text"]]
        targets = examples["target_text"]

        # Tokenize input
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
        )

        # Tokenize target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_TARGET_LENGTH,
                padding="max_length",
                truncation=True,
            )

        # Thay pad_token_id bằng -100 để loss bỏ qua padding
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(lid if lid != tokenizer.pad_token_id else -100) for lid in label]
            for label in labels_ids
        ]
        model_inputs["labels"] = labels_ids

        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # ==========================
    # 4. Data collator
    # ==========================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
    )

    # ==========================
    # 5. TrainingArguments
    # ==========================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        fp16=torch.cuda.is_available(),  # nếu GPU hỗ trợ mixed precision
        report_to="none",                # bật wandb/mlflow nếu cần
    )

    # ==========================
    # 6. AdamW optimizer (đúng yêu cầu)
    # ==========================
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
    )

    # ==========================
    # 7. Trainer + EarlyStopping
    # ==========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # scheduler = None => Trainer tạo linear scheduler
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.0,
            )
        ],
    )

    # ==========================
    # 8. Train
    # ==========================
    trainer.train()

    # ==========================
    # 9. Save final model
    # ==========================
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nTraining finished. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
