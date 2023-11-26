from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer, MarianConfig, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,  PreTrainedTokenizerFast, AutoTokenizer
import torch
import evaluate
import numpy as np
from configs import *

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=MODEL_MAX_LENGTH)

books = load_dataset(DATASET_NAME)

#TODO: Find vocab_size
VOCAB_SIZE = 63261

books = books["train"].train_test_split(test_size=0.2)

pre_trained_config = {
  "_num_labels": 3,
  "activation_dropout": 0.0,
  "activation_function": "swish",
  "add_bias_logits": False,
  "add_final_layer_norm": False,
  "architectures": [
    "MarianMTModel"
  ],
  "attention_dropout": 0.0,
  "bad_words_ids": [
    [
      63260
    ]
  ],
  "bos_token_id": 0,
  "classif_dropout": 0.0,
  "classifier_dropout": 0.0,
  "d_model": 512,
  "decoder_attention_heads": 8 if TRANSFER_LEARNING else HEADS,
  "decoder_ffn_dim": 2048,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 6 if TRANSFER_LEARNING else DECODER_LAYERS,
  "decoder_start_token_id": VOCAB_SIZE - 1,
  "decoder_vocab_size": VOCAB_SIZE,
  "dropout": 0.1,
  "encoder_attention_heads": 8 if TRANSFER_LEARNING else HEADS,
  "encoder_ffn_dim": 2048,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 6 if TRANSFER_LEARNING else ENCODER_LAYERS,
  "eos_token_id": 0,
  "forced_eos_token_id": 0,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "init_std": 0.02,
  "is_encoder_decoder": True,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "max_length": 512,
  "max_position_embeddings": 512,
  "model_type": "marian",
  "normalize_before": False,
  "normalize_embedding": False,
  "num_beams": 4,
  "num_hidden_layers": 6 if TRANSFER_LEARNING else HIDDEN_LAYERS,
  "pad_token_id": VOCAB_SIZE - 1,
  "scale_embedding": True,
  "share_encoder_decoder_embeddings": True,
  "static_position_embeddings": True,
  "transformers_version": "4.35.2",
  "use_cache": True,
  "vocab_size": VOCAB_SIZE
}

configuration = MarianConfig(**pre_trained_config)

model = MarianMTModel(configuration).to(DEVICE)

tuned_model = MarianMTModel.from_pretrained(BASE_MODEL).to(DEVICE)

parts = list(dict(model.named_parameters()).keys())
parts.remove('model.shared.weight')

def transfer (tuned, to_tune, parts):
  target = dict(to_tune.named_parameters())
  source = dict(tuned.named_parameters())

  for part in parts:
    target[part].data.copy_(source[part].data)

if TRANSFER_LEARNING:
  transfer(tuned_model, model, parts)

source_lang = "es"
target_lang = "fi"


def preprocess_function(examples):
    inputs = [example[SOURCE_LANG] for example in examples["translation"]]
    targets = [example[TARGET_LANG] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model",
    evaluation_strategy="epoch",
    learning_rate=INIT_LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

#fast_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
