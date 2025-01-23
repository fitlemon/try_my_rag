import os

os.chdir("..")

key = os.getenv("HF_KEY")

from datasets import load_dataset

# Load dataset from Hugging Face Hub
dataset = load_dataset("fitlemon/rag-labor-codex-dataset")

training_dataset = dataset["train"]
test_dataset = dataset["test"]


training_dataset = training_dataset.train_test_split(test_size=0.1)


# save datasets to disk
training_dataset["train"].to_json("data/train_dataset.json", orient="records")
training_dataset["test"].to_json("data/val_dataset.json", orient="records")


import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets

model_id = "BAAI/bge-m3"  # Hugging Face model ID
matryoshka_dimensions = [768, 512, 256, 128, 64]

model = SentenceTransformer(
    model_id, device="cuda" if torch.cuda.is_available() else "cpu"
)

val_dataset = load_dataset("json", data_files="data/val_dataset.json", split="train")
train_dataset = load_dataset(
    "json", data_files="data/train_dataset.json", split="train"
)


corpus = val_dataset["chunk"]
queries = val_dataset["question"]

# corpus ids as indexes of list
corpus = dict(zip(map(str, range(len(corpus))), corpus))  # Our corpus (cid => document)
queries = dict(
    zip(map(str, range(len(queries))), queries)
)  # Our queries (qid => question)

# relevant docs as indexes of list
relevant_docs = {}
for qid, corpus_id in zip(queries.keys(), corpus.keys()):
    relevant_docs[qid] = {corpus_id}


matryoshka_evaluators = []
# Iterate over the different dimensions
for dim in matryoshka_dimensions:
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,  # Truncate the embeddings to a certain dimension
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)

# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)


# Evaluate the model
results = evaluator(model)

# # COMMENT IN for full results
# print(results)

# Print the main score
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print
    print(f"{key}: {results[key]}")


from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer


model_id = "BAAI/bge-m3"

# load model with SDPA for using Flash Attention 2
model = SentenceTransformer(
    model_id,
    model_kwargs={"attn_implementation": "sdpa"},
    model_card_data=SentenceTransformerModelCardData(
        language="uz",
        license="apache-2.0",
        model_name="BGE m3 Uzbek Legal Matryoshka",
    ),
)


from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)


from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

# load train dataset again
train_dataset = load_dataset("json", data_files="train_dataset.json", split="train")

# define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="bge-m3-uz-legal-matryoshka",  # output directory and hugging face model ID
    num_train_epochs=4,  # number of epochs
    per_device_train_batch_size=32,  # train batch size
    gradient_accumulation_steps=16,  # for a global batch size of 512
    per_device_eval_batch_size=16,  # evaluation batch size
    warmup_ratio=0.1,  # warmup ratio
    learning_rate=2e-5,  # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
    tf32=True,  # use tf32 precision
    bf16=True,  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",  # evaluate after each epoch
    save_strategy="epoch",  # save after each epoch
    logging_steps=10,  # log every 10 steps
    save_total_limit=3,  # save only the last 3 models
    load_best_model_at_end=True,  # load the best model when training ends
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
)


from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,  # bg-base-en-v1
    args=args,  # training arguments
    train_dataset=train_dataset.select_columns(
        ["chunk", "question"]
    ),  # training dataset
    loss=train_loss,
    evaluator=evaluator,
)


trainer.train()

# save the best model
trainer.save_model()

# push model to hub
trainer.model.push_to_hub("bge-m3-uz-legal-matryoshka", token=key)
