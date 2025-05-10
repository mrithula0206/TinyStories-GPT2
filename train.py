from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Load the TinyStories dataset
ds = load_dataset("roneneldan/TinyStories")
print(ds)

# Check a sample
print("\nSample from training set:")
print(ds["train"][0])

# Load tokenizer 
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ensure tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
print("\nTokenized example:")
print(tokenized_ds["train"][0])

# Model Definition and Training 
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load GPT-2 model for causal language modeling
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Resize embedding if pad token was added
model.resize_token_embeddings(len(tokenizer))

# Define training arguments (removed unsupported keys for older transformers)
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Define data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"].select(range(5000)),  # subset for speed
    eval_dataset=tokenized_ds["validation"].select(range(1000)),
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()


# Evaluation and Inference
import math
from transformers import pipeline

# Evaluate: Compute loss on validation set and convert to perplexity
eval_results = trainer.evaluate()
eval_loss = eval_results["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"\nEvaluation loss: {eval_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")


# Updated Plotting Code: Training vs Validation Loss
import pandas as pd
import matplotlib.pyplot as plt

log_history = trainer.state.log_history if hasattr(trainer.state, 'log_history') else []

# Convert log history to DataFrame
log_df = pd.DataFrame(log_history)

# Print what's inside
print("\nLogged values:\n", log_df.columns)
print(log_df.head())

# Filter rows with either 'loss' or 'eval_loss'
filtered_df = log_df[(log_df.get("loss").notnull()) | (log_df.get("eval_loss").notnull())]

# Plotting
plt.figure(figsize=(10, 5))

if "loss" in filtered_df:
    plt.plot(filtered_df["step"], filtered_df["loss"], label="Training Loss", marker="o")

if "eval_loss" in filtered_df:
    plt.plot(filtered_df["step"], filtered_df["eval_loss"], label="Validation Loss", marker="o")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Text Generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate a short story
prompt = "Once upon a time, there was a little robot who"
outputs = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True)

print("\nGenerated story:\n")
print(outputs[0]["generated_text"])


# Save model
model.save_pretrained("./tiny-gpt2")
tokenizer.save_pretrained("./tiny-gpt2")

# Reload 
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./tiny-gpt2")
tokenizer = AutoTokenizer.from_pretrained("./tiny-gpt2")


from transformers import pipeline

# Reload your trained model and tokenizer
model_path = "./tiny-gpt2"  # or use model directly if not saved
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test prompts 
prompts = [
    "The boy who lived under the sea",
    "In a world where robots ruled",
    "Once upon a time, a cat learned to talk",
    "The wizard cast a powerful spell",
    "A lonely planet drifted through space"
]

# Generate stories
print("ERROR ANALYSIS: Generated Outputs\n")
bad_outputs = []

for prompt in prompts:
    output = generator(prompt, max_length=50, do_sample=True, num_return_sequences=1)[0]["generated_text"]
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {output}\n")

    # Mark possible issues
    if len(output.split()) < 20:
        print("Issue: SHORT COMPLETION\n")
        bad_outputs.append((prompt, "Short"))
    elif output.lower().count("the") > 8:
        print("Issue: REPETITIVENESS (e.g., too many 'the')\n")
        bad_outputs.append((prompt, "Repetitive"))
    elif "asdf" in output or any(w not in tokenizer.get_vocab() for w in output.split()):
        print("Issue: NONSENSE DETECTED\n")
        bad_outputs.append((prompt, "Nonsense"))


from nltk.translate.bleu_score import sentence_bleu

reference = ["the little robot became a hero in the village.".split()]
candidate = "the little robot became a hero in the village.".split()
print("BLEU score:", sentence_bleu(reference, candidate))


pip install gradio


import gradio as gr

def generate_story(prompt):
    output = generator(prompt, max_length=100, do_sample=True)
    return output[0]["generated_text"]

gr.Interface(fn=generate_story, inputs="text", outputs="text", title="TinyStories Generator").launch()
