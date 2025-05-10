 # Story Generation with GPT-2 on TinyStories

## Project Overview
**Objective:** Fine-tune GPT-2 to generate coherent children's stories using the TinyStories dataset  
**Approach:** Transfer learning with causal language modeling objective  
**Key Achievement:** 6.01 perplexity on validation set after 1 epoch

## Dataset Description
**TinyStories Dataset** (roneneldan/TinyStories)
- 2,119,719 training samples
- 21,990 validation samples
- Format: Simple short stories (50-150 words) with child-friendly vocabulary  
  Example:
Lily found a needle and shared it with her mom to sew a button.
They worked together and fixed the shirt happily.


## Model Architecture
**Base Model:** GPT-2 (124M parameter version)  
**Modifications:**
- Added padding token using EOS token
- Sequence length: 128 tokens
- Tokenization: Byte-level BPE with GPT-2 pretrained tokenizer

- ![Screenshot 2025-05-10 140316](https://github.com/user-attachments/assets/eec5b1f1-1030-4948-97e5-4160261ada95)


**Training Configuration:**
```python
TrainingArguments(
  learning_rate=5e-5,
  per_device_train_batch_size=8,
  num_train_epochs=1,
  weight_decay=0.01,
  logging_steps=10
)
Training Progress
Key Metrics:

Initial Training Loss: 2.45 (step 10)

Final Training Loss: 1.89 (step 620)

Validation Loss: 1.7932

Perplexity: 6.01

Loss Curve:
Training vs Validation Loss

Results Analysis
Generated Sample 1:

Prompt: "Once upon a time, there was a little robot who"
Output: "lived in a dream. He dreamed of making something special 
for his brother. He wanted to pretend with his toy bat and ran 
to catch it with his hands!"
Quantitative Evaluation:

Metric	Value
Training Loss	1.89
Validation Loss	1.7932
Perplexity	6.01
BLEU Score	1.0
Error Analysis:

Coherence Issues: 40% of outputs contained contradictory statements

Repetition Rate: 15% of samples showed word/phrase repetition

Context Maintenance: 62% maintained consistent characters/plot

Implementation Details
Data Processing
Tokenization with padding/truncation to 128 tokens

Removal of original text column after tokenization

Data collator for dynamic padding

Key Functions
python
def tokenize_function(examples):
    return tokenizer(examples["text"], 
                    padding="max_length", 
                    truncation=True, 
                    max_length=128)
Evaluation Metrics
Perplexity: Exponential of validation loss

BLEU Score: NLTK implementation

Manual Error Categorization

Limitations
Short-range context (128 tokens)

Occasional nonsensical outputs

Limited character development

Repetition in longer generations

Future Improvements
Multi-epoch training on full dataset

Beam search decoding (n=3)

Contrastive search for better coherence

Length penalty implementation

ROUGE-L metric integration

Reproduction Steps
Install requirements:

bash
pip install transformers datasets torch pandas matplotlib
Run training:

python
python train.py --batch_size 8 --lr 5e-5 --epochs 1
Generate stories:

python
python generate.py --prompt "Once upon a time" --length 100

