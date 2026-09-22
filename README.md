# Local LoRA Training Example

This project is a small example of local LoRA fine-tuning for a causal
language model using Hugging Face Transformers, PEFT, Datasets, and PyTorch.

It includes:

- A small synthetic example dataset
- A LoRA training script
- A local chat script for the trained adapter
- An optional Ollama `Modelfile`
- Git-friendly ignore rules

## Project layout

```text
llmtraining/
|-- dataset.csv
|-- train.py
|-- lorachat.py
|-- Modelfile
|-- requirements.txt
|-- .gitignore
|-- README.md
`-- LICENSE
```

The downloaded base model and all training output are intentionally excluded
from Git.

## Example dataset format

`dataset.csv` is a two-column CSV with no header.

Each row contains:

```text
input,output
```

For example:

```text
"What is photosynthesis?","Photosynthesis is the process plants use to turn light energy into chemical energy."
```

The included dataset is synthetic example material intended only to
demonstrate the required format. Replace or expand it with data appropriate
for your own training goal.

## Ubuntu 24.04 setup

Install Python and Git:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install PyTorch for your hardware.

A basic installation is:

```bash
pip install torch
```

If you have an NVIDIA GPU, use the PyTorch installation command appropriate
for your CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Download the base model

The scripts expect the base model in:

```text
model/
```

One compatible example is Meta Llama 3.2 3B.

Authenticate with Hugging Face on your machine:

```bash
hf auth login
```

Then download the model:

```bash
hf download meta-llama/Llama-3.2-3B --local-dir model
```

You may need to accept the model provider's license terms on Hugging Face
before downloading it.

Do not place access tokens in source files or commit them to Git.

## Train

Run:

```bash
python train.py
```

The script reads:

```text
dataset.csv
```

and writes the LoRA adapter to:

```text
local_models/local-lora/
```

It also writes a merged model to:

```text
local_models/local-merged/
```

Checkpoints are written to:

```text
checkpoints/local-finetune/
```

On a CUDA GPU, the training script uses 4-bit quantization to reduce memory
use. If CUDA is not available, it falls back to an unquantized CPU load,
which is much slower and requires substantially more memory.

## Chat with the LoRA adapter

After training:

```bash
python lorachat.py
```

The chat script uses the same paths produced by `train.py`:

```text
model/
local_models/local-lora/
```

Type:

```text
exit
```

or:

```text
quit
```

to leave the chat.

## Optional Ollama use

The included `Modelfile` references:

```text
local_models/local-lora
```

On a system with Ollama installed, you can experiment with:

```bash
ollama create my-local-model -f Modelfile
```

Adapter compatibility depends on the base model and Ollama version.

## Customizing the dataset

Keep `dataset.csv` as a two-column CSV without a header:

```text
"question or instruction","desired response"
```

Use properly quoted CSV fields when text contains commas.

For a real fine-tune, use a larger, carefully reviewed dataset that matches
the behavior you want the model to learn.

## Git setup

Initialize the repository:

```bash
git init -b main
git add .
git status
git commit -m "Initial release"
```

Create a private repository:

```bash
gh repo create llmtraining --private --source=. --remote=origin --push
```

Or create a public repository:

```bash
gh repo create llmtraining --public --source=. --remote=origin --push
```

The `.gitignore` excludes downloaded model weights, checkpoints, local
training output, virtual environments, and caches.

## License

An MIT license template is included. Replace `YOUR NAME` in `LICENSE` before
publishing if you want to use the MIT license.
