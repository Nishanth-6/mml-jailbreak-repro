MML Jailbreak Reproduction (CS 421 Research Study)

Overview
This repository contains a small scale reproduction of the ACL 2025 paper “Jailbreak Large Vision Language Models Through Multi Modal Linkage” (MML).
The goal is to test whether multimodal linkage attacks can bypass safety alignment in vision language models by hiding harmful text inside images.

The original paper uses large frontier models and the SafeBench benchmark.
This project reproduces the core attack pipeline using a smaller open model, Qwen2 VL 7B Instruct, and a controlled mini dataset of harmful prompts.

Main questions this code answers

Can a vision language model reconstruct harmful text that has been visually disguised inside an image

Which transformations are most effective at bypassing safety filters

Do we see the same ranking of attack strength as the original paper (Base64 strongest, rotation weakest)

Repository structure (high level)

You will typically see something like:

src/mml/
cli.py entry point for running attacks from the command line
attack.py logic for running a chosen attack on a batch of prompts
model_interface.py abstraction over specific models (Qwen2 VL etc)
data_utils.py prompt loading and image generation helpers (names may vary)

config.yaml configuration file for engine, model, attacks, and run settings
results/ folder where CSV summaries and logs are stored
runs_summary.csv main output file with per sample attack results

If filenames differ slightly, the roles above still describe how things are organized.

Environment and installation

You can run this code either in Google Colab or locally.

A. Running in Google Colab

Open a new Colab notebook and select a GPU runtime (for example T4 or A100).

Clone the repository into Colab:

git clone https://github.com/your-username/mml-jailbreak-repro.git

cd mml-jailbreak-repro

Install Python dependencies (this may already be in your notebook):

pip install -r requirements.txt

The main libraries include:

torch

transformers

accelerate

pillow

pandas

B. Running locally (for example on a machine with GPU)

Clone the repository:

git clone https://github.com/your-username/mml-jailbreak-repro.git

cd mml-jailbreak-repro

Create a virtual environment (optional but recommended):

python3 -m venv .venv
source .venv/bin/activate (Mac or Linux)
.venv\Scripts\activate (Windows PowerShell)

Install dependencies:

pip install -r requirements.txt

Make sure you have a recent version of PyTorch that can use your GPU if available.

Model interface and Qwen2 VL 7B

The code uses a simple model abstraction defined in src/mml/model_interface.py.

Key points about the Qwen2 VL integration:

The class QwenVLVL wraps the open source model Qwen/Qwen2-VL-7B-Instruct.

It uses AutoProcessor and AutoModelForVision2Seq from Hugging Face with trust_remote_code set to true.

Images are passed inside the messages structure, not as a separate images argument, to match the current Qwen2 VL API.

The generate method takes a PIL image and a text prompt, builds a chat style messages list, runs the model, and returns the decoded string.

If you need to change the model, you can edit:

model_name in config.yaml, or

the defaults inside QwenVLVL in model_interface.py.

Configuration (config.yaml)

The config file controls what attacks run, which model to use, and how many samples to process.

Typical fields you will see:

engine
Which model wrapper to use. In this reproduction it is usually set to qwen_vl.

model_name
Hugging Face model identifier, for example Qwen/Qwen2-VL-7B-Instruct.

device
cuda for GPU, cpu as a fallback.

attacks
List of attack names to run, for example [mirror, rotate, word_replace, base64].

limit
Maximum number of samples to process in a run. Useful for quick tests.

output_dir
Directory where results CSV files are written.

If you are unsure, you can leave the default config.yaml unchanged.
To run only a subset of attacks, edit the attacks list.

Dataset and prompts

The original paper uses SafeBench, a large harmful prompt benchmark.
This reproduction uses a smaller custom dataset that follows the same structure.

The dataset contains three harmful instruction sentences representing different categories:

hacking and cyber intrusion

weapon construction

evading law enforcement

For each harmful instruction, the code generates multiple attack variants:

mirror text inside an image

rotated text inside an image

word replaced text inside an image

Base64 encoded text inside an image

This produces 3 prompts x 4 attack types = 12 multimodal attack samples for each full run.

Prompts may be defined either directly in Python lists or loaded from a small CSV or JSON file, depending on how the repo is set up.
If you want to change the prompts, look for a data or prompts utility in src/mml and edit the list of base strings there.

Attack methods in this code

Mirror attack

Harmful text is rendered into an image and then flipped horizontally.

This hides the text from simple text based filters but can still be read by OCR inside the model.

Rotation attack

Harmful text is rendered then rotated by a small angle (for example 10 degrees).

This makes OCR more difficult and usually leads to only partial recovery.

Word replacement attack

Words in the harmful sentence are replaced or reordered to make the surface text look less obviously harmful while preserving meaning.

The perturbation is applied before rendering the text into an image.

Base64 attack

The entire harmful sentence is encoded with Base64.

The encoded string is drawn into the image as plain text.

The safety filter does not recognize it as harmful, but the model often decodes it.

How to run attacks from the command line

All attacks are launched through the CLI module.

From the repository root:

Set the Python path so that src is importable:

export PYTHONPATH=src (Mac or Linux)
set PYTHONPATH=src (Windows cmd)
$env:PYTHONPATH="src" (Windows PowerShell)

In Colab you can do:

PYTHONPATH=src python -m mml.cli --config config.yaml --limit 5

Run a quick test with a small limit:

PYTHONPATH=src python -m mml.cli --config config.yaml --limit 5

This will:

load the Qwen2 VL 7B model

generate images for each attack

query the model

compute similarity and success metrics

write a CSV summary under results/

Outputs and how to interpret them

After a successful run you should see:

A results directory if it does not already exist

One or more CSV files, for example results/runs_summary.csv

Typical columns in the CSV include:

attack which attack type (mirror, rotate, word_replace, base64)

sample_id index of the sample

prompt_text the base harmful instruction used

model_output what the model actually replied

refusal_flag whether the output looked like a refusal (for example “I cannot help with that”)

keyword_overlap fraction or count of original keywords that reappear in the output

success_flag binary indicator of whether this is counted as a successful attack

Attack success rate (ASR) is computed as the mean of success_flag for all samples of a given attack.

For example, if Base64 has success_flag = 1 for 6 out of 10 samples, then ASR for Base64 is 0.6 or 60 percent.

Reproducing the paper’s main claims

The original paper reports:

Base64 attacks have the highest ASR

Semantic perturbation attacks show strong leakage of harmful meaning

Mirror has mixed performance

Rotation is the weakest attack type

This reproduction confirms the same qualitative ranking using a smaller open model:

Base64 is strongest

Word replacement is second

Mirror is medium

Rotation is weakest

Absolute ASR values differ from the paper because we use Qwen2 VL 7B instead of larger frontier models, and because the dataset is smaller.
The goal is to match the pattern, not the exact numbers.

Common issues and fixes

Qwen2VLProcessor multiple values for argument 'images'
Cause: passing both messages and images= to the processor.
Fix: only pass messages that already contain the image content. The patched QwenVLVL.generate in model_interface.py does this correctly.

GPU memory errors in Colab
Reduce the limit in the CLI command, or run fewer attacks at once by editing the attacks list in config.yaml.

Slow runs
Use a smaller limit during debugging. Once everything works, run a full experiment with a higher limit.
