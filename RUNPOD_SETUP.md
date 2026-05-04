# RunPod Setup

Bootstrapping a fresh RunPod pod (Ubuntu, no sudo password needed — root by default) for verl experiments. Assumes you picked a PyTorch template; if you picked a barebones Ubuntu image, the CUDA bits matter more — call out which one when running this.

## 0. Sanity check the pod

```bash
nvidia-smi                  # GPUs visible?
nvcc --version || true      # CUDA toolkit present? (often not, that's fine)
python3 --version           # what's preinstalled
df -h /workspace            # network volume mounted here on RunPod
```

Do all heavy work under `/workspace` (the network volume) so it survives pod restarts. `/root` and `/` are ephemeral.

## 1. Apt basics

```bash
apt-get update
apt-get install -y \
    git git-lfs curl wget rsync unzip \
    build-essential ca-certificates \
    tmux htop nvtop \
    openssh-client vim less jq
git lfs install
```

## 2. Miniconda

Install into `/workspace/miniconda3` so it persists on the network volume.

```bash
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda3
rm miniconda.sh
/workspace/miniconda3/bin/conda init bash
exec bash   # reload shell
conda config --set auto_activate_base false
```

Make conda available in every new shell on this pod (RunPod web terminals don't always source `~/.bashrc`):

```bash
echo 'source /workspace/miniconda3/etc/profile.d/conda.sh' >> /root/.bashrc
```

## 3. Create the verl env

verl wants Python 3.10+; the Apptainer container was 3.12, so match that.

```bash
conda create -n verl python=3.12 -y
conda activate verl
```

Install PyTorch matching the pod's CUDA (check `nvidia-smi` top-right for the driver's CUDA version — install a torch built for ≤ that). For H100/A100 with recent drivers, cu124 wheels are safe:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision torchaudio
```

Then install verl + extras from the cloned repo (see step 5).

## 4. Claude Code

Needs Node 18+. Easiest path:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code
claude --version
```

First run will prompt for auth — paste your API key or do the browser flow.

Persist your Claude config on the volume so it survives pod restarts:

```bash
mkdir -p /workspace/.claude
ln -sfn /workspace/.claude /root/.claude
```

(Do this *before* the first `claude` login, or `mv /root/.claude/* /workspace/.claude/` first.)

## 5. Clone the repo

```bash
cd /workspace
git clone git@github.com:<you>/verl.git    # or https:// if no SSH key yet
cd verl
pip install -e .                           # editable install
# plus whatever extras the SFT/RL scripts need — check requirements*.txt
```

If you go SSH: `ssh-keygen -t ed25519 -C "runpod"` → add `~/.ssh/id_ed25519.pub` to GitHub.

## 6. Secrets / env vars

Put these in `/workspace/.env` and `source` it from `~/.bashrc`:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
export HF_HOME=/workspace/hf_cache       # don't fill / with model weights
export TRANSFORMERS_CACHE=/workspace/hf_cache
```

```bash
echo 'set -a; source /workspace/.env; set +a' >> /root/.bashrc
```

## 7. Data + base checkpoints

Whichever is faster for you:
- `rsync` from Delta over SSH (slow off-cluster, but one-shot)
- `huggingface-cli download <repo>` if base models live on the Hub
- `aws s3 sync` / `gcloud storage cp` if you have a bucket

Land everything under `/workspace/` — never `/root/` or `/tmp/`.

## 8. Quick smoke test

```bash
conda activate verl
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import verl; print(verl.__file__)"
```

## 9. Running experiments

No SLURM, no `chain_job.sh`. Just:

```bash
tmux new -s run
conda activate verl
cd /workspace/verl
bash experiments/refusal_clean/sft/refusal_clean_sft.sh \
  && bash experiments/refusal_clean/rl/refusal_clean_rl.sh
# Ctrl-b d to detach; tmux attach -t run to reattach
```

Things to fix in the experiment scripts before running:
- Hardcoded `/work/hdd/bbsg/twei2/rl/verl` → `/workspace/verl` (grep for it).
- `apptainer exec ... torch2501.sif` prefixes → drop entirely (you're already in the conda env).
- Output dirs pointing at Delta paths → `/workspace/outputs/...`.

A one-shot fix for most of this:

```bash
grep -rl "/work/hdd/bbsg/twei2/rl/verl" experiments/ | \
    xargs sed -i 's|/work/hdd/bbsg/twei2/rl/verl|/workspace/verl|g'
grep -rl "apptainer exec" experiments/ scripts/ | \
    xargs sed -i 's|apptainer exec /work/hdd/bbsg/twei2/rl/torch2501.sif ||g'
```

Eyeball the diff before committing.

## 10. Don't lose work

- Checkpoints → `/workspace/outputs/...` (network volume, survives pod stop).
- `git commit && git push` often — pod can die unexpectedly.
- If the pod is "Spot", assume it can be preempted; checkpoint frequently and make sure your RL script can resume.
