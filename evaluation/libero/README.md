# Evaluation on LIBERO

## 1. Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## 2. Start Server

Run the policy server in the `simvla` environment. The checkpoint's
`num_views`, image size, language length, action horizon, and Cross-Self action
head configuration are loaded from `config.json`.

```bash
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint runs/libero_object_cross_self/ckpt-20000 \
    --norm_stats norm_stats/libero_object_norm.json \
    --port 8102
```

If the SmolVLM backbone is stored locally at a different path, specify it
without changing the checkpoint:

```bash
export SMOLVLM_MODEL_PATH=/path/to/SmolVLM-500M-Instruct
```

## 3. Run Evaluation

Set `LIBERO_ROOT` to your local LIBERO repository. The evaluation client will
generate a machine-local `config.yaml` and writable Numba cache automatically.
No repository-local LIBERO config needs to be edited.

```bash
conda activate libero
export LIBERO_ROOT=/path/to/LIBERO
```

Quick evaluation on selected tasks:

```bash
python evaluation/libero/libero_client.py \
    --libero_root /path/to/LIBERO \
    --host 127.0.0.1 \
    --port 8102 \
    --task_suite libero_object \
    --task_ids 0 \
    --num_trials 1 \
    --no_video
```

Full evaluation on all task suites:

```bash
bash evaluation/libero/run_eval_all.sh 8102 10 "eval_cross_self_20k" "0 1 2 3"
bash evaluation/libero/run_eval_all.sh 8102 50 "eval_cross_self_20k" "0 1 2 3"
```
