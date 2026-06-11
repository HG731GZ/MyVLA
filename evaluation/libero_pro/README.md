# MyVLA LIBERO-PRO Evaluation

This directory evaluates the existing MyVLA LIBERO policy server on the
official LIBERO-PRO benchmark. It reuses `evaluation/libero/libero_client.py`
for rollout and changes only the benchmark registration, BDDL/init data, and
the language prompt read from each perturbed BDDL file.

No Python file contains a machine-specific LIBERO path. Paths can be supplied
with command-line arguments or environment variables. A temporary
`LIBERO_CONFIG_PATH` is generated for each run, so `~/.libero` and the
LIBERO-PRO checkout are not modified.

## 1. Install LIBERO-PRO

Run in the environment used for LIBERO simulation:

```bash
conda activate libero
cd /path/to/MyVLA
pip install -U huggingface_hub

bash evaluation/libero_pro/setup_libero_pro.sh
```

The default locations are:

```text
third_party/LIBERO-PRO
third_party/LIBERO-PRO-data
```

To keep them elsewhere, use either environment variables:

```bash
export LIBERO_PRO_ROOT=/path/to/LIBERO-PRO
export LIBERO_PRO_DATA_ROOT=/path/to/LIBERO-PRO-data
bash evaluation/libero_pro/setup_libero_pro.sh
```

or setup arguments:

```bash
bash evaluation/libero_pro/setup_libero_pro.sh \
  --libero_pro_root /path/to/LIBERO-PRO \
  --libero_pro_data_root /path/to/LIBERO-PRO-data
```

The official downloaded package includes `object`, `position`, `semantic`,
and `task` perturbations. In official suite names, `position` maps to `swap`
and `semantic` maps to `lan`.

## 2. Validate the Installation

This checks path resolution, benchmark registration, and all files needed by
one suite without connecting to the policy server:

```bash
python evaluation/libero_pro/libero_pro_client.py \
  --libero_pro_root /path/to/LIBERO-PRO \
  --libero_pro_data_root /path/to/LIBERO-PRO-data \
  --task_suite libero_object \
  --perturbation position \
  --validate_only
```

The two path arguments can be omitted when the environment variables above
are exported or the default `third_party` locations are used.

## 3. Start the MyVLA Policy Server

Run in the MyVLA training environment:

```bash
conda activate simvla
cd /path/to/MyVLA

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export SMOLVLM_MODEL_PATH=/path/to/SmolVLM-500M-Instruct

python -u evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint runs/libero_object_cross_self/ckpt-20000 \
  --norm_stats norm_stats/libero_object_norm.json \
  --host 127.0.0.1 \
  --port 8102
```

## 4. Evaluate One LIBERO-PRO Suite

Run in the LIBERO simulation environment:

```bash
conda activate libero
cd /path/to/MyVLA

export LIBERO_PRO_ROOT=/path/to/LIBERO-PRO
export LIBERO_PRO_DATA_ROOT=/path/to/LIBERO-PRO-data
export MUJOCO_GL=egl

bash evaluation/libero_pro/run_eval_libero_pro.sh \
  --host 127.0.0.1 \
  --port 8102 \
  --task_suite libero_object \
  --perturbation position \
  --num_trials 20 \
  --replan_steps 5 \
  --video_out ./eval_results_libero_pro
```

Supported base suites:

```text
libero_goal
libero_spatial
libero_10
libero_object
```

Supported perturbations:

```text
object
position
semantic
task
environment
```

For a quick smoke test, add:

```bash
--task_ids 0 --num_trials 1 --max_steps 20 --no_video
```

## 5. Generate Environment Perturbations

Environment perturbation files are not part of the official pre-generated
data package. Generate one base suite with:

```bash
bash evaluation/libero_pro/prepare_environment_suite.sh libero_object \
  --libero_pro_root "$LIBERO_PRO_ROOT" \
  --libero_pro_data_root "$LIBERO_PRO_DATA_ROOT"
```

Use `all` instead of `libero_object` to generate all four base suites. This
step is slower because it renders new initial states.

## 6. Evaluate All Suites and Perturbations

After environment files have been generated for all suites:

```bash
bash evaluation/libero_pro/run_eval_libero_pro.sh \
  --host 127.0.0.1 \
  --port 8102 \
  --all \
  --num_trials 50 \
  --no_video
```

Videos are written under `<video_out>/<actual_suite_name>/`, and the final
success rates are printed with names such as `libero_object_swap`.
