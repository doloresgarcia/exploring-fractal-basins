#!/bin/bash
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=flr
#SBATCH --output=%x_%j.out
set -e
export OMP_NUM_THREADS=8
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

ROOT=${ROOT_DIR:-/gpfs/scratch/ehpc1013/cern829880/kotaro_flr/flr}
cd "$ROOT"
PY=/gpfs/scratch/ehpc1013/cern829880/l40s/bin/python

STEPS=${STEPS:-300000}
OUT=${OUT:-runs/full}
CKPT_EVERY=${CKPT_EVERY:-2000}
MAXLOOPS=${MAXLOOPS:-20}
EVALLOOPS=${EVALLOOPS:-32}
BATCH=${BATCH:-1024}
EVAL_EVERY=${EVAL_EVERY:-2000}
LR=${LR:-1e-3}
CURR=${CURR:-80000}
COMPILE=${COMPILE:-1}
CFLAG=""; [ "$COMPILE" = "1" ] && CFLAG="--compile"
DEEPSUP=${DEEPSUP:-0}
[ "$DEEPSUP" = "1" ] && CFLAG="$CFLAG --deep_sup"

P=${P:-3}
NVAR=${NVAR:-8}
MVAR=${MVAR:-8}
echo "host $(hostname)  p=$P M=$MVAR N=$NVAR steps=$STEPS out=$OUT batch=$BATCH lr=$LR curr=$CURR compile=$COMPILE"
nvidia-smi -L
$PY train.py --p "$P" --M "$MVAR" --N "$NVAR" --out "$OUT" --steps "$STEPS" --ckpt_every "$CKPT_EVERY" --batch "$BATCH" \
    --lr "$LR" --curriculum_steps "$CURR" \
    --eval_every "$EVAL_EVERY" --log_every 200 --max_loops "$MAXLOOPS" --eval_loops "$EVALLOOPS" $CFLAG

if [ "${RUN_BASINS:-1}" = "1" ]; then
  echo "=== computing basins (hard instance) ==="
  $PY basins.py --run "$OUT" --out "$OUT/basins_hard" --res "${RES:-400}" --n_ckpts "${NCKPTS:-16}" \
      --instance_seed "${INST:-7}" --plane_seed "${PLANE:-3}" --instance_direct 0
  echo "=== computing basins (mixed instance) ==="
  $PY basins.py --run "$OUT" --out "$OUT/basins_mixed" --res "${RES:-400}" --n_ckpts "${NCKPTS:-16}" \
      --instance_seed "${INST:-7}" --plane_seed "${PLANE:-3}" --instance_direct "${INSTMIX:-4}"
fi
echo ALL_DONE
