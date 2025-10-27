#!/bin/bash

# ===== Hugging Face token =====
export HF_TOKEN="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"

# ===== Ablation 설정 =====
BASE_CKPT_DIR="./ablation_study_1026"
POOL_TYPES=("sag" "mincut" "diffpool" "none")
ROUNDS=10            # 라운드 수
SEED_BASE=42         # 1라운드의 seed (라운드마다 +1)
POOL_REG_LAMBDA=1e-2

# ===== 기타 학습 하이퍼파라미터 =====
EPOCHS=100
BATCH_SIZE=32
DEVICE="cuda"
CUDA_DEV=1
MAIN_PY="main5.py"

for ((r=1; r<=ROUNDS; r++)); do
  echo "==================== Round ${r}/${ROUNDS} ===================="
  seed=$((SEED_BASE + r - 1))   # << 라운드별 고정 시드

  for pool_type in "${POOL_TYPES[@]}"; do
    RUN_ID="${pool_type}_s${seed}"
    CHECKPOINT_DIR="${BASE_CKPT_DIR}/${pool_type}/s${seed}"

    echo "==================================================="
    echo "Starting training | pool=${pool_type} | seed=${seed}"
    echo "Checkpoints -> ${CHECKPOINT_DIR}"
    echo "==================================================="

    CUDA_VISIBLE_DEVICES=${CUDA_DEV} python "${MAIN_PY}" \
      --hf_token "${HF_TOKEN}" \
      --protein_pool_type "${pool_type}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --run_id "${RUN_ID}" \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --device "${DEVICE}" \
      --seed ${seed} \
      --pool_reg_lambda ${POOL_REG_LAMBDA} \
      --amp false \
      --num_workers 4  \
      --prefetch_factor 2 \
      --persistent_workers true \
      --pin_memory true \
      --precomputed_root ./precomputed/Kd_s42

    echo "Finished | pool=${pool_type} | seed=${seed}"
    echo "---------------------------------------------------"
  done
done

echo "Ablation study completed."
