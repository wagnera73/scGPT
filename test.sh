export NUMBA_CACHE_DIR=/scratch/wag9iz/numba
mkdir -p "$NUMBA_CACHE_DIR"
python scripts/main.py --dataset_name AML --model scGPT --task finetune --load_model save/scGPT_human --project scGPT-AML-celltype --label_column celltype
