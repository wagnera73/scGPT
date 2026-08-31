conda activate rtd
module load cuda/12.2
sh quick_submit.sh

python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi   # confirm it reports an A100 and the driver's CUDA version

nvcc --version     # if missing, `module avail cuda` and `module load cuda/11.6` (name varies by cluster)
export CUDA_HOME=


export MAX_JOBS=4   # tune down further (2) if you still see the build get killed
pip install "flash-attn==1.0.4" --no-build-isolation

python -c "from flash_attn.flash_attention import FlashMHA; print('ok')"


python scripts/main.py \
  --dataset_name AML \
  --model scGPT \
  --task finetune \
  --load_model save/scGPT_human \
  --project scGPT-AML-celltype-smoketest



bsub -q gpu-a100 -W 15:00 -n 4 -M 200000 -gpu "num=2" -R "span[hosts=1]" -o "./logs/%J.out" -e "./logs/%J.err" sh test.sh



# Fast API test
pip install -r webapp/requirements.txt
python -m uvicorn webapp.app:app --reload --port 8000
