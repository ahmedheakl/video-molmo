
CUDA_VISIBLE_DEVICES=1 python train_script.py 

CUDA_VISIBLE_DEVICES=3,7 torchrun --nproc-per-node=2  train_script.py 

scp -O -r connector/ shehan@10.127.30.114:/share/data/drive_2/shehan/videomolmo/


python train_memory.py --base_data_dir --output_dir --annotation_di