export PYTHONPATH="/home/muzammal/heakl/video-molmo/archive:$PYTHONPATH"
# export MOLMO_DATA_DIR=/share/data/drive_1/heakl/molmo
# export HF_HOME=/share/data/drive_1/heakl/molmo/huggingface

python scripts/train_prev.py --batch_size 1 --bfloat16