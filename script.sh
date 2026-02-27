## backbone training
torchrun --nproc_per_node=8 backbone_train.py
# one-class classification pipeline (backbone evaluation)
python oc_main.py
## binary classification training
torchrun --nproc_per_node=8 bc_train.py
## binary classification evaluation
CUDA_VISIBLE_DEVICES=0 python bc_eval.py 
CUDA_VISIBLE_DEVICES=1 python bc_eval.py --eval_noise jpg --eval_noise_param 95
CUDA_VISIBLE_DEVICES=2 python bc_eval.py --eval_noise blur --eval_noise_param 1.0
CUDA_VISIBLE_DEVICES=3 python bc_eval.py --eval_noise resize --eval_noise_param 0.5
CUDA_VISIBLE_DEVICES=4 python bc_eval.py --test_image_path /data/icml/wild_AGI/commercial/260225_new
