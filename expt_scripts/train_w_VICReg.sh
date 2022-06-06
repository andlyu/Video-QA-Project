#This script will train with different losses
#bash expt_scripts/train_w_VICReg.sh

# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss 0.0 \
#          --var_loss .025 \
#          --inv_loss .025\
#          --cov_loss .01\
#          --loss_dim 512 \
#          &

# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss 0.0 \
#          --var_loss .25 \
#          --inv_loss .25\
#          --cov_loss .01\
#          --loss_dim 512\
#         &
        

# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss 0.0 \
#          --var_loss 2.5 \
#          --inv_loss 2.5\
#          --cov_loss 0.1\
#          --loss_dim 512 


# CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss 0.0 \
#          --var_loss .25 \
#          --inv_loss .25\
#          --cov_loss .01\
#          --loss_dim 384 \
#         & 

# CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss .0 \
#          --var_loss 2.5 \
#          --inv_loss 2.5\
#          --cov_loss .1\
#          --loss_dim 384 \
#         & 

# CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss .0 \
#          --var_loss 25 \
#          --inv_loss 25\
#          --cov_loss 1\
#          --loss_dim 384 \
#         &

# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss .0 \
#          --var_loss .25 \
#          --inv_loss .25\
#          --cov_loss .01\
#          --loss_dim 192 \
#         & 

# CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss .0 \
#          --var_loss 2.5 \
#          --inv_loss 2.5\
#          --cov_loss .1\
#          --loss_dim 192 \
#          &


# CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/tgif_qa_frameqa_vicreg.yml \
#          --metric balanced \
#          --loss .0 \
#          --var_loss .025 \
#          --inv_loss .025\
#          --cov_loss .001\
#          --loss_dim 192 \
#TODOS;
#[DONE] add losses percentages
#[DONE] Add dimesion 
#[DONE] report to wandb

CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa.yml \
         --metric balanced \
         --loss 0 \
         --var_loss 0 \
         --inv_loss 0.0\
         --cov_loss 0\
         --loss_dim 512 \
         --dropout .1 \
         & 
sleep 5
echo"THis is 0"
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa.yml \
         --metric balanced \
         --loss .0 \
         --var_loss 0.0 \
         --inv_loss 0\
         --cov_loss 0\
         --loss_dim 512 \
         --dropout .3 \
         & 
sleep 5
echo "THis is 1"
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/tgif_qa_frameqa.yml \
         --metric balanced \
         --loss .0 \
         --var_loss 0 \
         --inv_loss 0\
         --cov_loss 0\
         --loss_dim 512 \
         --dropout .5 \
