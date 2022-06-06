#sh expt_scripts/hpsearch.sh
#wandb sweep sweep.yaml
#Then run the highlighted command that is printed in the terminal
for I in 1 2 3
do
  CUDA_VISIBLE_DEVICES=0 wandb agent --count 1  alyudre/hcrn-videoqa/owy1clim &
done & 

for I in 1 2 3
do
  CUDA_VISIBLE_DEVICES=1 wandb agent --count 1  alyudre/hcrn-videoqa/owy1clim &
done & 

for I in 1 2 3
do
  CUDA_VISIBLE_DEVICES=2 wandb agent --count 1  alyudre/hcrn-videoqa/owy1clim &
done
