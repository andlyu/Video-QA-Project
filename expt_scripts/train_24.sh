#This file contains the instructions to run a model that use video representations with 24 clips per video
#this is an average of 1 second per clip
#I'm pretty sure this is correct
python train.py --cfg configs/tgif_qa_frameqa_24.yml --metric balanced

#To speed up training, run it with by redirecting HCRN to HCRN_3_layer model
#replace 'import model.HCRN' with 'import model.HCRN_3_layer'




