#this command will train the model with captions that are located in the directory identified in configs/tgif_qa_frameqa_captions.yml
#this is outdated. Training and validation split are now done automatically. 
CUDA_VISIBLE_DEVICES=1 python train.py --cfg configs/tgif_qa_frameqa_captions.yml --metric balanced \
            --captions True

CUDA_VISIBLE_DEVICES=1 python validate.py --cfg configs/tgif_qa_frameqa_captions.yml \
                 --metric balanced \
                 --model_name model \
                 --captions True \
                 --mode test
# saves to save_dir: 'results/results_w_captions'

#Eval with captiosn
CUDA_VISIBLE_DEVICES=1 python validate_24.py --cfg configs/tgif_qa_frameqa_captions.yml \
                 --metric balanced \
                 --model_name model \
                 --mode test