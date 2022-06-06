#This script will train with different losses
python train.py --cfg configs/tgif_qa_restore.yml --metric balanced --loss .01

python train.py --cfg configs/tgif_qa_restore.yml --metric balanced --loss 0

python validate.py --cfg configs/tgif_qa_frameqa.yml --metric balanced --mode test --model_name model_halfe_0.01 

python validate.py --cfg configs/tgif_qa_frameqa.yml --metric balanced --mode test --model_name model_halfe_0

run subset_results in results_fifth_data/expTGIF-QAFrameQA (Modify path for the right file)