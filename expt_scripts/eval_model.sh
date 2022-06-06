
#change save dir to where you want to save the files within tgif_qa_frameqa.yml
python validate.py --cfg configs/tgif_qa_frameqa.yml \
                 --metric balanced \
                 --model_name model_half_ep_0.0_0.0_0.0_0.0 \
                 --mode test

#Evaluate baseline
CUDA_VISIBLE_DEVICES=1 python validate.py --cfg configs/tgif_qa_frameqa_baseline.yml \
                 --metric balanced \
                 --model_name model \
                 --mode test

#To evaluate 24 features
#Set the model to the correct path
#change save dir to where you want to save the files within tgif_qa_frameqa.yml
python validate.py --cfg configs/tgif_qa_frameqa_24.yml \
                 --metric balanced \
                 --model_name model0
                 --mode test

#Save the results to csv within the folder
python results/process_preds.py --input_file 'results_24_clips/expTGIF-QAFrameQA/preds/test_preds_model0.json' \
                        --output_file 'results_24_clips/expTGIF-QAFrameQA/preds/test_results_model0.json'

python results/process_preds.py --input_file 'results_fifth_data/expTGIF-QAFrameQA/preds/test_preds_model.json' \
                        --output_file 'results_fifth_data/expTGIF-QAFrameQA/preds/test_results_model.json'

    


