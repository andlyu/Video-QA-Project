
# #change save dir to where you want to save the files within tgif_qa_frameqa.yml
# CUDA_VISIBLE_DEVICES=1 python validate.py --cfg configs/tgif_qa_frameqa.yml \
#                  --metric balanced \
#                  --model_name fin_model_a \
#                  --mode test &
# #change save dir to where you want to save the files within tgif_qa_frameqa.yml
# CUDA_VISIBLE_DEVICES=1 python validate.py --cfg configs/tgif_qa_frameqa.yml \
#                  --metric balanced \
#                  --model_name fin_model_b \
#                  --mode test &
# #change save dir to where you want to save the files within tgif_qa_frameqa.yml
# CUDA_VISIBLE_DEVICES=1 python validate.py --cfg configs/tgif_qa_frameqa.yml \
#                  --metric balanced \
#                  --model_name fin_model_c \
#                  --mode test


# python results/process_preds.py --input_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_preds_fin_model_a.json' \
#                         --output_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_results_fin_model_a.json' \
#                         &

# python results/process_preds.py --input_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_preds_fin_model_b.json' \
#                         --output_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_results_fin_model_b.json' \
#                         &

# python results/process_preds.py --input_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_preds_fin_model_c.json' \
#                         --output_file 'results/results_results_w_loss/expTGIF-QAFrameQA/preds/test_results_fin_model_c.json' \
                        
                        


#Save the results to csv within the folder
# python results/process_preds.py --input_file 'results_24_clips/expTGIF-QAFrameQA/preds/test_preds_model0.json' \
#                         --output_file 'results_24_clips/expTGIF-QAFrameQA/preds/test_results_model0.json'

# python results/process_preds.py --input_file 'results_fifth_data/expTGIF-QAFrameQA/preds/test_preds_model.json' \
#                         --output_file 'results_fifth_data/expTGIF-QAFrameQA/preds/test_results_model.json'


