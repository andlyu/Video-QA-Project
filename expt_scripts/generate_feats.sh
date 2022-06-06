#This script will generate video features with captions
#Make sure the path in preprocess_questions is correct
#next either prep
python preprocess/preprocess_questions.py --dataset tgif-qa --question_type frameqa --glove_pt data/glove/glove.840.300d.pkl --mode train --metric balanced