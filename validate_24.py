import torch
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import VideoQADataLoader
from utils import todevice

import model.HCRN_3_layer as HCRN

from config import cfg, cfg_from_file
import resource
from IPython import embed
import pickle

SAVE_HIDDEN_STATE = False


def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')


    #embed()
    # # Visualize feature maps
    batch_activations = []
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            clip_level_crn_output = torch.cat(
                [frame_relation.unsqueeze(1) for frame_relation in output],
                dim=1)
            #output[0].shape[0] is batch_size
            clip_level_crn_output = clip_level_crn_output.view(output[0].shape[0], -1, 512).to(device)
            #embed()
            if(name not in activation):
                activation[name] = clip_level_crn_output.unsqueeze(1)
            else:
                activation[name] = torch.cat((activation[name],clip_level_crn_output.unsqueeze(1) ),dim = 1)
        return hook

    if(SAVE_HIDDEN_STATE):
        model.visual_input_unit.clip_level_question_cond.register_forward_hook(get_activation('clip_level_question_cond'))
        # model.conv1.register_forward_hook(get_activation('conv1'))


    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]

            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()

            if answers.dim() == 0:
                answers = answers.unsqueeze(0)
            batch_size = answers.size(0)
            logits = model(*batch_input)[0].to(device)
            if cfg.dataset.question_type in ['action', 'transition']:
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                agreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                preds = logits.detach().argmax(1)
                agreeings = (preds == answers)
            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count']:
                    preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition']:
                    answer_vocab = data.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        all_preds.append(predict.item())
                    else:
                        all_preds.append(answer_vocab[predict.item()])
                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id.cpu())
            if cfg.dataset.question_type == 'count':
                total_acc += batch_mse.float().sum().item()
                count += answers.size(0)
            else:
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)

            if(SAVE_HIDDEN_STATE):
                #Andrew storing the activations
                batch_activations.append(activation['clip_level_question_cond'])
                activation = {}
    
        acc = total_acc / count

    if(SAVE_HIDDEN_STATE):
        norms = []
        d_clips_norms = []
        d1_norms = []
        d2_norms = []
        d5_norms = []
        for ba in tqdm(batch_activations): #takes 4 minutes
            norms.append(torch.norm(ba, dim=(2,3)).flatten())
            d_clips = ba[1:,:,:,:] - ba[:-1,:,:,:]
            d_clips_norms.append(torch.norm(d_clips, dim=(2,3)).flatten())
            d1 = ba[:,1:,:,:] - ba[:,:-1,:,:]
            d1_norms.append(torch.norm(d1, dim=(2,3)).flatten())
            d2 = ba[:,2:,:,:] - ba[:,:-2,:,:]
            d2_norms.append(torch.norm(d2, dim=(2,3)).flatten())
            d5 = ba[:,5:,:,:] - ba[:,:-5,:,:]
            d5_norms.append(torch.norm(d5, dim=(2,3)).flatten())

        norms = torch.cat(norms, dim=0)
        d_clips_norms = torch.cat(d_clips_norms, dim=0)
        d1_norms = torch.cat(d1_norms, dim=0)
        d2_norms = torch.cat(d2_norms, dim=0)
        d5_norms = torch.cat(d5_norms, dim=0)
        hist_dict = {'norms': norms,'d_clips_norms':d_clips_norms, 'd1_norms': d1_norms, 'd2_norms': d2_norms, 'd5_norms': d5_norms}

        with open('hist_dict.pickle', 'wb') as handle:
            pickle.dump(hist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # batch_activations: list of list of tensors (num_batches, num_clips)
    # each tensor is of shape (batch_size, 512)
    if not write_preds:
        return acc
    else:
        return acc, all_preds, gts, v_ids, q_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='tgif_qa_action.yml', type=str)
    parser.add_argument('--mode', dest='mode', help='train, val, or test', default='val', type=str)

    parser.add_argument('--metric', dest='metric', help='question set', default='all', type=str)

    parser.add_argument('--sample', dest='sample', help='True if take sample, false if do whole set', default=False, type=bool)
    parser.add_argument('--model_name', dest='model_name', help='Name of the experiment', default='model', type=str)
    
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    
    # MODEL CHANGE
    ############## CHOOSE WHICH MODEL TO USE ######################
    # This one is the most recent model
    #ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model-current.pt')
    # This one is the highest validation
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', args.model_name + '.pt')
    #ckpt = '/usr0/home/alyubovs/agqa/hcrn-videoqa/results_25_35/expTGIF-QAFrameQA/ckpt/model.pt'

    print(ckpt)
    assert os.path.exists(ckpt)
    ##############################################################
    
    
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    print("dataset name", cfg.dataset.name)
    
    print("test write preds", cfg.test.write_preds)
    
    # This increases the resource limit
    cfg.test.write_preds = True
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


    if cfg.dataset.name == 'tgif-qa': # TODO: PATH [change all paths]
        cfg.dataset.vocab_json = '../storage/questions_w_captions/tgif-qa_frameqa_vocab-balanced.json'# % args.metric
        if args.mode == 'val':
            cfg.dataset.test_question_pt = '../storage/questions_w_captions/tgif-qa_frameqa_val_questions-%s.pt' % args.metric
        elif args.mode == 'test':
            cfg.dataset.test_question_pt = '../storage/questions_w_captions/tgif-qa_frameqa_test_questions-%s.pt' % args.metric
        elif args.mode == 'train':
            cfg.dataset.test_question_pt = '../storage/questions_w_captions/tgif-qa_frameqa_train_questions-%s.pt' % args.metric
        else: 
            print("invalid mode", args.mode)
        cfg.dataset.appearance_feat = '../drive_data/tgif-qa_frameqa_appearance_feat.h5'
        cfg.dataset.motion_feat = '../drive_data/tgif-qa_frameqa_motion_feat.h5'
        #cfg.dataset.appearance_feat = '../storage/gen_vid_features/appearance/tgif-qa_frameqa_appearance_feat_new.h5'
        #cfg.dataset.motion_feat = '../storage/gen_vid_features/appearance/tgif-qa_frameqa_motion_feat_new1.h5'
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.test_question_pt = '{}_test_questions.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False,
        'sample': args.sample,
    }
    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        print("IN WRITE PREDS")
        print(cfg)

        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device, cfg.test.write_preds)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, f"test_preds_{args.model_name}.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': dict[str(q_id)][0], 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 samples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], org_v_ids[idx]]
            
            instances = [
                    {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'csv_q_id': str(dict[str(q_id)][2]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 examples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()