import os, sys

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import argparse
import time
import logging
from termcolor import colored
import wandb
import time


from IPython import embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from DataLoader import VideoQADataLoader
from utils import todevice
from validate import validate

import model.HCRN as HCRN
from config import cfg, cfg_from_file

torch.autograd.set_detect_anomaly(True)

def train(cfg):
    start_time = time.time()
    run_name =  f'{round(start_time%10,4)}dif_{round(cfg.train.loss_ratio,2)}'+ \
                f'_var_{round(cfg.train.var_loss,2)}_inv_{round(cfg.train.inv_loss,2)}' + \
                f'_cov_{round(cfg.train.cov_loss,2)}'
    print(run_name)
    print('num wrkers', cfg.num_workers)
    logging.info("Create train_loader and val_loader.........")
    train_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.train_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'train_num': cfg.train.train_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': True,
        'sample': False,
    }
    train_loader = VideoQADataLoader(**train_loader_kwargs)
    logging.info("number of train instances: {}".format(len(train_loader.dataset)))
    if cfg.val.flag:
        val_loader_kwargs = {
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'sample': False,
        }
        val_loader_kwargs_sample = {
            'question_type': cfg.dataset.question_type,
            'question_pt': cfg.dataset.val_question_pt,
            'vocab_json': cfg.dataset.vocab_json,
            'appearance_feat': cfg.dataset.appearance_feat,
            'motion_feat': cfg.dataset.motion_feat,
            'val_num': cfg.val.val_num,
            'batch_size': cfg.train.batch_size,
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'sample': False,
        }                                                                                                                                           
        val_loader = VideoQADataLoader(**val_loader_kwargs)
        logging.info("number of val instances: {}".format(len(val_loader.dataset)))

    logging.info("Create model.........")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {
        'vision_dim': cfg.train.vision_dim,
        'module_dim': cfg.train.module_dim,
        'word_dim': cfg.train.word_dim,
        'k_max_frame_level': cfg.train.k_max_frame_level,
        'k_max_clip_level': cfg.train.k_max_clip_level,
        'spl_resolution': cfg.train.spl_resolution,
        'vocab': train_loader.vocab,
        'question_type': cfg.dataset.question_type,
        'loss_dim': cfg.train.loss_dim,
        'dropout_rate': cfg.train.dropout
    }
    model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('num of params: {}'.format(pytorch_total_params))
    logging.info(model)
    wandb.watch(model)

    if cfg.train.glove:
        logging.info('load glove vectors')
        train_loader.glove_matrix = torch.FloatTensor(train_loader.glove_matrix).to(device)
        with torch.no_grad():
            model.linguistic_input_unit.encoder_embed.weight.set_(train_loader.glove_matrix)
    if torch.cuda.device_count() > 1 and cfg.multi_gpus:
        model = model.cuda()
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=None) 
        print('creating many diveces')

    optimizer = optim.Adam(model.parameters(), cfg.train.lr, weight_decay=1e-5)

    start_epoch = 0
    if cfg.dataset.question_type == 'count':
        best_val = 100.0
    else:
        best_val = 0
    if cfg.train.restore:
        print("Restore checkpoint and optimizer...")
        #ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model0.001.pt')
        ckpt = '/usr0/home/alyubovs/agqa/hcrn-videoqa/results_25_35/expTGIF-QAFrameQA/ckpt/model.pt'
        #ckpt = '/usr0/home/alyubovs/agqa/hcrn-videoqa/results_fifth_data/expTGIF-QAFrameQA/ckpt/model.pt'
        ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        start_epoch = ckpt['epoch'] + 1 
        #ckpt['state_dict'].pop('linguistic_input_unit.encoder_embed.weight')
        #model = torch.nn.DataParallel(model)
        model.load_state_dict(ckpt['state_dict']) #made strict false
        optimizer.load_state_dict(ckpt['optimizer'])
    if cfg.dataset.question_type in ['frameqa', 'none']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.dataset.question_type == 'count':
        criterion = nn.MSELoss().to(device)

    logging.info("Start training........") 
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if(epoch == 11):
            return
        logging.info('>>>>>> epoch {epoch} <<<<<<'.format(epoch=colored("{}".format(epoch), "green", attrs=["bold"])))
        model.train()
        total_acc, count = 0, 0
        batch_mse_sum = 0.0
        total_loss, avg_loss = 0.0, 0.0
        avg_loss = 0
        total_d1_mean , avg_d1_mean = 0.0, 0.0
        total_losses, avg_losses = {'var_loss':0,'inv_loss':0,'cov_loss':0}, {'var_loss':0,'inv_loss':0,'cov_loss':0}
        train_accuracy = 0

        #save model
        ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
        save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model.pt'))

        for i, batch in enumerate(iter(train_loader)):
            progress = epoch + i / len(train_loader)
            _, _, answers, *batch_input = [todevice(x, device) for x in batch]
            answers = answers.cuda().squeeze()
            batch_size = answers.size(0)
            optimizer.zero_grad()
            logits, d1_losses = model(*batch_input)

            if cfg.dataset.question_type in ['action', 'transition']:
                batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                                   [1, 5])) * 5  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
                answers_agg = tile(answers, 0, 5)
                loss = torch.max(torch.tensor(0.0).cuda(),
                                 1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
                loss = loss.sum()
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                aggreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                loss = criterion(logits, answers.float())
                loss.backward()
                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                d1_mean, d1_var, d1_inv, d1_cov = d1_losses
                loss =  criterion(logits, answers) + \
                            d1_mean * cfg.train.loss_ratio + \
                            d1_var * cfg.train.var_loss + \
                            d1_inv * cfg.train.inv_loss + \
                            d1_cov * cfg.train.cov_loss
                            
                if(i%1000 == 0):
                    print()
                    print('total_loss , d1_loss (lambda = ', cfg.train.loss_ratio ,') ' , loss, d1_mean)
                loss.backward()

                total_loss += loss.detach()
                avg_loss = total_loss / (i + 1) 
                total_d1_mean += d1_mean.detach()
                avg_d1_mean = total_d1_mean / (i + 1)
                for loss_name, d1_loss in zip(['var_loss', 'inv_loss', 'cov_loss'], [d1_var, d1_inv, d1_cov]):
                    total_losses[loss_name] += d1_loss.detach()
                    avg_losses[loss_name] = total_losses[loss_name] / (i + 1)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=12)
                optimizer.step()
                aggreeings = batch_accuracy(logits, answers)

            if cfg.dataset.question_type == 'count':
                batch_avg_mse = batch_mse.sum().item() / answers.size(0)
                batch_mse_sum += batch_mse.sum().item()
                count += answers.size(0)
                avg_mse = batch_mse_sum / count
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_mse = {train_mse}    avg_mse = {avg_mse}    exp: {exp_name}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        train_mse=colored("{:.4f}".format(batch_avg_mse), "blue",
                                          attrs=['bold']),
                        avg_mse=colored("{:.4f}".format(avg_mse), "red", attrs=['bold']),
                        exp_name=cfg.exp_name))
                sys.stdout.flush()
            else:
                total_acc += aggreeings.sum().item()
                count += answers.size(0)
                train_accuracy = total_acc / count
                # wandb.log({'avg_acc': train_accuracy, \
                #             'avg_loss': avg_loss, \
                #             'avg_d1_loss': avg_d1_mean, \
                #             'avg_var_loss': avg_losses['var_loss'], \
                #             'avg_inv_loss': avg_losses['inv_loss'], \
                #             'avg_cov_loss': avg_losses['cov_loss']})
                sys.stdout.write(
                    "\rProgress = {progress}   ce_loss = {ce_loss}   avg_loss = {avg_loss}    train_acc = {train_acc}    avg_acc = {avg_acc}    exp: {exp_name}".format(
                        progress=colored("{:.3f}".format(progress), "green", attrs=['bold']),
                        ce_loss=colored("{:.4f}".format(loss.item()), "blue", attrs=['bold']),
                        avg_loss=colored("{:.4f}".format(avg_loss), "red", attrs=['bold']),
                        train_acc=colored("{:.4f}".format(aggreeings.float().mean().cpu().numpy()), "blue",
                                          attrs=['bold']),
                        avg_acc=colored("{:.4f}".format(train_accuracy), "red", attrs=['bold']),
                        exp_name=cfg.exp_name))
                sys.stdout.flush()
            if( i==325 or (i % 3250 == 0 and i != 0)): #3250 == 0 and i != 0: #need to wait an hour and a half before running each evaluation
                                        #100 bathces / 2.5 min : 400b/10m: 800b/h : 3k/4h
                output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    assert os.path.isdir(output_dir)
                valid_acc = validate(cfg, model, VideoQADataLoader(**val_loader_kwargs_sample), device, write_preds=False) 

                wandb.log({'avg_acc': train_accuracy, \
                            'avg_loss': avg_loss, \
                            'avg_d1_loss': avg_d1_mean, \
                            'avg_var_loss': avg_losses['var_loss'], \
                            'avg_inv_loss': avg_losses['inv_loss'], \
                            'avg_cov_loss': avg_losses['cov_loss'], \
                            'valid_acc': valid_acc})
                #wandb.log({'valid_acc': valid_acc})
                logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
                sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                    valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
                sys.stdout.flush()
                model.train()
                print('saving to ')


        sys.stdout.write("\n")
        if cfg.dataset.question_type == 'count':
            if (epoch + 1) % 5 == 0:
                optimizer = step_decay(cfg, optimizer)
        else:
            if (epoch + 1) % 10 == 0:
                optimizer = step_decay(cfg, optimizer)
        sys.stdout.flush()
        logging.info("Epoch = %s   avg_loss = %.3f    avg_acc = %.3f" % (epoch, avg_loss, train_accuracy))

        if cfg.val.flag:
            output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            else:
                assert os.path.isdir(output_dir)
            valid_acc = validate(cfg, model, val_loader, device, write_preds=False)
            if (valid_acc > best_val and cfg.dataset.question_type != 'count') or (valid_acc < best_val and cfg.dataset.question_type == 'count'):
                best_val = valid_acc
                # Save best model
                ckpt_dir = os.path.join(cfg.dataset.save_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                else:
                    assert os.path.isdir(ckpt_dir)
                save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model'+str(cfg.train.loss_ratio)+ f'{run_name}.pt'))
                sys.stdout.write('\n >>>>>> save highest validation to %s <<<<<< \n' % (ckpt_dir))
                sys.stdout.flush()

            save_checkpoint(epoch, model, optimizer, model_kwargs_tosave, os.path.join(ckpt_dir, 'model-current'+str(cfg.train.loss_ratio)+ f'{run_name}.pt'))
            sys.stdout.write('\n >>>>>> save current model to %s <<<<<< \n' % (ckpt_dir))
            sys.stdout.flush()

            logging.info('~~~~~~ Valid Accuracy: %.4f ~~~~~~~' % valid_acc)
            sys.stdout.write('~~~~~~ Valid Accuracy: {valid_acc} ~~~~~~~\n'.format(
                valid_acc=colored("{:.4f}".format(valid_acc), "red", attrs=['bold'])))
            sys.stdout.flush()

# Credit https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)


def step_decay(cfg, optimizer):
    # compute the new learning rate based on decay rate
    cfg.train.lr *= 0.5
    logging.info("Reduced learning rate to {}".format(cfg.train.lr))
    sys.stdout.flush()
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.train.lr

    return optimizer


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    predicted = predicted.detach().argmax(1)
    agreeing = (predicted == true)
    return agreeing


def save_checkpoint(epoch, model, optimizer, model_kwargs, filename):
    if(isinstance(model, nn.DataParallel)):
        model = model.module

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
    }
    time.sleep(10)
    torch.save(state, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/tgif_qa_frameqa.yml', type=str)
    parser.add_argument('--metric', dest='metric', help='type of training', default='balanced', type=str)
    parser.add_argument('--loss', dest='loss', help='weight added to loss', default='0', type=float)
    parser.add_argument('--var_loss', dest='var_loss', help='weight added to variance loss in VICReg', default='0', type=float)
    parser.add_argument('--inv_loss', dest='inv_loss', help='weight added to invariance loss in VICReg', default='0', type=float)
    parser.add_argument('--cov_loss', dest='cov_loss', help='weight added to covariance loss in VICReg', default='0', type=float)
    parser.add_argument('--loss_dim', dest='loss_dim', help='dimension of the middle layer to which to apply the loss to', default=512, type=int)
    parser.add_argument('--dropout', dest='dropout', help='the dropout value that should be applied to each CRN layer', default=0, type=float)
    parser.add_argument('--captions', dest='captions', help='boolean to see whether to load the captions', default=False, type=bool)
    parser.add_argument('--wandb', dest='wandb', help='boolean to see whether to run with weights & Biasses', default=True, type=bool)
    args = parser.parse_args()


    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    
    cfg.train.loss_ratio = args.loss
    cfg.train.var_loss = args.var_loss
    cfg.train.inv_loss = args.inv_loss
    cfg.train.cov_loss = args.cov_loss
    cfg.train.loss_dim = args.loss_dim
    cfg.train.dropout = args.dropout

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)
    # check if k_max is set correctly
    assert cfg.train.k_max_frame_level <= 16
    assert cfg.train.k_max_clip_level <= 24


    if not cfg.multi_gpus:
        print("In multiple gpus - id i s", cfg.gpu_id)
        torch.cuda.set_device(0)
    # make logging.info display into both shell and file
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    if not os.path.exists(cfg.dataset.save_dir):
        os.makedirs(cfg.dataset.save_dir)
    else:
        assert os.path.isdir(cfg.dataset.save_dir)
    log_file = os.path.join(cfg.dataset.save_dir, "log")
    if not cfg.train.restore and not os.path.exists(log_file):
        os.mkdir(log_file)
    else:
        pass
        #assert os.path.isdir(log_file)

    fileHandler = logging.FileHandler(os.path.join(log_file, 'stdout_'+str(cfg.train.loss_ratio)+f'{args.var_loss}.log'), 'w+')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(cfg).items():
        logging.info(k + ':' + str(v))
    # concat absolute path of input files


    
    if cfg.dataset.name == 'tgif-qa': # TODO: PATH [change all paths below]
        cfg.dataset.train_question_pt = '/usr0/home/alyubovs/agqa/storage/questions/tgif-qa_frameqa_train_questions-%s.pt' % args.metric
        cfg.dataset.val_question_pt = '/usr0/home/alyubovs/agqa/storage/questions/tgif-qa_frameqa_val_questions-%s.pt' % args.metric
        cfg.dataset.vocab_json = '/usr0/home/alyubovs/agqa/storage/questions/tgif-qa_frameqa_vocab-balanced.json'
        
        if(args.captions):
            cfg.dataset.train_question_pt = '/usr0/home/alyubovs/agqa/storage/questions_baseline_gen_captions/tgif-qa_frameqa_train_questions-%s.pt' % args.metric
            cfg.dataset.val_question_pt = '/usr0/home/alyubovs/agqa/storage/questions_baseline_gen_captions/tgif-qa_frameqa_val_questions-%s.pt' % args.metric
            cfg.dataset.vocab_json = '/usr0/home/alyubovs/agqa/storage/questions_baseline_gen_captions/tgif-qa_frameqa_vocab-balanced.json'

        cfg.dataset.appearance_feat = cfg.appearance_feat
        cfg.dataset.motion_feat = cfg.motion_feat
        if(cfg.dataset.appearance_feat is None or cfg.dataset.motion_feat is None):
            cfg.dataset.appearance_feat = '/usr0/home/alyubovs/agqa/drive_data/tgif-qa_frameqa_appearance_feat.h5'
            cfg.dataset.motion_feat = '/usr0/home/alyubovs/agqa/drive_data/tgif-qa_frameqa_motion_feat.h5'
        cfg.dataset.caption_dir = cfg.caption_dir
   
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat.h5'
        cfg.dataset.motion_feat = '{}_motion_feat.h5'
        cfg.dataset.vocab_json = '{}_vocab.json'
        cfg.dataset.train_question_pt = '{}_train_questions.pt'
        cfg.dataset.val_question_pt = '{}_val_questions.pt'
        cfg.dataset.train_question_pt = os.path.join(cfg.dataset.data_dir,
                                                     cfg.dataset.train_question_pt.format(cfg.dataset.name))
        cfg.dataset.val_question_pt = os.path.join(cfg.dataset.data_dir,
                                                   cfg.dataset.val_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    # set random seed
    #cfg.seed = random.randint(1, 10000)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    run_name =  f'{cfg.train.dropout}_dif_{cfg.train.loss_ratio}_var_{cfg.train.var_loss}_inv_{cfg.train.inv_loss}_cov_{cfg.train.cov_loss}_loss_dim_{cfg.train.loss_dim}'


    wandb_configs= {'d1_loss': cfg.train.loss_ratio,'cov_loss': cfg.train.cov_loss, \
                     'inv_loss': cfg.train.inv_loss, 'var_loss': cfg.train.var_loss, \
                     'loss_dim': cfg.train.loss_dim, 'loss_ratio': cfg.train.loss_ratio}

    if(args.wandb):
        wandb.init(project="horn", entity="alyudre", config=wandb_configs, name = run_name)
    else:
        wandb.init(mode="disabled")


    train(cfg)


if __name__ == '__main__':
    main()

#after an epoch : 0.4584  val accuracy. test: 42.98 test accuracy 
