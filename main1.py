import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list,event_list1, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list1)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
        self.event_list1 = event_list1
        self.sample_rate = sample_rate
        self.temporal_aug = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess = postprocess

        self.model = ASDiffusionModel(encoder_params, decoder_params, diffusion_params, self.num_classes, self.device)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset, loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay, label_dir, result_dir, log_freq, log_train_results=True):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        restore_epoch = -1
        step = 1

        if os.path.exists(result_dir):
            if 'latest.pt' in os.listdir(result_dir):
                if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
                    saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
                    self.model.load_state_dict(saved_state['model'])
                    optimizer.load_state_dict(saved_state['optimizer'])
                    restore_epoch = saved_state['epoch']
                    step = saved_state['step']

        if class_weighting:
            class_weights = train_train_dataset.get_class_weights()
            class_weights = torch.from_numpy(class_weights).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')
        
        train_train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)
        
        if result_dir:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                # distinct_values = torch.unique(boundary)
                # print(distinct_values)
                # print(video)
                # print(feature.shape)
                # return 
                # print(label.shape)
                # print(boundary.shape)
                # print(label.cpu().numpy())
                # print(boundary)
                # file_path = 'my_array.txt'

                # Save the array to a text file
                # np.savetxt(file_path, boundary.cpu().numpy().squeeze(), fmt='%d', delimiter=', ')
                # print(soft_label)
                # return 
                # event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1)
                # torch.Size([1, 913])
                # torch.Size([1, 11, 913])
                # print(label.shape)
                # print(event_gt.shape)
                # return 

                loss_dict = self.model.get_training_loss(feature, 
                    event_gt=F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt=boundary,
                    encoder_ce_criterion=ce_criterion, 
                    encoder_mse_criterion=mse_criterion,
                    encoder_boundary_criterion=bce_criterion,
                    decoder_ce_criterion=ce_criterion,
                    decoder_mse_criterion=mse_criterion,
                    decoder_boundary_criterion=bce_criterion,
                    soft_label=soft_label
                )

                # ##############
                # # feature    torch.Size([1, F, T])
                # # label      torch.Size([1, T])
                # # boundary   torch.Size([1, 1, T])
                # # output    torch.Size([1, C, T]) 
                # ##################

                total_loss = 0

                for k,v in loss_dict.items():
                    total_loss += loss_weights[k] * v

                if result_dir:
                    for k,v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')
        
            if result_dir:

                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step
                }

            if epoch % log_freq == 0:

                if result_dir:

                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')
        
                # for mode in ['encoder', 'decoder-noagg', 'decoder-agg']:
                for mode in ['decoder-agg']: # Default: decoder-agg. 
                    # The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        for k,v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k,v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            for k,v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        if result_dir:
            logger.close()

    def plot_segm4(self, path, segmentation, colors,\
         actions_dict_r, name=''):
        fig = plt.figure(figsize=(16, 4))
        plt.axis('off')
        plt.title(name, fontsize=20)

        gt_segm = segmentation['gt']
        ax_idx = 1
        plots_number = len(segmentation)+1
        ax = fig.add_subplot(plots_number, 1, ax_idx)
        ax0=ax
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.box(False)
        ax.set_ylabel('GT', fontsize=20, rotation=0, labelpad=0, verticalalignment='center')
        v_len = len(gt_segm)
        
        delta=5
        label_set=set()
        i=0
        for start, end, label in self.bounds(gt_segm):
            if actions_dict_r[int(label)] in label_set:
                i=1
            else:
                i=0
                label_set.add(actions_dict_r[int(label)])
            # print(start / v_len, end / v_len, label)
            ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1,label='_'*i+actions_dict_r[int(label)])
        ax.legend(loc=1,bbox_to_anchor= (1.15, 1.2))

        plt.box(False)
        cmap = plt.get_cmap('coolwarm')
        for key in segmentation:
            key=str(key)
            if key =='gt':continue
            segm=segmentation[key]
            ax_idx += 1
            ax = fig.add_subplot(plots_number, 1, ax_idx)
            ax.set_ylabel(key, fontsize=20, rotation=0, labelpad=0, verticalalignment='center')
            if key.find('P')!=-1:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                plt.box(False)
                
                for start, end, label in self.bounds(segm):
                    ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                plt.box(False)
                for i,label in enumerate(segm):
                    ax.axvspan(i/v_len, (i+1)/ v_len, facecolor=cmap(label), alpha=1.0)
                ax.axhline(y=0, color='black', linestyle='--', label='Horizontal Line')
        # print(segmentation['gt'])
        # print(segmentation['P'])
        # print(segmentation['gt'] == segmentation['P'])
        print(segmentation['gt'].shape,segmentation['P'].shape)
        correct1 = (segmentation['gt'] == segmentation['P']).sum()
        total = len(gt_segm)
        path_1=path+'/'+'{:.3f}'.format(correct1/total)+'_'+name+'.png'
        print(path_1)
        fig.savefig(path_1, transparent=False)
    def bounds(self,segm):
        start_label = segm[0]
        start_idx = 0
        idx = 0
        while idx < len(segm):
            try:
                while start_label == segm[idx]:
                    idx += 1
            except IndexError:
                yield start_idx, idx, start_label
                break

            yield start_idx, idx, start_label
            start_idx = idx
            start_label = segm[start_idx]
    def test_single_video(self, result_dir, video_idx, test_dataset, mode, device, model_path=None):  
        
        assert(test_dataset.mode == 'test')
        assert(mode in ['encoder', 'decoder-noagg', 'decoder-agg'])
        assert(self.postprocess['type'] in ['median', 'mode', 'purge', None])


        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if self.set_sampling_seed:
            seed = video_idx
        else:
            seed = None
            
        with torch.no_grad():

            feature, label, _, video = test_dataset[video_idx]

            # feature:   [torch.Size([1, F, Sampled T])]
            # label:     torch.Size([1, Original T])
            # output: [torch.Size([1, C, Sampled T])]
            # print(mode)
            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                output = [self.model.ddim_sample(feature[i].to(device), seed) 
                           for i in range(len(feature))] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-noagg':  # temporal aug must be true
                output = [self.model.ddim_sample(feature[len(feature)//2].to(device), seed)] # output is a list of tuples
                output = [i.cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = 0

            assert(output[0].shape[0] == 1)

            min_len = min([i.shape[2] for i in output])
            output = [i[:,:,:min_len] for i in output]
            
            output = torch.mean(torch.cat(output, 0), dim=0)  # torch.Size([sample_rate, C, T])
            top2_scores = torch.topk(output, k=2, dim=0)[0]
            top2_scores1= (top2_scores[0, :] - top2_scores[1, :]).numpy()

            # print("# location 2")
            # print(output.shape)
            # print(top2_scores1.shape)
            # print(np.max(top2_scores1),np.min(top2_scores1))

            output = output.numpy()


            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)
    
            # print(output.shape,type(output))
            # print(np.max(output),np.min(output))
            # print("# location 1")
            # print(output.shape)
            # return 
            output = np.argmax(output, 0)
            segmentation={}
            segmentation['gt']=label.cpu().numpy().astype(np.int32).squeeze()
            segmentation['P']=output
            segmentation['C']=top2_scores1
            print("# location 1")
            print(video)
            # return 

            # print("# location 1")
            print(output.shape,top2_scores1.shape, label.shape,left_offset,right_offset,self.sample_rate)
            # return
            # print(type(output),type(top2_scores1),type(label))
            # return 
            # print(set(segmentation['P']))
            # print(set(segmentation['gt']))
            vid_gt_ex2=set(list(segmentation['P'])+list(segmentation['gt']))
            # print(vid_gt_ex2)
            colors = {}
            cmap = plt.get_cmap('tab20')
            for label_idx, label1 in enumerate(vid_gt_ex2):
                if label1 == -100:
                    colors[label1] = (0, 0, 0)
                else:
                    # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                    colors[label1] = cmap(label_idx / len(vid_gt_ex2))
            self.plot_segm4(result_dir+'/', segmentation, colors,self.event_list,video)
            print(label.shape)
            output,frame_ticks = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )
            # label1=label[:,frame_ticks]
            # print(label1.shape,output.shape)
            # return

            if self.postprocess['type'] == 'mode': # after restoring full sequence
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':

                trans, starts, ends = get_labels_start_end_time(output)
                
                for e in range(0, len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e+1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e-1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid] = trans[e-1]
                            output[mid:ends[e]] = trans[e+1]

            label = label.squeeze(0).cpu().numpy()
            print(output.shape)
            assert(output.shape == label.shape)
            
            return video, output, label


    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        # print("# location ")
        # print(model_path)
            model_name=model_path.split('/')[-1].split('.')[0]
        else:
            model_name='prediction'
            
        with torch.no_grad():
            result_dir1=os.path.join(result_dir, model_name)
            if not os.path.exists(result_dir1):
                os.makedirs(result_dir1)
            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label = self.test_single_video(result_dir1,
                    video_idx, test_dataset, mode, device, model_path)

                pred = [self.event_list1[int(i)] for i in pred]

                file_name = os.path.join(result_dir1, f'{video}.txt')
                print(file_name)
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

        acc, edit, f1s = func_eval(
            label_dir, result_dir1, test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2],
            'Avg':(acc+edit+f1s[0]+f1s[1]+f1s[2])/5.0
        }
        
        return result_dict


# def main():
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--action', type=str,default='train')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--result_dir', default=None)
    parser.add_argument('--naming', default=None)
    parser.add_argument('--sample_rate', default=None)

    args = parser.parse_args()

    all_params = load_config_file(args.config)
    if args.result_dir!=None:
        all_params['result_dir']=args.result_dir
    if args.naming!=None:
        all_params['naming']=args.naming
    if args.sample_rate!=None:
        all_params['sample_rate']=int(args.sample_rate)
    locals().update(all_params)

    if args.dataset_name==None:
        args.dataset_name=dataset_name
    
    feature_dir = os.path.join(root_data_dir, args.dataset_name, 'features')
    label_dir = os.path.join(root_data_dir, args.dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, args.dataset_name, 'mapping.txt')
    mapping_file1 = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]

    event_list1 = np.loadtxt(mapping_file1, dtype=str)
    event_list1 = [i[1] for i in event_list1]
    num_classes = len(event_list1)

    
    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, args.dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, args.dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    train_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=train_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list,event_list1, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    
    # print(args)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if args.action=='train':
        trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
            loss_weights, class_weighting, soft_label,
            num_epochs, batch_size, learning_rate, weight_decay,
            label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
            log_freq=log_freq, log_train_results=log_train_results
        )
    elif args.action=='test':
        test_result_dict = trainer.test(test_test_dataset, 'decoder-agg', trainer.device, label_dir, \
                result_dir=os.path.join(result_dir, naming), model_path=args.model_path)

        print(test_result_dict) 

