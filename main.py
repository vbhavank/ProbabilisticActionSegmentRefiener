import os
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
#from torch.utils.tensorboard import SummaryWriter
from dataset import restore_full_sequence
from dataset import get_data_dict
from dataset import VideoFeatureDataset
from model import ASDiffusionModel
from tqdm import tqdm
from utils import load_config_file, func_eval, set_random_seed, get_labels_start_end_time
from utils import mode_filter
import matplotlib.pyplot as plt
import json
from json import JSONEncoder

class NumpyFloatEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return str(obj)
        return JSONEncoder.default(self, obj)

class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params, 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess, device):

        self.device = device
        self.num_classes = len(event_list)
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.event_list = event_list
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

        result_loss = []

        # if os.path.exists(result_dir):
        #     if 'latest.pt' in os.listdir(result_dir):
        #         if os.path.getsize(os.path.join(result_dir, 'latest.pt')) > 0:
        #             saved_state = torch.load(os.path.join(result_dir, 'latest.pt'))
        #             self.model.load_state_dict(saved_state['model'])
        #             optimizer.load_state_dict(saved_state['optimizer'])
        #             restore_epoch = saved_state['epoch']
        #             step = saved_state['step']

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
            # logger = SummaryWriter(result_dir)
        
        for epoch in range(restore_epoch+1, num_epochs):

            self.model.train()
            
            epoch_running_loss = 0
            
            for _, data in enumerate(train_train_loader):

                feature, label, boundary, video = data
                feature, label, boundary = feature.to(device), label.to(device), boundary.to(device)
                
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

                # if result_dir:
                #     for k,v in loss_dict.items():
                #         logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                #     logger.add_scalar('Train-Total', total_loss.item() / batch_size, step)

                total_loss /= batch_size
                total_loss.backward()
        
                epoch_running_loss += total_loss.item()
                
                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                
            epoch_running_loss /= len(train_train_dataset)

            print(f'Epoch {epoch} - Running Loss {epoch_running_loss}')

            result_loss.append(epoch_running_loss)
        
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
                for mode in ['decoder-agg']: # Default: decoder-agg. The results of decoder-noagg are similar

                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        result_dir=result_dir, model_path=None)

                    if result_dir:
                        # for k,v in test_result_dict.items():
                        #     logger.add_scalar(f'Test-{mode}-{k}', v, epoch)

                        np.save(os.path.join(result_dir, 
                            f'test_results_{mode}_epoch{epoch}.npy'), test_result_dict)

                    for k,v in test_result_dict.items():
                        print(f'Epoch {epoch} - {mode}-Test-{k} {v}')


                    if log_train_results:

                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            result_dir=result_dir, model_path=None)

                        if result_dir:
                            # for k,v in train_result_dict.items():
                            #     logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                                 
                            np.save(os.path.join(result_dir, 
                                f'train_results_{mode}_epoch{epoch}.npy'), train_result_dict)
                            
                        for k,v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')
                        
        # if result_dir:
        #     logger.close()
                            
        plt.figure(figsize=(16,9))
        epochs = np.arange(len(result_loss))
        plt.plot(epochs, result_loss, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"./training_loss.png", dpi=300)
        plt.close()


    def get_most_uncertain_segment(self, top2_scores, output):
        action_segments = {}
        segment_action = -1
        segment_uncertainty = []
        segment_index = 0
        for i in range(len(output)):
            if i == 0:
                segment_action = output[i]
                segment_uncertainty.append(top2_scores[i])
                action_segments[segment_index] = [i, -1]
            elif i == len(output) - 1:
                action_segments[segment_index][1] = i
                segment_uncertainty[segment_index] += top2_scores[i]
                segment_uncertainty[segment_index] /= (action_segments[segment_index][1] - action_segments[segment_index][0] + 1)
            elif segment_action == output[i]:
                segment_uncertainty[segment_index] += top2_scores[i]
            elif segment_action != output[i]:
                action_segments[segment_index][1] = i - 1
                segment_uncertainty[segment_index] /= (action_segments[segment_index][1] - action_segments[segment_index][0] + 1)
                segment_index += 1
                segment_action = output[i]
                action_segments[segment_index] = [i, -1]
                segment_uncertainty.append(top2_scores[i])
                
        most_uncertain_segment = np.argmin(np.array(segment_uncertainty))
        return action_segments[most_uncertain_segment]
    
    def mistaken_segments(self, output, ground_truth):
        mismatch = output != ground_truth
        print(f"mismatch: {mismatch}")
        indices = np.where(np.all(output != ground_truth))
        print(f"indices: {indices}")


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None, most_uncertain_segments=None):  
        
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
            if most_uncertain_segments is not None:
                frames = most_uncertain_segments[video_idx]
                feature[0][:,:,frames[0]:frames[1]] = 0

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
            # output = torch.cat(output, 0)  # torch.Size([sample_rate, C, T])
            # output = output.mean(0).numpy()

            output = torch.mean(torch.cat(output, 0), dim=0)  # torch.Size([sample_rate, C, T])
            top2_scores = torch.topk(output, k=2, dim=0)[0]
            top2_scores1= (top2_scores[0, :] - top2_scores[1, :]).numpy()
            output = output.numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)

            output = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )
            # print(f"restore seq: {output}")
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
                        # print(f"output: {output}")
            label = label.squeeze(0).cpu().numpy()
            if most_uncertain_segments is None:
                most_uncertain_segment = self.get_most_uncertain_segment(top2_scores1, output)
            else:
                most_uncertain_segment = None

            self.mistaken_segments(output, label)
            assert(output.shape == label.shape)
            
            return video, output, label, most_uncertain_segment

    def print_preds(self, pred):
        curr_action = -1
        print_pred = ''
        for p in pred:
            # if curr_action == -1:
            #     curr_action = p
            # elif curr_action != p:
            #     print_pred += '\n'
            #     curr_action = p
            print_pred += f'{p}\n'
        return print_pred.strip()
            

    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, most_uncertain_segments=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if most_uncertain_segments is None:
            most_uncertain_segments_1 = []
        
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                
                video, pred, label, most_uncertain_segment = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path, most_uncertain_segments)

                pred = [self.event_list[int(i)] for i in pred]

                if most_uncertain_segment is not None:
                    most_uncertain_segments_1.append(most_uncertain_segment)

                if not os.path.exists(os.path.join(result_dir, 'prediction')):
                    os.makedirs(os.path.join(result_dir, 'prediction'))

                file_name = os.path.join(result_dir, 'prediction', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(' '.join(pred))
                file_ptr.close()

                if not os.path.exists(os.path.join(result_dir, 'prediction_print')):
                    os.makedirs(os.path.join(result_dir, 'prediction_print'))
                print_pred = self.print_preds(pred)
                file_name = os.path.join(result_dir, 'prediction_print', f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(print_pred)
                file_ptr.close()

        acc, edit, f1s = func_eval(
            label_dir, os.path.join(result_dir, 'prediction'), test_dataset.video_list)

        result_dict = {
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }
        if most_uncertain_segments is None:
            most_uncertain_segments = most_uncertain_segments_1
        print(f"\nresult: {result_dict}\n\nmostuncertain_segs: {most_uncertain_segments}")
        return result_dict, most_uncertain_segments


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    print(args.config)
    print(all_params)

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    feature_dir = os.path.join(root_data_dir, dataset_name, 'features')
    label_dir = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    event_list = np.loadtxt(mapping_file, dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    train_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'train.split{split_id}.bundle'), dtype=str)
    test_video_list = np.loadtxt(os.path.join(
        root_data_dir, dataset_name, 'splits', f'test.split{split_id}.bundle'), dtype=str)

    train_video_list = [i.split('.')[0] for i in train_video_list]
    test_video_list = [i.split('.')[0] for i in test_video_list]

    # train_data_dict = get_data_dict(
    #     feature_dir=feature_dir, 
    #     label_dir=label_dir, 
    #     video_list=train_video_list, 
    #     event_list=event_list, 
    #     sample_rate=sample_rate, 
    #     temporal_aug=temporal_aug,
    #     boundary_smooth=boundary_smooth
    # )

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, 
        label_dir=label_dir, 
        video_list=test_video_list, 
        event_list=event_list, 
        sample_rate=sample_rate, 
        temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth
    )
    
    # train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
    # train_test_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='test')
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(dict(encoder_params), dict(decoder_params), dict(diffusion_params), 
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # trainer.train(train_train_dataset, train_test_dataset, test_test_dataset, 
    #     loss_weights, class_weighting, soft_label,
    #     num_epochs, batch_size, learning_rate, weight_decay,
    #     label_dir=label_dir, result_dir=os.path.join(result_dir, naming), 
    #     log_freq=log_freq, log_train_results=log_train_results
    # )
    model_path = f"./trained_models/{naming}/release.model"
    result_dict, most_uncertain_segments = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/{naming}", model_path=model_path)
    uncertain_segments_result = f"./uncertain_frames/{naming}"
    if not os.path.exists(uncertain_segments_result):
        os.makedirs(uncertain_segments_result)

    np.save(f"{uncertain_segments_result}/most_uncertain_frames.npy", most_uncertain_segments)

    result_matrices = f"./result_matrices/{naming}"
    if not os.path.exists(result_matrices):
        os.makedirs(result_matrices)
    
    with open(f"{result_matrices}/without_mask_metrices.json", "w") as outfile: 
        json.dump(result_dict, outfile, cls=NumpyFloatEncoder)

    most_uncertain_segments = np.load(f"{uncertain_segments_result}/most_uncertain_frames.npy")
    result_dict, _ = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/{naming}", model_path=model_path, most_uncertain_segments=most_uncertain_segments)
    with open(f"{result_matrices}/with_mask_metrices.json", "w") as outfile: 
        json.dump(result_dict, outfile, cls=NumpyFloatEncoder)