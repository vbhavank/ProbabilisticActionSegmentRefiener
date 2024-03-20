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
from transitionduration import get_action_mappings, get_action_occurences_train, get_test_action_occurences, build_transition_matrix, get_total_probabilities
from transitiondurationBreakfast import get_action_mappings_breakfast, load_splits_breakfast, load_action_sequences_breakfast, build_transition_matrix_breakfast, action_occurrences_from_predictions_breakfast, get_total_probabilities_breakfast
from transitiondurationSalads import get_action_mappings_salads, load_splits_salads, load_action_sequences_salads, build_transition_matrix_salads, get_action_occurrences_test_salads, get_aggregated_probabilities_salads

class NumpyFloatEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return str(obj)
        return JSONEncoder.default(self, obj)
    
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
        # print(f"length output: {len(output)}\nlength top2_scores: {len(top2_scores)}")
        # print(f"output: {output}")
        for i in range(len(output)):
            # print(f"i={i}")
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
        frames = np.where(output != ground_truth)[0]
        return frames
    
    def get_k_most_uncertain_frames(self, top2_score, k):
        values, frames = torch.topk(top2_score, k=k, largest=False)
        return frames, values
    
    def get_random_frames(self, total_frames, k):
        all_frames = np.arange(total_frames, dtype=np.int32)
        frames = np.random.choice(all_frames, size=k, replace=False)
        return frames


    def test_single_video(self, video_idx, test_dataset, mode, device, model_path=None, most_uncertain_frames=None, mistaken_frames=None, random_mask=None, seq_segment_mask=None):  
        
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

            if mode == 'encoder':
                output = [self.model.encoder(feature[i].to(device)) 
                       for i in range(len(feature))] # output is a list of tuples
                
                output = [F.softmax(i, 1).cpu() for i in output]
                left_offset = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            if mode == 'decoder-agg':
                if mistaken_frames is not None:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, mistaken_frames=mistaken_frames[video_idx]) 
                            for i in range(len(feature))] # output is a list of tuples
                elif most_uncertain_frames is not None:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, most_uncertain_segment=most_uncertain_frames[video_idx]) 
                            for i in range(len(feature))]
                elif random_mask is not None:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, random_mask=random_mask[video_idx]) 
                            for i in range(len(feature))]
                elif seq_segment_mask is not None:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, seq_segment_mask=seq_segment_mask) 
                            for i in range(len(feature))]
                else:
                    output = [self.model.ddim_sample(feature[i].to(device), seed) 
                            for i in range(len(feature))]
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
            top2_scores1 = top2_scores[0, :] - top2_scores[1, :]
            output = output.numpy()

            if self.postprocess['type'] == 'median': # before restoring full sequence
                smoothed_output = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed_output[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed_output / smoothed_output.sum(0, keepdims=True)

            output = np.argmax(output, 0)
            
            output1 = output
            output, frame_ticks = restore_full_sequence(output, 
                full_len=label.shape[-1], 
                left_offset=left_offset, 
                right_offset=right_offset, 
                sample_rate=self.sample_rate
            )
            label1 = label.squeeze(0).cpu().numpy()[frame_ticks]
            # print(f"label1: {label1}")

            if most_uncertain_frames is None:
                acc = (output1 == label1).sum() / len(output1)
                most_uncertain_frames, values =  self.get_k_most_uncertain_frames(top2_scores1, int(len(top2_scores) * acc))
            else:
                most_uncertain_frames, values = None, None

            if mistaken_frames is None:
                mistaken_frames = self.mistaken_segments(output1, label1)
            else:
                mistaken_frames = None

            if random_mask is None:
                acc = (output1 == label1).sum() / len(output1)
                # print(f"output1 len: {len(output1)}\nacc: {acc}")
                random_mask = self.get_random_frames(len(output1), int(len(output1) * acc))
            else:
                random_mask = None
            # print(f"mistaken frames: {mistaken_frames}\n\nframe ticks: {frame_ticks}")
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

            
            
            assert(output.shape == label.shape)
            return video, output, label, most_uncertain_frames, mistaken_frames, random_mask, values

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
            

    def test(self, test_dataset, mode, device, label_dir, result_dir=None, model_path=None, most_uncertain_segments=None, mistaken_frames=None, random_mask=None, video_most_uncertain_segment_map=None):
        
        assert(test_dataset.mode == 'test')

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        if most_uncertain_segments is None:
            most_uncertain_segments_1 = []
            most_uncertain_segments_1_dict = {}
        
        if mistaken_frames is None:
            mistaken_frames_1 = []
            mistaken_frames_1_dict = {}

        if random_mask is None:
            random_mask_1 = []
            random_mask_1_dict = {}
        labels = {}
        with torch.no_grad():

            for video_idx in tqdm(range(len(test_dataset))):
                _, _, _, video = test_dataset[video_idx]

                video, pred, label, most_uncertain_segment, mistaken_frames_per_video, random_mask_per_video, most_uncertain_values = self.test_single_video(
                    video_idx, test_dataset, mode, device, model_path, most_uncertain_segments, mistaken_frames, random_mask, video_most_uncertain_segment_map[video] if video_most_uncertain_segment_map is not None else None)

                pred = [self.event_list[int(i)] for i in pred]
                label = [self.event_list[int(i)] for i in label]
                labels[video] = label
                # print(f"for video: {video}\nlabels: {labels[video]}\nlen label: {len(labels[video])}\nlen pred: {len(pred)}")
                # exit()
                if most_uncertain_segments is None:
                    most_uncertain_segments_1.append(most_uncertain_segment)
                    most_uncertain_segments_1_dict[video] = (most_uncertain_segment.numpy(), most_uncertain_values.numpy())

                if mistaken_frames is None:
                    mistaken_frames_1.append(mistaken_frames_per_video)
                    mistaken_frames_1_dict[video] = mistaken_frames_per_video

                if random_mask is None:
                    random_mask_1.append(random_mask_per_video)
                    random_mask_1_dict[video] = random_mask_per_video

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
            with open(f'most_uncertain_segment_map_{naming}.json', 'w') as fp:
                json.dump(most_uncertain_segments_1_dict, fp, cls=NumpyArrayEncoder)

        if mistaken_frames is None:
            mistaken_frames = mistaken_frames_1
            with open(f'mistaken_frames_map_{naming}.json', 'w') as fp:
                json.dump(mistaken_frames_1_dict, fp, cls=NumpyArrayEncoder)

        if random_mask is None:
            random_mask = random_mask_1
            with open(f'random_frames_map_{naming}.json', 'w') as fp:
                json.dump(random_mask_1_dict, fp, cls=NumpyArrayEncoder)

        print(f"\nresult: {result_dict}")
        return result_dict, most_uncertain_segments, mistaken_frames, random_mask

def get_uncertain_segment_PGM(naming, action_mapping, action_occurrences_train):
    prediction_dir = f"./result/PGM/{naming}/prediction_print"
    aggregated_probabilities = None

    if 'GTEA' in naming:
        action_occurrences_test = get_test_action_occurences(prediction_dir, action_mapping)
        # print(f"action_occurs test: {action_occurrences_test}")
        transition_probabilities, average_occurrences = build_transition_matrix(action_occurrences_train)
        # print(f"transition probs: {transition_probabilities}\naverage occurs: {average_occurrences}")
        aggregated_probabilities, _ = get_total_probabilities(action_occurrences_test, transition_probabilities, average_occurrences)
    elif 'Breakfast' in naming:
        transition_probabilities, average_durations = build_transition_matrix_breakfast(action_occurrences_train)
        action_occurrences_test = action_occurrences_from_predictions_breakfast(prediction_dir, action_mapping)
        aggregated_probabilities, _ = get_total_probabilities_breakfast(action_occurrences_test, transition_probabilities, average_durations)
    elif '50salads' in naming:
        transition_probabilities, average_durations = build_transition_matrix_salads(action_occurrences_train)
        action_occurrences_test = get_action_occurrences_test_salads(prediction_dir, action_mapping)
        aggregated_probabilities, _ = get_aggregated_probabilities_salads(action_occurrences_test, transition_probabilities, average_durations)
    return aggregated_probabilities


def get_segments(pred_file, mapping_file, naming, trainer: Trainer):
    action_mapping, _ = get_action_mappings(mapping_file)
    sequence_segments = {}

    with open(pred_file, 'r') as f:
        # print(f"pred_file: {pred_file}")
        sequence = [action_mapping[line.strip()] for line in f if line.strip() in action_mapping.keys()]
        if '50salads' in naming:
            full_len = len(sequence)
            left_offset = trainer.sample_rate // 2
            right_offset = (trainer.sample_rate - 1) // 2
            frame_ticks = np.arange(left_offset, full_len-right_offset, trainer.sample_rate)
            
            sequence = [sequence[i] for i in frame_ticks]

        segment_index = 0
        seq = -1
        for i in range(len(sequence)):
            if i == 0:
                seq = sequence[i]
                sequence_segments[0] = [i]
            elif seq == sequence[i]:
                sequence_segments[segment_index].append(i)
            elif seq != sequence[i]:
                segment_index += 1
                sequence_segments[segment_index] = [i]
                seq = sequence[i]
    return sequence_segments



def get_most_uncertain_segment_PGM(naming, label_dir_seq, previous_pred_dir, trainer: Trainer, test_dataset, model_path, device):

    trainer.model.eval()
    trainer.model.to(device)

    if model_path:
        trainer.model.load_state_dict(torch.load(model_path))

    prediction_dir = f"./result/PGM/{naming}/prediction_print"
    
    if 'GTEA' in naming:
        label_dir = "./datasets/gtea/labels"
        
        mapping_file = "./datasets/gtea/mapping.txt"
        action_mapping, num_action_mapping = get_action_mappings(mapping_file)
        # print(f"actions: {action_mapping}")
        action_occurrences_train = get_action_occurences_train(label_dir, action_mapping)

    elif "Breakfast" in naming:
        label_dir = './datasets/breakfast/groundTruth'
        mapping_file = './datasets/breakfast/mapping.txt'
        train_split_file = './datasets/breakfast/splits/train.split1.bundle'
        # test_split_file = '/nfs/hpc/dgx2-6/data/breakfast/splits/test.split1.bundle'
        train_filenames = load_splits_breakfast(train_split_file)
        # test_filenames = load_splits_breakfast(test_split_file)
        train_filenames = [f  for f in load_splits_breakfast(train_split_file)]
        # test_filenames = [f  for f in load_splits_breakfast(test_split_file)]
        action_mapping, num_action_mapping = get_action_mappings_breakfast(mapping_file)
        action_occurrences_train = load_action_sequences_breakfast(train_filenames, label_dir, action_mapping)
    
    elif "50salads" in naming:
        label_dir = './datasets/50salads/groundTruth'
        mapping_file = './datasets/50salads/mapping.txt'
        train_split_file = './datasets/50salads/splits/train.split1.bundle'

        train_filenames = load_splits_salads(train_split_file)
        train_filenames = [f  for f in load_splits_salads(train_split_file)]
        action_mapping, num_action_mapping = get_action_mappings_salads(mapping_file)
        action_occurrences_train = load_action_sequences_salads(train_filenames, label_dir, action_mapping)
    else:
        action_mapping = None
        action_occurrences_train = None

    segments = {}
    for pred_file in os.listdir(label_dir_seq):
        if pred_file.endswith('.txt'):
            sequence_segments = get_segments(f"{label_dir_seq}/{pred_file}", mapping_file, naming, trainer)
            video_name = pred_file.split('.')[0]
            segments[video_name] = sequence_segments
    #         print(f"\nfor video {video_name} seqs: {segments[video_name]}")
    # exit()
    video_most_uncertain_segment_map = {}

    video_segments_uncertainty_map = {}
    with torch.no_grad():
        for video_idx in tqdm(range(len(test_dataset))):
            _, _, _, video = test_dataset[video_idx]
            probs = -1
            most_uncertain_segment_index = None
            for segment_idx in segments[video].keys():
                video, pred, label, _, _, _, _ = trainer.test_single_video(
                video_idx, test_dataset, "decoder-agg", device, model_path, None, None, None, seq_segment_mask=segments[video][segment_idx])

                pred = [trainer.event_list[int(i)] for i in pred]
                if not os.path.exists(prediction_dir):
                    os.makedirs(prediction_dir)
                print_pred = trainer.print_preds(pred)
                file_name = os.path.join(prediction_dir, f'{video}.txt')
                file_ptr = open(file_name, 'w')
                file_ptr.write('### Frame level recognition: ###\n')
                file_ptr.write(print_pred)
                file_ptr.close()

                aggregated_probabilities = get_uncertain_segment_PGM(naming, action_mapping, action_occurrences_train)
                # print(f"segment {segment_idx} aggregated: {aggregated_probabilities}")
                if f"{video}.txt" not in video_segments_uncertainty_map.keys():
                    video_segments_uncertainty_map[f"{video}.txt"] = [(segments[video][segment_idx][0], segments[video][segment_idx][-1], aggregated_probabilities[f"{video}.txt"])]
                else:
                    video_segments_uncertainty_map[f"{video}.txt"].append((segments[video][segment_idx][0], segments[video][segment_idx][-1], aggregated_probabilities[f"{video}.txt"]))

                if probs < aggregated_probabilities[f"{video}.txt"]:
                    probs = aggregated_probabilities[f"{video}.txt"]
                    most_uncertain_segment_index = segment_idx
            video_most_uncertain_segment_map[video] = segments[video][most_uncertain_segment_index]

            file_name = os.path.join(prediction_dir, f'{video}.txt')
            if os.path.exists(file_name):
                os.remove(file_name)
        # print(f"video uncertain segment map: {video_most_uncertain_segment_map}")
    # print(f"video_segments_uncertainty: {video_segments_uncertainty_map}")
    with open(f'most_uncertain_segment_PGM_map_{naming}.json', 'w') as fp:
        json.dump(video_most_uncertain_segment_map, fp)
    return video_most_uncertain_segment_map            



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
    print(f"Without any mask:")
    model_path = f"./trained_models/{naming}/release.model"
    result_dict, most_uncertain_segments, mistaken_frames, random_frames = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/{naming}", model_path=model_path)
    uncertain_segments_result = f"./uncertain_frames/{naming}"
    if not os.path.exists(uncertain_segments_result):
        os.makedirs(uncertain_segments_result)
    # np.save(f"{uncertain_segments_result}/most_uncertain_frames.npy", most_uncertain_segments)

    result_matrices = f"./result_matrices/{naming}"
    if not os.path.exists(result_matrices):
        os.makedirs(result_matrices)
    with open(f"{result_matrices}/without_mask_metrices.json", "w") as outfile: 
        json.dump(result_dict, outfile, cls=NumpyFloatEncoder)
    
    print(f"PGM most uncertain mask:")
    video_most_uncertain_segment_map = get_most_uncertain_segment_PGM(naming, label_dir, f"{result_dir}/{naming}/prediction_print", trainer, test_test_dataset, model_path, device='cuda')
 
    result_dict, most_uncertain_segments, mistaken_frames, random_frames = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/most_uncertain_segment_PGM/{naming}", model_path=model_path, video_most_uncertain_segment_map=video_most_uncertain_segment_map)
    # with open(f"{result_matrices}/without_mask_metrices.json", "w") as outfile: 
    #     json.dump(result_dict, outfile, cls=NumpyFloatEncoder)

    print(f"\nwith most uncertain mask:")
    # most_uncertain_segments = np.load(f"{uncertain_segments_result}/most_uncertain_frames.npy")
    result_dict, _, _, _ = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/most_uncertain/{naming}", model_path=model_path, most_uncertain_segments=most_uncertain_segments)
    # with open(f"{result_matrices}/with_uncertain_mask_metrices.json", "w") as outfile: 
    #     json.dump(result_dict, outfile, cls=NumpyFloatEncoder)
        
    print(f"\nWith mismatch mask:")
    result_dict, _, _, _ = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/mismatch/{naming}", model_path=model_path, most_uncertain_segments=None, mistaken_frames=mistaken_frames)
    # with open(f"{result_matrices}/with_mismatch_mask_metrices.json", "w") as outfile: 
    #     json.dump(result_dict, outfile, cls=NumpyFloatEncoder)

    print(f"\nWith random mask:")
    result_dict, _, _, _ = trainer.test(test_test_dataset, mode="decoder-agg", device='cuda', label_dir=label_dir, result_dir=f"{result_dir}/random/{naming}", model_path=model_path, random_mask=random_frames)
    # with open(f"{result_matrices}/with_random_mask_metrices.json", "w") as outfile: 
    #     json.dump(result_dict, outfile, cls=NumpyFloatEncoder)

    
    