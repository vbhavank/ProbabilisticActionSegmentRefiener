import os 
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse
def read_dict_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def plot_segm4(path, segmentation, colors, actions_dict_r, name=''):
    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=20)

    gt_segm = segmentation['GT']
    ax_idx = 1
    plots_number = len(segmentation)+1
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax0=ax
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.box(False)
    ax.set_ylabel('GT', fontsize=16, rotation=0, labelpad=0, verticalalignment='center')
    v_len = len(gt_segm)
    
    delta=5
    label_set=set()
    i=0
    for start, end, label in bounds(gt_segm):
        if actions_dict_r[int(label)] in label_set:
            i=1
        else:
            i=0
            label_set.add(actions_dict_r[int(label)])
        # print(start / v_len, end / v_len, label)
        ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1,label='_'*i+actions_dict_r[int(label)])
    ax.legend(loc=1,bbox_to_anchor= (1.12, 1.2))

    plt.box(False)
    cmap = plt.get_cmap('coolwarm')
    for key in segmentation:
        key=str(key)
        if key =='GT':continue
        segm=segmentation[key]
        ax_idx += 1
        ax = fig.add_subplot(plots_number, 1, ax_idx)
        ax.set_ylabel(key, fontsize=16, rotation=0, labelpad=0, verticalalignment='center')
        if key.find('*')==-1:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.box(False)
            
            for start, end, label in bounds(segm):
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
    # print("# location 1")
    # print(segmentation['GT'].shape,segmentation['WM'].shape)
    correct1 = (segmentation['GT'] == segmentation['WM']).sum()
    total = len(gt_segm)
    path_1=path+'/'+'{:.3f}'.format(correct1/total)+'_'+name+'.png'
    print(path_1)
    fig.savefig(path_1, transparent=False)
def bounds(segm):
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
def ensure_directory_exists(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--split', type=int, default=1)

    args = parser.parse_args()

    mapping_file = '/nfs/hpc/dgx2-6/data/gtea/mapping.txt'
    label_dir = '/nfs/hpc/dgx2-6/data/gtea/groundTruth/'

    action_mapping = {}
    num_action_mapping = {}

    with open(mapping_file, 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    keys=['without_mask','random','mismatch','most_uncertain','PGM_WM']
    keys1=['WM','RM','MM','MU','PGM']
    # keys2=[None,'random_frames_map_GTEA-Trained-S{}.json'.format(args.split),'MM','MU','PGM']

    # paths={'mismatch':'rafid/GTEA-Trained-S{}/{}/prediction_print/'.format(args.split),
    #        }
    # pathA='rafid/GTEA-Trained-S1/after/prediction_print/'
    # pathB='rafid/GTEA-Trained-S1/before/prediction_print/'
    files = os.listdir('rafid/GTEA-Trained-S{}/{}/'.format(args.split,keys[0]))
    dataM=[None]
    for key in ['random_frames_map_GTEA-Trained-S{}.json'.format(args.split),
                'mistaken_frames_map_GTEA-Trained-S{}.json'.format(args.split),
                'most_uncertain_segment_map_GTEA-Trained-S{}.json'.format(args.split),
                # 'most_uncertain_segment_PGM_map_GTEA-Trained-S{}.json'.format(args.split),
                'most_uncertain_segment_PGM_WM_map_GTEA-Trained-S{}.json'.format(args.split)]:
        with open('rafid/GTEA-Trained-S{}/{}'.format(args.split,key), 'r') as file:
            data1 = json.load(file)
        dataM.append(data1)

    # print(data1)
    # return 

    # my_dict = read_dict_from_file('rafid/GTEA-Trained-S1/message1.json')
    # print(video_u.keys())
    # return 

    for name in files:
        name1=name.split('.')[0]
        event_file = label_dir+name
        print(event_file)
        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
        data=[]
        for i,key in enumerate(keys):
            filepath='rafid/GTEA-Trained-S{}/{}/{}'.format(args.split,keys[i],name)
            with open(filepath, 'r') as f:
                # print(filepath)
                data.append([action_mapping[line.strip()] for line in f if line.strip() in action_mapping])
                print(frame_num,len(data[-1]))
        # continue
        
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in action_mapping:
                event_seq_raw[i] = action_mapping[event[i]]
            else:
                event_seq_raw[i] = -100  # background
        segmentation={}
        segmentation['GT']=event_seq_raw.astype(np.int32)
        # print(dataM[1].keys())
        # print(files)
        # return 
        for i in range(len(keys1)):
            segmentation[keys1[i]]=np.array(data[i]).astype(np.int32)
            if i==0:continue
        
            uncertainty_raw = np.zeros((frame_num,))
            print(i)
            if i in [1,2]:
                if i==1:print(len(dataM[i][name1]))
                uncertainty_raw[dataM[i][name1]]=1
            elif i==3:
                uncertainty_raw[dataM[i][name1][0]]=dataM[i][name1][1]
            elif i in [4,5]:
                # uncertainty_raw[dataM[i][name]]=1
                # print(dataM[i][name1])
                # return 
            #     continue
                vm=0
                sm,em=-1,-1
                for s,e,v in dataM[i][name]:            
                    if v>vm:
                        vm=v
                        sm,em=s,e
                uncertainty_raw[sm:em]=vm
                # vm=max(vm,v)
                # uncertainty_raw/=vm
            else:continue
            segmentation[keys1[i]+'*']=uncertainty_raw
        # print(uncertainty_raw)
        # return 


        vid_gt_ex2=[i for i in range(12)]
        colors = {}
        cmap = plt.get_cmap('tab20')
        for label_idx, label1 in enumerate(vid_gt_ex2):
            if label1 == -100:
                colors[label1] = (0, 0, 0)
            else:
                colors[label1] = cmap(label_idx / len(vid_gt_ex2))
        save_path='rafid/GTEA-Trained-S1/plot'.format(args.split)
        ensure_directory_exists(save_path)
        plot_segm4(save_path, segmentation, colors, num_action_mapping, name=name1)
if __name__ == "__main__":
    main()