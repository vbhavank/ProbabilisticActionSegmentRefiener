import os
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2
import math
import pdb
import sys
import argparse
from transitiondurationBreakfast import action_occurrences_from_predictions_breakfast
import re
        
def poisson_confidence_interval(k, alpha=0.05):
    lower = chi2.ppf(alpha / 2, 2 * k) / 2
    upper = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return (lower, upper)


def uncertainity(k, alpha=0.05):
    lambda_ci = poisson_confidence_interval(k)
    lambda_lower, lambda_upper = lambda_ci
    duration_lower = poisson.ppf(0.025, lambda_lower)  
    duration_upper = poisson.ppf(0.975, lambda_upper) 
    return duration_lower, duration_upper 


def plot_duration_dist(average_occurences):
    num_actions = len(average_occurences)
    grid_size = math.ceil(math.sqrt(num_actions))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i, (action, lambda_param, _) in enumerate(average_occurences):
        data = poisson.rvs(mu=lambda_param, size=1000)
        row = i // grid_size
        col = i % grid_size
        axs[row, col].hist(data, bins=30, density=True, alpha=0.6, color='g')
        axs[row, col].set_title(f"Action {action} (λ={lambda_param:.2f})")
        axs[row, col].set_xlim(0, max(data))
    for ax in axs.flat[len(average_occurences):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('./poisson_50salad.png')
    plt.close()
    

def plot_transition_diagram(transition_probabilities, num_action_mapping):
    G = nx.DiGraph()
    for source, targets in transition_probabilities.items():
        for target, probability in targets.items():
            G.add_edge(num_action_mapping[source], num_action_mapping[target], weight=probability, label=f'{probability:.2f}')
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos)
    edges = nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Probabilistic Graphical Model of Action Transitions')
    plt.axis('off')
    plt.savefig('./transition_diagram_50salad.png')
    plt.close()
    

def build_transition_matrix(action_occurrences):
    transition_counts = defaultdict(Counter)
    duration_counts = defaultdict(list)
    for i in range(len(action_occurrences) - 1):
        current_action, _, fn1 = action_occurrences[i]
        next_action, next_duration, fn2 = action_occurrences[i + 1]
        if fn1 == fn2:
            transition_counts[current_action][next_action] += 1
            duration_counts[next_action].append(next_duration)
    transition_probabilities = {action: {next_action: count / sum(transitions.values()) 
                                         for next_action, count in transitions.items()} 
                                for action, transitions in transition_counts.items()}
    average_durations = {action: np.mean(durations) for action, durations in duration_counts.items()}
    return transition_probabilities, average_durations


def compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences, dur_A = False, dur_B = False):
    if action_b in transition_probabilities.get(action_a, {}):
        transition_probability = transition_probabilities[action_a][action_b]
        lambda_a = average_occurrences.get(action_a, 0)
        lambda_b = average_occurrences.get(action_b, 0)
        duration_probability_a = poisson.pmf(duration_a, lambda_a) if lambda_a > 0 else 0
        duration_probability_b = poisson.pmf(duration_b, lambda_b) if lambda_b > 0 else 0
        total_probability = transition_probability
        if dur_A:
            total_probability *= duration_probability_a 
        if dur_B:
            total_probability *= duration_probability_b
        return total_probability
    return 0

def load_splits(split_file):
    with open(split_file, 'r') as file:
        filenames = file.read().splitlines()
    return filenames

def parse_file_with_occurrences(filepath, action_mapping):
    occurrences = []
    filename_only = os.path.basename(filepath)  
    with open(filepath, 'r') as file:
        last_action = None
        count = 0
        for line in file:
            action = line.strip()
            if action not in action_mapping:
                continue
            mapped_action = action_mapping[action]
            if mapped_action == last_action:
                count += 1
            else:
                if last_action is not None:
                    occurrences.append((last_action, count, filename_only))
                last_action = mapped_action
                count = 1
        if last_action is not None:
            occurrences.append((last_action, count, filename_only))
    return occurrences


def load_action_sequences(filenames, directory, action_mapping):
    action_sequences = []
    for filename in filenames:
        filepath = os.path.join(directory, filename )  
        occurrences = parse_file_with_occurrences(filepath, action_mapping)
        action_sequences += occurrences 
    return action_sequences


def parse_prediction_gtea(prediction_dir, action_mapping):
    action_occurrences_test = []
    for filename in os.listdir(prediction_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(prediction_dir, filename)
            with open(filepath, 'r') as f:
                # Assuming each file contains a single long string of actions
                long_string = f.read().strip()
                # Splitting the string into individual actions based on a space delimiter
                actions_list = long_string.split(' ')
                # Mapping actions to their corresponding numbers, if they exist in the mapping
                sequence = [action_mapping[action] for action in actions_list if action in action_mapping]
                
                actions = []
                current_action = None
                occurrence = 0
                for action in sequence:
                    if action == current_action:
                        occurrence += 1
                    else:
                        if current_action is not None:
                            actions.append((current_action, occurrence, filename))
                        current_action = action
                        occurrence = 1
                if current_action is not None:
                    actions.append((current_action, occurrence, filename))
                action_occurrences_test.extend(actions)
    return action_occurrences_test


def compute_ood_percentage(sorted_aggregated_probabilities):
    import math
    total_files = len(sorted_aggregated_probabilities)

    ood_files = [file for file in sorted_aggregated_probabilities if file[0].startswith('P')]
    
    thresholds = {
        "top_5%": math.ceil(total_files * 0.05),
        "top_10%": math.ceil(total_files * 0.1),
        "top_50%": math.ceil(total_files * 0.5),
        "top_75%": math.ceil(total_files * 0.75),
        "top_95%": math.ceil(total_files * 0.95),
        
    }

    counts = {key: 0 for key in thresholds}

    for threshold_name, threshold_value in thresholds.items():
        top_files = sorted_aggregated_probabilities[:threshold_value]
        count_ood = sum(1 for file in top_files if file[0].startswith('P')) # Change for ablation
        counts[threshold_name] = count_ood/threshold_value

    percentages = {key: (value) * 100 for key, value in counts.items()}
    return percentages


def load_gtea_dataset():
    prediction_dir = '/nfs/hpc/dgx2-6/data/result1/50salads-Trained-S1/release'
    action_mapping = {}
    num_action_mapping = {}

    with open('/nfs/hpc/dgx2-6/data/50salads/mapping.txt', 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action

    action_occurrences_test = parse_prediction_gtea(prediction_dir, action_mapping)
    return action_occurrences_test

def load_breakfast_dataset():
    action_mapping = {}
    num_action_mapping = {}

    with open('/nfs/hpc/dgx2-6/data/50salads/mapping.txt', 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    prediction_dir = '/nfs/hpc/dgx2-6/data/result1/50salads-Trained-S1-Tested-BF-S1/release'
    action_occurrences_test = action_occurrences_from_predictions_breakfast(prediction_dir, action_mapping)
    return action_occurrences_test

def do_log(prob):
    if prob == 0.0:
        return 999
    else:
        return np.abs(np.log(prob))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    label_dir = '/nfs/hpc/dgx2-6/data/50salads/groundTruth'
    mapping_file = '/nfs/hpc/dgx2-6/data/50salads/mapping.txt'
    prediction_dir = '/nfs/hpc/dgx2-6/data/50salads/prediction_print'
    train_split_file = '/nfs/hpc/dgx2-6/data/50salads/splits/train.split1.bundle'
    test_split_file = '/nfs/hpc/dgx2-6/data/50salads/splits/test.split1.bundle'

    action_mapping = {}
    num_action_mapping = {}

    with open(mapping_file, 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
            
    train_filenames = load_splits(train_split_file)
    test_filenames = load_splits(test_split_file)
    train_filenames = [f  for f in load_splits(train_split_file)]
    test_filenames = [f  for f in load_splits(test_split_file)]

    action_sequences_train = load_action_sequences(train_filenames, label_dir, action_mapping)
    action_sequences_test = load_action_sequences(test_filenames, label_dir, action_mapping)
    
    transition_probabilities, average_durations = build_transition_matrix(action_sequences_train)
    # plot_transition_diagram(transition_probabilities, num_action_mapping)



    for dura_a in [True, False]:
        for dura_b in [True, False]:
            action_occurrences_test = []
            for filename in os.listdir(prediction_dir):
                if filename.endswith('.txt'):
                    filepath = os.path.join(prediction_dir, filename)
                    with open(filepath, 'r') as f:
                        sequence = [action_mapping[line.strip()] for line in f if line.strip() in action_mapping]
                        actions = []
                        current_action = None
                        occurrence = 0
                        for action in sequence:
                            if action == current_action:
                                occurrence += 1
                            else:
                                if current_action is not None:
                                    actions.append((current_action, occurrence, filename))
                                current_action = action
                                occurrence = 1
                        if current_action is not None:
                            actions.append((current_action, occurrence, filename))
                        action_occurrences_test.extend(actions)
            aggregated_probabilities = defaultdict(float)
            action_occurrences_test_breakfast = load_breakfast_dataset()# Change for ablation
            # action_occurrences_test_gtea = load_gtea_dataset()# Change for ablation
            action_occurrences_test.extend(action_occurrences_test_breakfast)# Change for ablation
            total_probabilities_test = []
            fn_counter = defaultdict(int)
            for i in range(len(action_occurrences_test) - 1):
                action_a, duration_a, f_n = action_occurrences_test[i]
                action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
                if f_n == f_n2:
                    total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_durations, dura_a, dura_b)
                    aggregated_probabilities[f_n2] += do_log(total_probability)

            sorted_aggregated_probabilities = sorted(aggregated_probabilities.items(), key=lambda x: x[1], reverse=True)
            file_name_list = []
            percentages = compute_ood_percentage(sorted_aggregated_probabilities)

            with open('salads_{}_{}.txt'.format(dura_a, dura_b), 'w') as file:
                file.write("\nAggregated Uncertainity per Video Segment:\n")
                file_name_list = []

                for filn, aggregated_probability in sorted_aggregated_probabilities:
                    file.write(f"{filn}: Total Aggregated Uncertainity = {aggregated_probability}\n")
                file.write("\nF@K%:\n")
                file.write(f"Percentage of OOD files in the top 5%: { percentages['top_5%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 10%: {percentages['top_10%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 50%: {percentages['top_50%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 75%: {percentages['top_75%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 95%: {percentages['top_95%']:.2f}% \n")
                