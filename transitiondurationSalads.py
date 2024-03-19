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
    

def build_transition_matrix_salads(action_occurrences):
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


def compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences):
    if action_b in transition_probabilities.get(action_a, {}):
        transition_probability = transition_probabilities[action_a][action_b]
        lambda_a = average_occurrences.get(action_a, 0)
        lambda_b = average_occurrences.get(action_b, 0)
        duration_probability_a = poisson.pmf(duration_a, lambda_a) if lambda_a > 0 else 0
        duration_probability_b = poisson.pmf(duration_b, lambda_b) if lambda_b > 0 else 0
        total_probability = transition_probability * duration_probability_a * duration_probability_b
        return total_probability
    return 0

def load_splits_salads(split_file):
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


def load_action_sequences_salads(filenames, directory, action_mapping):
    action_sequences = []
    for filename in filenames:
        filepath = os.path.join(directory, filename )  
        occurrences = parse_file_with_occurrences(filepath, action_mapping)
        action_sequences += occurrences 
    return action_sequences

def get_action_occurrences_test_salads(prediction_dir, action_mapping):
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
    return action_occurrences_test

def get_action_mappings_salads(mapping_file):
    action_mapping = {}
    num_action_mapping = {}

    with open(mapping_file, 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    return action_mapping, num_action_mapping

def get_aggregated_probabilities_salads(action_occurrences_test, transition_probabilities, average_durations):
    aggregated_probabilities = defaultdict(float)

    total_probabilities_test = []
    for i in range(len(action_occurrences_test) - 1):
        action_a, duration_a, f_n = action_occurrences_test[i]
        action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
        if f_n == f_n2:
            total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_durations)
            total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
            aggregated_probabilities[f_n2] += total_probability
    return aggregated_probabilities, total_probabilities_test

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

    # action_mapping = {}
    # num_action_mapping = {}

    # with open(mapping_file, 'r') as f:
    #     for line in f:
    #         number, action = line.strip().split()
    #         action_mapping[action] = int(number)
    #         num_action_mapping[int(number)] = action
    action_mapping, num_action_mapping = get_action_mappings_salads(mapping_file)

    train_filenames = load_splits_salads(train_split_file)
    test_filenames = load_splits_salads(test_split_file)
    train_filenames = [f  for f in load_splits_salads(train_split_file)]
    test_filenames = [f  for f in load_splits_salads(test_split_file)]

    action_sequences_train = load_action_sequences_salads(train_filenames, label_dir, action_mapping)
    action_sequences_test = load_action_sequences_salads(test_filenames, label_dir, action_mapping)

    transition_probabilities, average_durations = build_transition_matrix_salads(action_sequences_train)
    # plot_transition_diagram(transition_probabilities, num_action_mapping)

    # action_occurrences_test = []
    # for filename in os.listdir(prediction_dir):
    #     if filename.endswith('.txt'):
    #         filepath = os.path.join(prediction_dir, filename)
    #         with open(filepath, 'r') as f:
    #             sequence = [action_mapping[line.strip()] for line in f if line.strip() in action_mapping]
    #             actions = []
    #             current_action = None
    #             occurrence = 0
    #             for action in sequence:
    #                 if action == current_action:
    #                     occurrence += 1
    #                 else:
    #                     if current_action is not None:
    #                         actions.append((current_action, occurrence, filename))
    #                     current_action = action
    #                     occurrence = 1
    #             if current_action is not None:
    #                 actions.append((current_action, occurrence, filename))
    #             action_occurrences_test.extend(actions)

    action_occurrences_test = get_action_occurrences_test_salads(prediction_dir, action_mapping)

    # aggregated_probabilities = defaultdict(float)

    # total_probabilities_test = []
    # for i in range(len(action_occurrences_test) - 1):
    #     action_a, duration_a, f_n = action_occurrences_test[i]
    #     action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
    #     if f_n == f_n2:
    #         total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_durations)
    #         total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
    #         aggregated_probabilities[f_n2] += total_probability

    aggregated_probabilities, total_probabilities_test = get_aggregated_probabilities_salads(action_occurrences_test, transition_probabilities, average_durations)

    sorted_total_probabilities_test = sorted(total_probabilities_test, key=lambda x: x[0])
    sorted_aggregated_probabilities = sorted(aggregated_probabilities.items(), key=lambda x: x[1])

    with open('sorted_transitions_50salads.txt', 'w') as file:
        for probability, actions, durations, filn in sorted_total_probabilities_test:
            line = f"Transition in {filn} from {num_action_mapping[actions[0]]} to {num_action_mapping[actions[1]]}, Durations: {durations}, Total Probability: {probability}\n"
            file.write(line)
        file.write("\nAggregated Probabilities per Video Segment:\n")
        for filn, aggregated_probability in sorted_aggregated_probabilities:
            file.write(f"{filn}: Total Aggregated Probability = {aggregated_probability}\n")
