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
    num_actions = len(average_occurrences)
    grid_size = math.ceil(math.sqrt(num_actions))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    for i, (action, lambda_param) in enumerate(average_occurrences.items()):
        data = poisson.rvs(mu=lambda_param, size=1000)
        row = i // grid_size
        col = i % grid_size
        axs[row, col].hist(data, bins=30, density=True, alpha=0.6, color='g')
        axs[row, col].set_title(f"Action {action} (λ={lambda_param:.2f})")
        axs[row, col].set_xlim(0, max(data))
    for ax in axs.flat[len(average_occurrences):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig('./poisson.png')
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
    plt.savefig('./transition_diagram_pos.png')
    plt.close()
    

def build_transition_matrix(action_occurrences):
    occurrence_counts = defaultdict(list)
    for action, duration, _ in action_occurrences:
        occurrence_counts[action].append(duration)
    
    average_occurrences = {action: np.mean(occurrences) for action, occurrences in occurrence_counts.items()}
    
    # Extract action sequences from occurrences for transition probabilities
    action_sequences = [[action for action, _, _ in action_occurrences]]
    transition_counts = defaultdict(Counter)
    
    for sequence in action_sequences:
        for i in range(len(sequence) - 1):
            current_action = sequence[i]
            next_action = sequence[i + 1]
            transition_counts[current_action][next_action] += 1
    transition_probabilities = defaultdict(dict)
    for current_action, transitions in transition_counts.items():
        total = sum(transitions.values())
        for next_action, count in transitions.items():
            transition_probabilities[current_action][next_action] = count / total
    return transition_probabilities, average_occurrences


def compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences):
    if action_b in transition_probabilities.get(action_a, {}):
        transition_probability = transition_probabilities[action_a][action_b]
        lambda_a = average_occurrences.get(action_a, 0)
        lambda_b = average_occurrences.get(action_b, 0)
        duration_probability_a = poisson.pmf(duration_a, lambda_a) if lambda_a > 0 else 0
        duration_probability_b = poisson.pmf(duration_b, lambda_b) if lambda_b > 0 else 0
        # total_probability = transition_probability * duration_probability_a * duration_probability_b
        total_probability = transition_probability #* duration_a
        return total_probability
    return 0


def parse_line(line, filename, action_mapping):
    try:
        action_info, frame_range, split = line.strip().split(' ')
        action_verb = action_info.split('><')[0].strip('<')  # Extract the verb
        split = split[1]  
        frame_start, frame_end = map(int, frame_range.strip('()').split('-'))
        duration = frame_end - frame_start + 1
        return action_mapping[action_verb], duration, filename.split('.')[0], split
    except ValueError: 
        return None

    
# label_dir = '/nfs/hpc/dgx2-6/data/gtea/labels'
# prediction_dir = '/nfs/hpc/dgx2-6/data/gtea/prediction_print'
# mapping_file = '/nfs/hpc/dgx2-6/data/gtea/mapping.txt'
# action_mapping = {}
# num_action_mapping = {}

# with open(mapping_file, 'r') as f:
#     for line in f:
#         number, action = line.strip().split()
#         action_mapping[action] = int(number)
#         num_action_mapping[int(number)] = action
        
# action_occurrences_train = []
# action_occurrences_test = []

# for filename in os.listdir(label_dir):
#     if filename.endswith('.txt'):
#         filepath = os.path.join(label_dir, filename)
#         with open(filepath, 'r') as f:
#             for line in f:
#                 parsed_line = parse_line(line, filename, action_mapping)
#                 if parsed_line == None:
#                     continue
#                 if not int(parsed_line[3]):
#                     action_occurrences_train.append(parsed_line[:3])


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
            
# transition_probabilities, average_occurrences = build_transition_matrix(action_occurrences_train)
# plot_transition_diagram(transition_probabilities, num_action_mapping)
# plot_duration_dist(average_occurrences)

# total_probabilities_test = []
# for i in range(len(action_occurrences_test) - 1):
#     action_a, duration_a, f_n = action_occurrences_test[i]
#     action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
#     if f_n == f_n2:
#         total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences)
#         total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
# sorted_total_probabilities_test = sorted(total_probabilities_test, key=lambda x: x[0])

# with open('sorted_transitions.txt', 'w') as file:
#     for probability, actions, durations, filn in sorted_total_probabilities_test:
#         line = f"Transition in {filn} from {num_action_mapping[actions[0]]} to {num_action_mapping[actions[1]]}, Durations: {durations}, Total Probability: {probability}\n"
#         file.write(line)
def get_test_action_occurences(prediction_dir, action_mapping):
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

def get_action_occurences_train(label_dir, action_mapping):
    action_occurrences_train = []

    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parsed_line = parse_line(line, filename, action_mapping)
                    if parsed_line == None:
                        continue
                    if not int(parsed_line[3]):
                        action_occurrences_train.append(parsed_line[:3])
    return action_occurrences_train

def get_action_mappings(mapping_file):
    action_mapping = {}
    num_action_mapping = {}

    with open(mapping_file, 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    return action_mapping, num_action_mapping

def do_log(prob):
    if prob == 0.0:
        return 0
    else:
        return np.abs(np.log(prob))
    
def get_total_probabilities(action_occurrences_test, transition_probabilities, average_occurrences):
    aggregated_probabilities = defaultdict(float)

    total_probabilities_test = []
    for i in range(len(action_occurrences_test) - 1):
        action_a, duration_a, f_n = action_occurrences_test[i]
        action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
        if f_n == f_n2:
            total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences)
            total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
            # aggregated_probabilities[f_n2] *= total_probability
            aggregated_probabilities[f_n2] += do_log(total_probability)
    return aggregated_probabilities, total_probabilities_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    label_dir = "./datasets/gtea/labels" # '/nfs/hpc/dgx2-6/data/gtea/labels'
    prediction_dir = "./result/GTEA-Trained-S1/prediction_print" # '/nfs/hpc/dgx2-6/data/gtea/prediction_print'
    mapping_file = "./datasets/gtea/mapping.txt" # '/nfs/hpc/dgx2-6/data/gtea/mapping.txt'
    # action_mapping = {}
    # num_action_mapping = {}

    # with open(mapping_file, 'r') as f:
    #     for line in f:
    #         number, action = line.strip().split()
    #         action_mapping[action] = int(number)
    #         num_action_mapping[int(number)] = action

    action_mapping, num_action_mapping = get_action_mappings(mapping_file)
            
    action_occurrences_train = get_action_occurences_train(label_dir, action_mapping)

    action_occurrences_test = get_test_action_occurences(prediction_dir, action_mapping)
                
    transition_probabilities, average_occurrences = build_transition_matrix(action_occurrences_train)
    plot_transition_diagram(transition_probabilities, num_action_mapping)
    plot_duration_dist(average_occurrences)
    # aggregated_probabilities = defaultdict(float)

    # total_probabilities_test = []
    # for i in range(len(action_occurrences_test) - 1):
    #     action_a, duration_a, f_n = action_occurrences_test[i]
    #     action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
    #     if f_n == f_n2:
    #         total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences)
    #         total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
    #         aggregated_probabilities[f_n2] += total_probability
            
    aggregated_probabilities, total_probabilities_test = get_total_probabilities(action_occurrences_test, transition_probabilities, average_occurrences)
    
    sorted_total_probabilities_test = sorted(total_probabilities_test, key=lambda x: x[0])
    sorted_aggregated_probabilities = sorted(aggregated_probabilities.items(), key=lambda x: x[1])

    with open('sorted_transitions.txt', 'w') as file:
        for probability, actions, durations, filn in sorted_total_probabilities_test:
            line = f"Transition in {filn} from {num_action_mapping[actions[0]]} to {num_action_mapping[actions[1]]}, Durations: {durations}, Total Probability: {probability}\n"
            file.write(line)
        file.write("\nAggregated Probabilities per Video Segment:\n")
        for filn, aggregated_probability in sorted_aggregated_probabilities:
            file.write(f"{filn}: Total Aggregated Probability = {aggregated_probability}\n")
