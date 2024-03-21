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

def load_breakfast_dataset():
    action_mapping = {}
    num_action_mapping = {}

    with open('/nfs/hpc/dgx2-6/data/gtea/mapping.txt', 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    prediction_dir = '/nfs/hpc/dgx2-6/data/result1/GTEA-Trained-S1-Tested-BF-S1/release'
    action_occurrences_test = action_occurrences_from_predictions_breakfast(prediction_dir, action_mapping)
    return action_occurrences_test


          
def compute_ood_percentage(sorted_aggregated_probabilities):
    import math
    total_files = len(sorted_aggregated_probabilities)

    thresholds = {
        "top_5%": math.ceil(total_files * 0.05),
        "top_10%": math.ceil(total_files * 0.1),
        "top_50%": math.ceil(total_files * 0.5),
        "top_95%": math.ceil(total_files * 0.95),
        
    }

    counts = {key: 0 for key in thresholds}

    for threshold_name, threshold_value in thresholds.items():
        top_files = sorted_aggregated_probabilities[:threshold_value]
        count_ood = sum(1 for file in top_files if file[0].startswith('P'))  # Change for ablation
        counts[threshold_name] = count_ood/threshold_value

    percentages = {key: (value)*100 for key, value in counts.items()}
    return percentages

def parse_prediction_gtea(prediction_dir, action_mapping):
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

def do_log(prob):
    if prob == 0.0:
        return 999
    else:
        return np.abs(np.log(prob))
    
def load_salads_dataset():
    action_mapping = {}
    num_action_mapping = {}

    with open('/nfs/hpc/dgx2-6/data/breakfast/mapping.txt', 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
    prediction_dir = '/nfs/hpc/dgx2-6/data/result1/Breakfast-Trained-S1-Tested-50salads-S1/release'
    action_occurrences_test = action_occurrences_from_predictions_breakfast(prediction_dir, action_mapping)
    return action_occurrences_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    label_dir = '/nfs/hpc/dgx2-6/data/gtea/labels'
    prediction_dir = '/nfs/hpc/dgx2-6/data/gtea/prediction_print'
    mapping_file = '/nfs/hpc/dgx2-6/data/gtea/mapping.txt'
    action_mapping = {}
    num_action_mapping = {}
  

    with open(mapping_file, 'r') as f:
        for line in f:
            number, action = line.strip().split()
            action_mapping[action] = int(number)
            num_action_mapping[int(number)] = action
            
    action_occurrences_train = []
    action_occurrences_test = []

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


    transition_probabilities, average_occurrences = build_transition_matrix(action_occurrences_train)
    for dura_a in [True, False]:
        for dura_b in [True, False]:
            aggregated_probabilities = defaultdict(float)
            action_occurrences_test = parse_prediction_gtea(prediction_dir, action_mapping)
            # action_occurrences_test_breakfast = load_breakfast_dataset()
            action_occurrences_test_salads = load_salads_dataset() # Change for ablation
            action_occurrences_test.extend(action_occurrences_test_salads)
            total_probabilities_test = []
            for i in range(len(action_occurrences_test) - 1):
                action_a, duration_a, f_n = action_occurrences_test[i]
                action_b, duration_b, f_n2 = action_occurrences_test[i + 1]
                if f_n == f_n2:
                    total_probability = compute_total_probability(action_a, duration_a, action_b, duration_b, transition_probabilities, average_occurrences)
                    total_probabilities_test.append((total_probability, (action_a, action_b), (duration_a, duration_b), f_n2))
                    aggregated_probabilities[f_n2] += do_log(total_probability)

            sorted_aggregated_probabilities = sorted(aggregated_probabilities.items(), key=lambda x: x[1], reverse=True)
            file_name_list = []
            percentages = compute_ood_percentage(sorted_aggregated_probabilities)

            with open('gtea_{}_{}.txt'.format(dura_a, dura_b), 'w') as file:
  
                file.write("\nAggregated Uncertainity per Video Segment:\n")
                file_name_list = []

                for filn, aggregated_probability in sorted_aggregated_probabilities:
                    file.write(f"{filn}: Total Aggregated Uncertainity = {aggregated_probability}\n")
                file.write("\nF@K%:\n")
                file.write(f"Percentage of OOD files in the top 5%: { percentages['top_5%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 10%: {percentages['top_10%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 50%: {percentages['top_50%']:.2f}% \n")
                file.write(f"Percentage of OOD files in the top 95%: {percentages['top_95%']:.2f}% \n")