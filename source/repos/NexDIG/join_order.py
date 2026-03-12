import itertools
import json
import math
import os
import time

import dimod
import hybrid
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp


def get_log_values(coeff, num_decimal_pos, use_rounding=True):
    if use_rounding:
        log_coeff = np.around(np.log10(coeff), num_decimal_pos)
    else:
        log_coeff = np.log10(coeff)
    return log_coeff.tolist()


def get_binary_slack_coeff(num_slack, precision):
    slack_coeff = []
    for i in range(num_slack):
        slack_coeff.append(pow(2, i))
    slack_coeff = [x * precision for x in slack_coeff]
    return slack_coeff


def get_binary_slack_variables_for_bound(model, bound, num_decimal_pos):
    precision = pow(0.1, num_decimal_pos)
    num_slack = int(math.floor(np.log2(bound / precision))) + 1
    slack = model.binary_var_list(num_slack)
    return slack, get_binary_slack_coeff(num_slack, precision)


def load_from_path(problem_path):
    data_file = os.path.abspath(problem_path)
    if os.path.exists(data_file):
        with open(data_file) as file:
            data = json.load(file)
            return data


def parse_selectivities(sel):
    """Extract predicates and corresponding selectivities"""
    pred = []
    pred_sel = []
    for (i, j) in itertools.combinations(range(len(sel)), 2):  # Generate Non-duplicate Pairs
        if sel[i][j] != 1:
            pred.append((i, j))
            pred_sel.append(sel[i][j])
    return pred, pred_sel


def format_loaded_pred(pred):
    form_pred = []
    for p in pred:
        form_pred.append(tuple(p))
    return form_pred


### TODO: Currently it is a fixed problem generator
def get_join_ordering_problem():
    card = [1, 1, 28889, 1380040.0, 2528310.0]
    sel = [[1.0, 1.0, 0.991969261656686, 1.0, 1.0], [1.0, 1.0, 1.0, 0.00018115416944436394, 1.0], [0.991969261656686, 1.0, 1.0, 1.5716373635702107e-06, 3.95521118850141e-07],
           [1.0, 0.00018115416944436394, 1.5716373635702107e-06, 1.0, 3.95521118850141e-07], [1.0, 1.0, 3.95521118850141e-07, 3.95521118850141e-07, 1.0]]
    '''card = [1, 1, 28889, 1380040.0]
    sel = [[1.0, 1.0, 0.991969261656686, 1.0], [0.991969261656686, 1.0, 1.0, 1.5716373635702107e-06],
           [1.0, 0.00018115416944436394, 1.0, 3.95521118850141e-07], [1.0, 1.0, 3.95521118850141e-07, 3.95521118850141e-07]]'''
    pred, pred_sel = parse_selectivities(sel)
    return card, pred, pred_sel


def generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, log_thres, num_decimal_pos, penalty_scaling=1,
                                           minimum_penalty_weight=20):
    # thres_penalty = [x / thres[0] for x in thres]
    # thres_penalty = [x / thres[len(thres)-1] for x in thres]

    card = get_log_values(card, num_decimal_pos)
    pred_sel = get_log_values(pred_sel, num_decimal_pos)
    # log_thres = get_log_values(thres, num_decimal_pos)

    '''print("Card:")
    print(card)
    print("Pred sel:")
    print(pred_sel)'''

    model = Model('docplex_model')

    num_relations = len(card)
    num_pred = len(pred_sel)
    num_joins = len(card) - 2

    v = model.binary_var_matrix(num_relations, num_joins)

    b = np.arange(2, num_joins + 2).tolist()

    # Incentivise that the right number of relations is present for every join (i.e., 2 for join 1, 3 for join 2, ...)
    H_A = model.sum((b[j] - model.sum(v[(t, j)] for t in range(num_relations))) ** 2 for j in range(num_joins))

    # Incentivise that, once joined, a relation is always part of subosequent joins
    H_B = model.sum(
        model.sum(v[(t, j - 1)] - v[(t, j - 1)] * v[(t, j)] for j in range(1, num_joins)) for t in range(num_relations))

    # Incentivise that a predicate is only applicable for a join if both associated relations are present
    pred_vars = model.binary_var_matrix(num_pred, num_joins)
    H_pred_a = model.sum(
        model.sum(pred_vars[(p, j)] - pred_vars[(p, j)] * v[(pred[p][0], j)] for p in range(num_pred)) for j in range(num_joins))
    H_pred_b = model.sum(
        model.sum(pred_vars[(p, j)] - pred_vars[(p, j)] * v[(pred[p][1], j)] for p in range(num_pred)) for j in range(num_joins))
    H_pred = H_pred_a + H_pred_b

    H_cost = 0
    penalty_weight = 0

    # Intermediate cardinality calculation
    for j in range(num_joins):
        # max_log_card = get_maximum_log_intermediate_outer_operand_cardinality(j, card)
        # penalty_weight = penalty_weight + pow(max_log_card - log_thres, 2)
        penalty_weight = penalty_weight + 1
        slack, slack_coeff = get_binary_slack_variables_for_bound(model, log_thres, num_decimal_pos)
        H_thres = (model.sum(slack_coeff[s] * slack[s] for s in range(len(slack))) - (
                model.sum(card[t] * v[(t, j)] for t in range(num_relations)) + model.sum(
            pred_sel[p] * pred_vars[(p, j)] for p in range(num_pred)))) ** 2
        H_cost = H_cost + H_thres

    #print("Vanilla penalty weight: " + str(penalty_weight))
    penalty_weight = penalty_weight * penalty_scaling
    if penalty_weight < minimum_penalty_weight:
        penalty_weight = minimum_penalty_weight

    H_valid = H_A + H_B + H_pred

    H = penalty_weight * H_valid + H_cost

    model.minimize(H)

    qubo = from_docplex_mp(model)

    return qubo, penalty_weight

def get_all_folders_in_target_directory_and_sorted(directory):
    """Sort in the real-number manners"""
    all_items = os.listdir(directory)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    sorted_folders = sorted(folders,
                            key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else float('inf'))
    return [os.path.join(directory, folder) for folder in sorted_folders]

def process_input(sql_folder_path):
    cardinalities_file_path = '{}/cardinalities.json'.format(sql_folder_path)
    selectivities_file_path = '{}/selectivities.json'.format(sql_folder_path)
    # Reading the JSON content from the files
    with open(cardinalities_file_path, 'r') as file:
        cardinalities_content = json.load(file)

    with open(selectivities_file_path, 'r') as file:
        selectivities_content = json.load(file)
    return cardinalities_content, selectivities_content



def generate_Fujitsu_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=1):
    ibmq_qubo, penalty_weight = generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, thres, num_decimal_pos, penalty_scaling=penalty_scaling)
    num_qubits = len(ibmq_qubo.objective.linear.to_array())
    #print("Number of IBMQ qubits: " + str(num_qubits))
    dwave_qubo = dimod.as_bqm(ibmq_qubo.objective.linear.to_array(), ibmq_qubo.objective.quadratic.to_array(), ibmq_qubo.objective.constant, dimod.BINARY)
    return dwave_qubo

def iterationNoSize(decomposer):
    return hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        decomposer()
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()

def iterationWithSize(decomposer, size):
    return hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        decomposer(size = size)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()

if __name__ == '__main__':
    card1, pred1, pred_sel1 = get_join_ordering_problem()
    cards = list()
    sels = list()
    preds = list()
    pred_sels = list()
    relative_path = "JOB/JOB"
    absolute_path_os = os.path.abspath(relative_path)

    folders = get_all_folders_in_target_directory_and_sorted(absolute_path_os)
    size = len(folders)
    
    for path in folders:
        card, sel = process_input(path)
        cards.append(card)
        pred, pred_sel = parse_selectivities(sel)
        preds.append(pred)
        pred_sels.append(pred_sel)
    
    bqms = [generate_Fujitsu_QUBO_for_left_deep_trees(cards[i], preds[i], pred_sels[i], 0.63, 2, penalty_scaling=2) for i in range(size)]
    #bqm2 = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1, 'ca': 1}, 0, dimod.SPIN)

    #bqm1 = generate_Fujitsu_QUBO_for_left_deep_trees(card1, pred1, pred_sel1, 0.63, 2, penalty_scaling=2)
    #bqm2 = bqms[0]
    
    # Define the workflow
    '''names = ["EnergyImpactDecomposer", "RandomSubproblemDecomposer", "ComponentDecomposer", "RoofDualityDecomposer"]
    decomposers_with_size = [hybrid.EnergyImpactDecomposer, hybrid.RandomSubproblemDecomposer]
    decomposers_no_size = [hybrid.ComponentDecomposer, hybrid.RoofDualityDecomposer]
    iterations_with_size = [iterationWithSize(dec, 2) for dec in decomposers_with_size]
    iterations_no_size = [iterationNoSize(dec) for dec in decomposers_no_size]'''
    #iteration1 = iterationWithSize(hybrid.EnergyImpactDecomposer, 2)

    '''iteration = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.AlternatingSizeDecomposer(size = 2, card=card1, pred=pred1, pred_sel=pred_sel1, traversal='diff_col')
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()

    iteration2 = hybrid.RacingBranches(
        hybrid.InterruptableTabuSampler(),
        hybrid.EnergyImpactDecomposer(size=2)
        | hybrid.QPUSubproblemAutoEmbeddingSampler()
        | hybrid.SplatComposer()
    ) | hybrid.ArgMin()

    workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
    current_time = time.time()
    init_state = hybrid.State.from_problem(bqm1)
    final_state1 = workflow.run(init_state).result()
    final_time = time.time() - current_time

    print(f":\nSolution: sample={final_state1.samples.first}\n\n")
    print(f"Time elapsed: {final_time}\n\n")

    workflow = hybrid.LoopUntilNoImprovement(iteration2, convergence=3)
    current_time = time.time()
    init_state = hybrid.State.from_problem(bqm1)
    final_state1 = workflow.run(init_state).result()
    final_time = time.time() - current_time

    print(f":\nSolution: sample={final_state1.samples.first}\n\n")
    print(f"Time elapsed: {final_time}\n\n")'''
    #workflows = [hybrid.LoopUntilNoImprovement(it, convergence = 3) for it in iterations_with_size]
    #workflows += [hybrid.LoopUntilNoImprovement(it, convergence = 3) for it in iterations_no_size]

    

    impact_energy_file = open('energy.txt', 'w')
    impact_time_file = open('time.txt', 'w')
    bfs_energy_file = open('bfs_energy.txt', 'w')
    bfs_time_file = open('bfs_time.txt', 'w')
    same_col_energy_file = open('same_energy.txt', 'w')
    same_col_time_file = open('same_time.txt', 'w')
    '''same_limited_energy = open('limited_e.txt', 'w')
    same_limited_time = open('limited_t.txt', 'w')
    same_limited_energy2 = open('limited_energy_e.txt', 'w')
    same_limited_time2 = open('limited_energy_t.txt', 'w')
    last_file_e = open('last_energy.txt', 'w')
    last_file_t = open('last_time.txt', 'w')'''

    for j in range(0, 113):
        total_time = 0
        total_energy = 0

        iteration = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=len(preds[j]))
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
        
        iteration2 = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=len(preds[j]), traversal='bfs')
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
        
        iteration3 = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.AlternatingSizeDecomposer(size=2, card=cards[j], pred=preds[j], pred_sel=pred_sels[j], traversal='same_col')
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
        
        iteration4 = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.AlternatingSizeDecomposer(size = 5, card=cards[j], pred=preds[j], pred_sel=pred_sels[j], traversal='same_col_limited')
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
        
        iteration5 = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.AlternatingSizeDecomposer(size = 5, card=cards[j], pred=preds[j], pred_sel=pred_sels[j], traversal='same_col_limited_energy')
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()
        
        iteration6 = hybrid.RacingBranches(
            hybrid.InterruptableTabuSampler(),
            hybrid.AlternatingSizeDecomposer(size=2, card=cards[j], pred=preds[j], pred_sel=pred_sels[j], traversal='same_col_both')
            | hybrid.QPUSubproblemAutoEmbeddingSampler()
            | hybrid.SplatComposer()
            ) | hybrid.ArgMin()

        print(f"q{j + 1}:\n")
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy eid: {total_energy / 10}\n")
        print(f"Average time elapsed eid: {total_time / 10}\n\n")
        impact_energy_file.write(f"{total_energy / 10}\n")
        impact_time_file.write(f"{total_time / 10}\n")

        total_time = 0
        total_energy = 0
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration2, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy eid: {total_energy / 10}\n")
        print(f"Average time elapsed eid: {total_time / 10}\n\n")
        bfs_energy_file.write(f"{total_energy / 10}\n")
        bfs_time_file.write(f"{total_time / 10}\n")

        total_time = 0
        total_energy = 0
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration3, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy sc: {total_energy / 10}\n")
        print(f"Average time elapsed sc: {total_time / 10}\n\n")
        same_col_energy_file.write(f"{total_energy / 10}\n")
        same_col_time_file.write(f"{total_time / 10}\n")

        '''total_time = 0
        total_energy = 0
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration4, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy dc: {total_energy / 10}\n")
        print(f"Average time elapsed dc: {total_time / 10}\n\n")
        same_limited_energy.write(f"{total_energy / 10}\n")
        same_limited_time.write(f"{total_time / 10}\n")

        total_time = 0
        total_energy = 0
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration5, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy dc: {total_energy / 10}\n")
        print(f"Average time elapsed dc: {total_time / 10}\n\n")
        same_limited_energy2.write(f"{total_energy / 10}\n")
        same_limited_time2.write(f"{total_time / 10}\n")

        total_time = 0
        total_energy = 0
        for i in range(10):

            workflow = hybrid.LoopUntilNoImprovement(iteration6, convergence=3)
            current_time = time.time()
            init_state = hybrid.State.from_problem(bqms[j])
            final_state1 = workflow.run(init_state).result()
            final_time = time.time() - current_time

            total_time += final_time
            total_energy += final_state1.samples.first.energy

            #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
            #print(f"Time elapsed: {final_time}\n\n")

        print(f"Average energy dc: {total_energy / 10}\n")
        print(f"Average time elapsed dc: {total_time / 10}\n\n")
        last_file_e.write(f"{total_energy / 10}\n")
        last_file_t.write(f"{total_time / 10}\n")'''

    impact_energy_file.close()
    impact_time_file.close()
    bfs_energy_file.close()
    bfs_time_file.close()
    same_col_energy_file.close()
    same_col_time_file.close()
    '''same_limited_time.close()
    same_limited_energy.close()
    same_limited_time2.close()
    same_limited_energy2.close()
    last_file_e.close()
    last_file_t.close()'''
    
    #hybrid.AlternatingSizeDecomposer(size = 2, num_rel = len(cards[j]), num_pred = len(preds[j]), traversal='same_col')

    #print(f":\nSolution: sample={final_state1.samples.first}\n\n")
    #print(f"Average energy: {total_energy}\n\n")
    #print(f"Average time elapsed: {final_time}\n\n")
    # Solve the problem
    '''for i in range(0, len(workflows)):
        current_time = time.time()
        init_state = hybrid.State.from_problem(bqm)
        final_state = workflows[i].run(init_state).result()
        final_time = time.time() - current_time

        for datum in final_state.samples.data(fields=['sample', 'energy']):   
            print(datum)
        
        print(f"{names[i]}:\n\n")
        print(f":\nSolution: sample={final_state.samples.first}\n")
        print(f"Time elapsed: {final_time}\n\n")'''