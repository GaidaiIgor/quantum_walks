import os.path
import pickle
import random
from functools import partial
from itertools import permutations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import random_statevector, Statevector
from tqdm import tqdm

from src.quantum_walks import (PathFinder, PathFinderLinear, PathFinderSHP, PathFinderMST, 
    PathFinderRandom, PathFinderGrayCode, GleinigPathFinder, BinaryTreePathFinder, GleinigWalk,
    PathFinderHammingWeight
    )
from src.validation import execute_circuit, get_state_vector, get_fidelity
from src.walks_gates_conversion import PathConverter

from networkx import Graph
from qclib.state_preparation.merge import MergeInitialize
from copy import deepcopy

def prepare_state(target_state: dict[str, complex], method: str, path_finder: PathFinder, basis_gates: list[str], optimization_level: int, check_fidelity: bool,
                  reduce_controls: bool, remove_leading_cx: bool, add_barriers: bool, fidelity_tol: float = 1e-8) -> int:
    if method == "qiskit":
        target_state_vector = get_state_vector(target_state)
        num_qubits = len(next(iter(target_state.keys())))
        circuit = QuantumCircuit(num_qubits)
        circuit.prepare_state(target_state_vector)
    elif method == "walks":
        # print("target st: ", target_state)
        path = path_finder.get_path(target_state)
        # print(path)
        circuit = PathConverter.convert_path_to_circuit(path, reduce_controls, remove_leading_cx, add_barriers)
    elif method=="gleinig":
        merger=MergeInitialize(target_state)
        circuit=merger._define_initialize()
    # elif method=="gleinig_qwalk":
    #     path = path_finder.get_path(target_state)
    #     circuit = PathConverter.convert_path_to_circuit(path, reduce_controls, remove_leading_cx, add_barriers)
    # elif method=="binary_tree":
    #     path = path_finder.get_path(target_state)
    #     circuit = PathConverter.convert_path_to_circuit(path, reduce_controls, remove_leading_cx, add_barriers)
    else:
        raise Exception("Unknown method")
    # print(circuit)
    circuit_transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    cx_count = circuit_transpiled.count_ops().get("cx", 0)

    if check_fidelity:
        output_state_vector = execute_circuit(circuit_transpiled)
        target_state_vector = get_state_vector(target_state)
        fidelity = get_fidelity(output_state_vector, target_state_vector)
        assert abs(1 - fidelity) < fidelity_tol, f"Failed to prepare the state. Fidelity: {fidelity} Target: {target_state} Output: {output_state_vector}"

    return cx_count


def generate_states():
    num_qubits = np.array(list(range(4, 12)))
    num_amplitudes = num_qubits ** 2
    num_states = 1000

    for n, m in zip(num_qubits, num_amplitudes):
        out_path = f"data/qubits_{n}/m_{m}/states.pkl"
        all_inds = list(range(2 ** n))
        states = []
        for i in range(num_states):
            state_vector = random_statevector(len(all_inds)).data
            zero_inds = random.sample(all_inds, len(all_inds) - m)
            state_vector[zero_inds] = 0
            state_vector /= sum(abs(amplitude) ** 2 for amplitude in state_vector) ** 0.5
            state_dict = Statevector(state_vector).to_dict()
            states.append(state_dict)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


def merge_state_files():
    num_qubits = np.array(list(range(3, 12)))
    num_amplitudes = 2 ** (num_qubits - 1)
    merged = {}
    for n, m in zip(num_qubits, num_amplitudes):
        file_path = f"data/qubits_{n}/m_{m}/states.pkl"
        with open(file_path, "rb") as f:
            state_list = pickle.load(f)
        merged[f"qubits_{n}_amplitudes_{m}"] = state_list
    with open("states_merged.pkl", "wb") as f:
        pickle.dump(merged, f)


def run_prepare_state():
    # method = "qiskit"
    # method = "gleinig"
    method = "walks"
    # path_finder = PathFinderRandom()
    # path_finder = PathFinderLinear()
    # path_finder = PathFinderGrayCode()
    # path_frinder = PathFinderSHP()
    # path_finder = GleinigPathFinder()
    # path_finder = BinaryTreePathFinder()
    path_finder=PathFinderHammingWeight()
    # path_finder = PathFinderMST()
    # num_qubits_all = np.array([5])
    num_qubits_all = np.array(list(range(5, 12)))
    num_amplitudes_all = num_qubits_all # possible values are [num_qubits_all, num_qubits_all**2, 2**(num_qubits_all-1)]
    # out_col_name = "qiskit"
    # out_col_name = "gleinig_qwalk"
    out_col_name = "hamming_weight_qwalk"
    num_workers = 6
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    process_func = partial(prepare_state, method=method, path_finder=path_finder, basis_gates=basis_gates, optimization_level=optimization_level, check_fidelity=check_fidelity,
                           reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx, add_barriers=add_barriers)

    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)
        # state_list=state_list[0:1:]
        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        df.to_csv(cx_counts_file_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")

# def bruteforce_orders():
#     method = "walks"
#     num_qubits_all = 5
#     num_amplitudes_all = num_qubits_all
#     reduce_controls = True
#     check_fidelity = True
#     optimization_level = 3
#     basis_gates = ["rx", "ry", "rz", "h", "cx"]

#     states_file_path = f"data/qubits_{num_qubits_all}/m_{num_amplitudes_all}/states.pkl"
#     with open(states_file_path, "rb") as f:
#         state_list = pickle.load(f)

#     # path_finder = PathFinderLinear([0, 4, 1, 2, 3])
#     # path_finder = PathFinderLinear([0, 2, 4, 1, 3])
#     path_finder = PathFinderGrayCode()
#     cx_count = prepare_state(state_list[1], method, path_finder, basis_gates, optimization_level, check_fidelity, reduce_controls=reduce_controls)

#     all_permutations = list(permutations(range(num_amplitudes_all), num_amplitudes_all))
#     results = []
#     for perm in all_permutations:
#         path_finder = PathFinderLinear(list(perm))
#         cx_count = prepare_state(state_list[1], method, path_finder, basis_gates, optimization_level, check_fidelity, reduce_controls=reduce_controls)
#         results.append(cx_count)

#     print(f"Min CX: {np.min(results)}\n")


def run_bruteforce_order_state(num_workers=1):
    '''Brute force search.'''
    out_col_name = "brute_force_linear"
    num_qubits_all = np.array(list(range(5, 12)))
    # num_workers = 1
    num_amplitudes_all = num_qubits_all


    results = []
    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        process_func = partial(prepare_state_brute, num_amplitudes=num_amplitudes)
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)
        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        df.to_csv(cx_counts_file_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")

def prepare_state_brute(target_state: dict[str, complex], num_amplitudes: int) -> int:
    '''Brute force search.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    all_permutations = list(permutations(range(num_amplitudes)))
    results = []
    for perm in all_permutations:
        path_finder = PathFinderLinear(list(perm))
        cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                 reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                 add_barriers=add_barriers)
        results.append(cx_count)

    # print(f"Min CX: {np.min(results)}\n")

    return np.min(results)

def run_greedy_order_state():
    '''Greedy linear.'''
    out_col_name = "greedy_linear"
    num_qubits_all = np.array(list(range(5, 12)))
    num_workers = 6
    num_amplitudes_all = num_qubits_all#2**(num_qubits_all-1)

    results = []
    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        process_func = partial(prepare_state_greedy, num_amplitudes=num_amplitudes)
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)
        results = []
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        # df.to_csv(cx_counts_file_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")

def prepare_state_greedy(target_state: dict[str, complex], num_amplitudes: int) -> int:
    '''Uses a greedy method. Add the cheapest basis state at a time.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    np.random.seed(0)
    start_idx=np.random.randint(0, num_amplitudes-1)
    basis_sts=list(target_state.keys())
    # start_idx=basis_sts.index(sorted(basis_sts, key=lambda x: int(x, 2))[0]) #get the smallest Hamming weight element.
    # start_idx=basis_sts.index(GleinigWalk.select_first_string(target_state))
    path=[basis_sts.pop(start_idx)]
    # path_finder = PathFinderLinear()
    while len(basis_sts)>1: # the last part of the path should prepare the full state.
        partial_cx=None
        next_basis_idx=None
        for idx, z1 in enumerate(basis_sts):
            # coeff=1/np.sqrt(len(path)+1)
            # temp_target={k:coeff for k in path+[z1]} #construct fake state (normalized of the partial path).
            # print(temp_target)

            # temp_cx_count = prepare_state(temp_target, method, path_finder, basis_gates, optimization_level, False, 
                                    # reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    # add_barriers=add_barriers) #cost of fake state.
            #cx cost of fake state. cx conjugation + num_controls.
            # convert each bit string to list of zeros and ones.
            visited=[[int(char) for char in elem] for elem in path]
            origin = [int(char) for char in path[-1]]
            destination = [int(char) for char in z1]
            diff_inds = np.where(np.array(destination) != np.array(origin))[0]
            interaction_ind = diff_inds[0]
            visited_transformed=deepcopy(visited)
            for ind in diff_inds[1::]:
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)
            origin_ind=visited.index(origin)
            temp_cx_count=len(diff_inds[1::])*2
            temp_cx_count+=40*len(PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind))
            
            if not partial_cx:
                partial_cx=temp_cx_count
                next_basis_idx=idx
            elif temp_cx_count<partial_cx:
                partial_cx=temp_cx_count
                next_basis_idx=idx
        path.append(basis_sts.pop(next_basis_idx))
    path=path+basis_sts # attach the last remaining basis element.
    # convert path list of indices. prepare actual state.
    basis_sts=list(target_state.keys())
    perm=[basis_sts.index(z1) for z1 in path]
    # print(perm)
    path_finder = PathFinderLinear(perm)
    cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                    reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    add_barriers=add_barriers)

    return cx_count

def prepare_state_greedy_pframe(target_state: dict[str, complex], num_amplitudes: int) -> int:
    '''Uses a greedy method. Add the cheapest basis state at a time. 
    Doesn't update the Pauli frame (don't use without the proper path converter.).'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    np.random.seed(0)
    start_idx=np.random.randint(0, num_amplitudes-1)
    basis_sts=list(target_state.keys())
    basis_sts_original=deepcopy(basis_sts)
    # start_idx=basis_sts.index(sorted(basis_sts, key=lambda x: int(x, 2))[0]) #get the smallest Hamming weight element.
    # start_idx=basis_sts.index(GleinigWalk.select_first_string(target_state))
    path_original=[basis_sts.pop(start_idx)]
    path=[basis_sts_original.pop(start_idx)]
    # path_finder = PathFinderLinear()
    while len(basis_sts)>1: # the last part of the path should prepare the full state.
        partial_cx=None
        next_basis_idx=None
        next_interaction_ind=None
        next_diff_inds=None
        iter_basis_sts=enumerate(deepcopy(basis_sts))
        for idx, _ in iter_basis_sts:
            z1=basis_sts[idx]
            # coeff=1/np.sqrt(len(path)+1)
            # temp_target={k:coeff for k in path+[z1]} #construct fake state (normalized of the partial path).
            # print(temp_target)

            # temp_cx_count = prepare_state(temp_target, method, path_finder, basis_gates, optimization_level, False, 
                                    # reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    # add_barriers=add_barriers) #cost of fake state.
            #cx cost of fake state. cx conjugation + num_controls.
            # convert each bit string to list of zeros and ones.
            visited=[[int(char) for char in elem] for elem in path]
            origin = [int(char) for char in path[-1]]
            destination = [int(char) for char in z1]
            diff_inds = np.where(np.array(destination) != np.array(origin))[0]
            interaction_ind = diff_inds[0]
            visited_transformed=deepcopy(visited)
            for ind in diff_inds[1::]:
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)
            origin_ind=visited.index(origin)
            temp_cx_count=len(diff_inds[1::])
            temp_cx_count+=40*len(PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind))
            
            if not partial_cx or temp_cx_count<partial_cx:
                partial_cx=temp_cx_count
                next_basis_idx=idx
                next_interaction_ind=interaction_ind
                next_diff_inds=diff_inds

        path.append(basis_sts.pop(next_basis_idx))
        #update basis
        # print("basis before: ", basis_sts)
        # print(next_diff_inds)
        temp_basis_sts=[[int(char) for char in elem] for elem in basis_sts]
        temp_path=[[int(char) for char in elem] for elem in path]
        for ind in next_diff_inds[1::]:
            PathConverter.update_visited(temp_basis_sts, next_interaction_ind, ind)
            PathConverter.update_visited(temp_path, next_interaction_ind, ind)
        temp_basis_sts=[[str(char) for char in elem] for elem in temp_basis_sts]
        temp_path=[[str(char) for char in elem] for elem in temp_path]
        basis_sts=["".join(elem) for elem in temp_basis_sts]
        path=temp_path
        # print("basis after: ", basis_sts)
        path_original.append(basis_sts_original.pop(next_basis_idx))
    path=path+basis_sts # attach the last remaining basis element.
    path_original=path_original+basis_sts_original
    # convert path list of indices. prepare actual state.
    basis_sts=list(target_state.keys())
    perm=[basis_sts.index(z1) for z1 in path_original]
    # print(perm)
    path_finder = PathFinderLinear(perm)
    cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                    reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    add_barriers=add_barriers)

    return cx_count

if __name__ == "__main__":
    # generate_states()
    # merge_state_files()
    # run_prepare_state()
    # run_greedy_order_state()
    run_bruteforce_order_state(num_workers=1)
