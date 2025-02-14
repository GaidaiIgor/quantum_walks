import os.path
import pickle
import random
from functools import partial
from itertools import permutations
from multiprocessing import Pool
import qiskit
import numpy as np
import pandas as pd
from qiskit import transpile, QuantumCircuit
import qiskit.qasm3
from qiskit.quantum_info import random_statevector, Statevector
from tqdm import tqdm
import networkx as nx
import matplotlib as plt
from qclib.gates.ldmcu import Ldmcu
from qiskit.circuit.library import UGate, UnitaryGate
from qiskit.circuit.library import RZGate, RXGate, PhaseGate

from src.quantum_walks import (PathFinder, PathFinderLinear, PathFinderSHP, PathFinderMST, 
    PathFinderRandom, PathFinderGrayCode,  
    PathFinderFromPairs, PathFinderMHSNonlinear,
    PathFinderMHSLinear, hamming_dist
    )
from src.validation import execute_circuit, get_state_vector, get_fidelity
from src.walks_gates_conversion import PathConverter

from networkx import Graph
from qclib.state_preparation.merge import MergeInitialize
# from src.gleinig import MergeInitialize #used for printing out the path

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
        path = path_finder.get_path_leafsm(target_state)
        # print(path)
        # print(target_state)
        circuit = PathConverter.convert_path_to_circuit_pframes_backwards_leafsm(path, reduce_controls, remove_leading_cx, add_barriers)
        # circuit = PathConverter.convert_path_to_circuit(path, reduce_controls, remove_leading_cx, add_barriers)
        # print(target_state)
        # print(circuit)
    elif method == "mhs_walks":
        path = path_finder.get_path_from_pairs_leafsm(target_state)
        # print(path)
        # add_barriers=True
        circuit = PathConverter.convert_path_to_circuit_pframes_backwards_leafsm(path, reduce_controls, remove_leading_cx, add_barriers)
        # circuit.draw(filename="circoriginal_walks.jpg", output="mpl")
        # print(PathConverter.convert_path_to_circuit_pframes_backwards(path, reduce_controls, remove_leading_cx, add_barriers))
    elif method=="merging_states":
        merger=MergeInitialize(target_state)
        circuit=merger._define_initialize()
        # print(circuit)
        # circuit.draw(fold=-1, filename="circoriginal_mergingsts.jpg", output="mpl")

    else:
        raise Exception("Unknown method")
    # circuit.draw(output="mpl", filename="circuit_example.jpeg")
    circuit_transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
    cx_count = circuit_transpiled.count_ops().get("cx", 0)
    # circuit_transpiled.draw(output="mpl", filename="circuit_transpiled_example.jpeg")
    # print(path)

    # if method=="mhs_walks":
    #     circuit_transpiled.draw(fold=-1, filename="circtranspiled_walks.jpg", output="mpl")
    # else:
    #     circuit_transpiled.draw(fold=-1, filename="cirtranspiled_mergingsts.jpg", output="mpl")

    # print(circuit_transpiled)
    # print("cx count: ", cx_count)
    # print(circuit.count_ops())
    # print(circuit_transpiled.count_ops())
    # circuit_transpiled.draw(fold=-1, filename="circtranspiled_gleinig.jpg", output="mpl")

    if check_fidelity:
        output_state_vector = execute_circuit(circuit_transpiled)
        # print("output vector ", output_state_vector)
        target_state_vector = get_state_vector(target_state)
        fidelity = get_fidelity(output_state_vector, target_state_vector)
        # print("fidelity ", fidelity)
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
    # print("min cx ", prepare_state_brute(init_state, len(init_state.keys())))
    num_qubits_all=np.array(list(range(5,12)))
    file_idxs=None
    # file_idxs=[4,5]
    num_amplitudes_all=num_qubits_all
    # path_finder=PathFinderLinear()
    path_finder=PathFinderMHSLinear()
    # path_finder=PathFinderMHSNonlinear()
    # path_finder=PathFinderSHP()
    # path_finder=PathFinderMST()
    # path_finder=PathFinderRandom()
    # method="walks"
    # method="qiskit"
    method="mhs_walks"
    # method="merging_states"
    # out_col_name="mhs_linear"
    # out_col_name="shp_reduced"
    # out_col_name="mst_reduced"
    # out_col_name="random_reduced"
    # out_col_name="random"
    # out_col_name="qiskit"
    out_col_name="mhs_nonlinear" 
    # out_col_name="mhs_nonlinear_not_reduced" 
    # out_col_name="linear_reduced"
    # out_col_name="merging_states"
    #qiskit, linear, linear_reduced, shp_reduced, mst_reduced, shp, mst, random, random_reduced
    #graycode_reudced, gleinig, mhs_linear, mhs_nonlinear

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
        if file_idxs:
            state_list=state_list[file_idxs[0]:file_idxs[1]:]
        results = []
        # state_list=[init_state]
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)
        # print(results)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df[out_col_name] = results
        # df.to_csv(cx_counts_file_path, index=False)
        print(f"Avg CX: {np.mean(df[out_col_name])}\n")


def run_bruteforce_order_state(num_workers, algo_type, num_qubits_all, file_idxs):
    '''Brute force search.'''
    out_col_name = "brute_force_linear"
    # num_qubits_all = np.array(list(range(5, 6)))
    # num_workers = 1
    num_amplitudes_all = num_qubits_all


    results = []
    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        if algo_type=="linear":
            process_func = partial(prepare_state_brute, num_amplitudes=num_amplitudes)
        elif algo_type=="star":
            process_func = partial(prepare_state_brute_star, num_amplitudes=num_amplitudes)
        else:
            assert False, "unkown exhaustive"
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)
        results = []
        # state_list=state_list[0:1:]
        state_list=state_list[file_idxs[0]:file_idxs[1]:]
        if num_workers == 1:
            for result in tqdm(map(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                results.append(result)
        else:
            with Pool(num_workers) as pool:
                for result in tqdm(pool.imap(process_func, state_list), total=len(state_list), smoothing=0, ascii=' █'):
                    results.append(result)

        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        # df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        # df[out_col_name] = results
        # # df.to_csv(cx_counts_file_path, index=False)
        # print(f"Avg CX: {np.mean(df[out_col_name])}\n")

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
    best_cx=None
    best_path=None
    for perm in all_permutations:
        path_finder = PathFinderLinear(list(perm))
        cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                 reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                 add_barriers=add_barriers)
        results.append(cx_count)
        if not best_cx or best_cx>cx_count:
            best_cx=cx_count
            best_path=perm
    path_finder=PathFinderLinear(list(best_path))
    path = path_finder.get_path_leafsm(target_state)
    circuit = PathConverter.convert_path_to_circuit_pframes_backwards_leafsm(path, reduce_controls, remove_leading_cx, add_barriers)
    # print(f"Min CX: {np.min(results)}\n")
    best_path_basis=[list(target_state.keys())[b] for b in best_path]
    circuit.draw(output="mpl", fold=-1, filename=f"linear_exhaust_{[best_path_basis[0], best_path_basis[1]]}.jpg")

    print(f"best linear path: ", list(zip(best_path_basis[0:-1:], best_path_basis[1::])))
    print("best cx: ", best_cx)
    
    return np.min(results)

def prepare_state_brute_star(target_state: dict[str, complex], num_amplitudes: int) -> int:
    '''Brute force search.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    results = []
    best_cx=None
    best_path=None
    best_circ_transpiled=None
    best_circ=None
    all_basis_states=list(target_state.keys())
    best_graph=None
    for root in all_basis_states:
        # gleinig_merger = MergeInitialize(target_state)
        # root, _, _, _=gleinig_merger._select_strings(target_state)
        all_permutations = list(permutations(range(num_amplitudes-1)))
        remaining_basis=list(target_state.keys())
        remaining_basis.remove(root)
        all_pairs=[[root, z2] for z2 in remaining_basis]
        # print("all pairs ", all_pairs)

        for perm in all_permutations:
            graph = Graph()
            temp_pairs=[all_pairs[idx] for idx in perm]
            for pair in temp_pairs:
                val=sum([b1 != b2 for b1, b2 in zip(pair[0],pair[1])])
                graph.add_edge(pair[0], pair[1], weight=val)
            graph.graph["start"] = root
            path_finder=PathFinderLinear()
            path_finder.set_graph_attributes_from_pairs(graph, temp_pairs, target_state)
            segments=path_finder.get_path_segments_from_pairs_leafsm(graph, temp_pairs, target_state)
            # print(segments)
            circuit = PathConverter.convert_path_to_circuit_pframes_backwards_leafsm(segments, reduce_controls, remove_leading_cx, add_barriers)
            circuit_transpiled = transpile(circuit, basis_gates=basis_gates, optimization_level=optimization_level)
            cx_count = circuit_transpiled.count_ops().get("cx", 0)
            print("cx count ", cx_count)
            
            # cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
            #                          reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
            #                          add_barriers=add_barriers)
            results.append(cx_count)
            if not best_cx or best_cx>cx_count:
                best_cx=cx_count
                best_path=temp_pairs
                best_circ=circuit
                best_circ_transpiled=circuit_transpiled
                best_graph=graph
            # print(circuit)
    circuit.draw(output="mpl", fold=-1, filename=f"star_exhaust_{best_path[0]}.jpg")
    labels = nx.get_edge_attributes(graph,'weight')
    pos = nx.spring_layout(graph)
    nx.draw(best_graph, pos, with_labels=True, node_color="lightblue")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    # Save the figure to a file
    plt.pyplot.savefig(f"graph_star_{best_path[0]}.png")
    plt.pyplot.clf()
    # print(f"Min CX: {np.min(results)}\n")
    # best_path_basis=[list(target_state.keys())[b] for b in best_path]
    print(f"best linear path: ", best_path)
    print(f"best cx count: ", best_cx)
    print(f"best circ: ", best_circ)
    output_state_vector = execute_circuit(best_circ_transpiled)
    # print("output vector ", output_state_vector)
    target_state_vector = get_state_vector(target_state)
    # print("target vector ", target_state_vector)
    fidelity = get_fidelity(output_state_vector, target_state_vector)
    fidelity_tol = 1e-8
    assert abs(1 - fidelity) < fidelity_tol, f"Failed to prepare the state. Fidelity: {fidelity} Target: {target_state} Output: {output_state_vector}"

    return np.min(results)

def run_greedy_order_state(num_workers = 1, start_type="mhs"):
    '''Greedy linear.'''
    if start_type=="mhs":
        out_col_name = "greedy_insertion_mhs"
    else:
        out_col_name = "greedy_insertion_ordered"
    # out_col_name = "greedy_tree"
    num_qubits_all = np.array(list(range(5, 12)))
    num_amplitudes_all = num_qubits_all#2**(num_qubits_all-1)

    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        process_func = partial(prepare_state_greedy_insertion, start_type=start_type)
        print(f"Num qubits: {num_qubits}; num amplitudes: {num_amplitudes}")
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        states_file_path = os.path.join(data_folder, "states.pkl")
        with open(states_file_path, "rb") as f:
            state_list = pickle.load(f)
        # state_list=state_list[4:5:]
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

def rename_column(old_name, new_name):
    num_qubits_all = np.array(list(range(5, 12)))
    num_amplitudes_all = num_qubits_all#2**(num_qubits_all-1)

    for num_qubits, num_amplitudes in zip(num_qubits_all, num_amplitudes_all):
        data_folder = f"data/qubits_{num_qubits}/m_{num_amplitudes}"
        cx_counts_file_path = os.path.join(data_folder, "cx_counts.csv")
        df = pd.read_csv(cx_counts_file_path) if os.path.isfile(cx_counts_file_path) else pd.DataFrame()
        df.rename(columns={old_name: new_name}, inplace=True)
        df.to_csv(cx_counts_file_path, index=False)

def prepare_state_greedy_tree(target_state: dict[str, complex], num_amplitudes: int) -> int:
    '''Uses a greedy method. Add the cheapest basis state at a time.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    basis_sts=list(target_state.keys())
    # start_idx=basis_sts.index(sorted(basis_sts, key=lambda x: int(x, 2))[0]) #get the smallest Hamming weight element.
    # start_idx=basis_sts.index(GleinigWalk.select_first_string(target_state))
    # path_finder = PathFinderLinear()
    # print(basis_sts)
    # basis_sts=_prepare_state_greedy_end(basis_sts)
    # basis_sts=list(reversed(GleinigWalk(target_state).get_gleinig_path()))
    basis_sts=list(reversed(GleinigWalk(target_state).get_gleinig_path_pframes()))
    # basis_sts=GleinigWalk(target_state).get_gleinig_path_pframes()
    # basis_sts=PathFinderHammingWeight().sort_bases_by_hamming_weight(basis_sts)
    path=[]
    path.append([basis_sts[0], basis_sts[1]])

    for z1 in basis_sts[2::]:
        partial_cx=None
        next_pair=None
        for origin in list(dict.fromkeys([elem for block in path for elem in block])):
            # print("current ", path)
            # convert each bit string to list of zeros and ones.
            # for temp_next_pair in [[origin, z1], [z1, origin]]:
            temp_next_pair=[origin, z1]
            temp_path=deepcopy(path)
            temp_path.append(temp_next_pair)
            # temp_perm=[basis_sts.index(z1) for z1 in temp_path]

            # fake state for getting the cnot count of intermediate trees.
            temp_visited=list(dict.fromkeys([elem for block in temp_path for elem in block]))
            # print(temp_path)
            # print(temp_visited)
            coeff=1/np.sqrt(len(temp_visited))
            fake_target={k:coeff for k in temp_visited}
            # print(fake_target)
            path_finder=PathFinderFromPairs(temp_path)
            temp_cx_count= prepare_state(fake_target, method, path_finder, basis_gates, 1, False, 
                                reduce_controls=reduce_controls, remove_leading_cx=True,
                                add_barriers=add_barriers)
            
            if not partial_cx:
                partial_cx=temp_cx_count
                next_pair=temp_next_pair
            elif temp_cx_count<partial_cx:
                partial_cx=temp_cx_count
                next_pair=temp_next_pair
            
        # path.append(basis_sts.pop(next_basis_idx))
        path.append(next_pair)


    # print(basis_sts)
    # convert path list of indices. prepare actual state.
    # basis_sts=list(target_state.keys())
    # perm=[basis_sts.index(z1) for z1 in path]
    # print(perm)
    # print(path)
    path_finder = PathFinderFromPairs(path)
    cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                    reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    add_barriers=add_barriers)
    # print("#########################")
    

    return cx_count

def prepare_state_greedy_insertion(target_state: dict[str, complex], start_type: str) -> int:
    '''Uses a greedy method. Add the cheapest basis state at a time.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    basis_sts=list(target_state.keys())
    if start_type=="mhs":
        _, path_original= PathFinderMHSLinear().build_travel_graph(list(target_state.keys()))
        path_original=[elem for block in path_original for elem in block[:-1:]]
        path_original=list(dict.fromkeys(path_original).keys())
    else:
        path_original= deepcopy(basis_sts)
        # np.random.seed(0)
        # path_original=list(np.random.permutation(np.array(path_original)))
        path_original= sorted(path_original, key=lambda x: int(x, 2))
    path=[path_original[0]] # first node
    for z1 in path_original[1::]:
        partial_cx=None
        next_basis_idx=None
        for temp_idx in list(range(len(path)+1)):
            
            temp_path=deepcopy(path)
            temp_path.insert(temp_idx, z1)
            # temp_perm=[basis_sts.index(z1) for z1 in temp_path]

            temp_cx_count=estimate_cx_count(temp_path)
            
            if not partial_cx:
                partial_cx=temp_cx_count
                next_basis_idx=temp_idx
            elif temp_cx_count<partial_cx:
                partial_cx=temp_cx_count
                next_basis_idx=temp_idx
        # path.append(basis_sts.pop(next_basis_idx))
        path.insert(next_basis_idx, z1)

    basis_sts=list(target_state.keys())
    perm=[basis_sts.index(z1) for z1 in path]
    # print(perm)
    path_finder = PathFinderLinear(perm)
    cx_count1 = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
                                    reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    add_barriers=add_barriers)
    # if start_type=="mhs":
    #     path_finder = PathFinderMHSLinear()
    #     method="mhs_walks"
    # else:
    #     method="walks"
    #     path_finder = PathFinderLinear([basis_sts.index(elem) for elem in path_original])
    # cx_count2 = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
    #                                 reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
    #                                 add_barriers=add_barriers)
    # return min([cx_count1, cx_count2])
    return cx_count1

def estimate_cx_count(linear_path):
    # convert each bit string to list of zeros and ones.
    # print("starting.....")
    # print("initial path ", linear_path)
    def int_list(elem):
        return [int(char) for char in elem]
    path_mutable=list(zip(linear_path[:-1:], linear_path[1::]))[::-1]
    path_mutable=[[int_list(elem) for elem in block] for block in path_mutable]
    linear_path=linear_path[::-1] #reverse since we will construct the circuit backwards.
    visited_transformed = [[int(char) for char in st] for st in linear_path]
    # print("mutable path ", path_mutable)
    temp_cx_count=0
    if len(linear_path)<=2:
        return 0
    for idx in list(range(len(linear_path)-1)):
        segment=path_mutable[idx]
        # print("path mutable ", path_mutable)
        origin = deepcopy(segment[0])
        destination = deepcopy(segment[1])
        diff_inds = list(np.where(np.array(origin) != np.array(destination))[0])

        # print("initial visited: ", visited)
        # print("destination: ", destination)
        # print("origin ", origin)
        visited_transformed=[elem for elem in visited_transformed if elem!=destination]
        # visited_transformed = deepcopy(visited)
        # visited_transformed = [elem for elem in visited_transformed if elem!=destination]
        # print("popped destination transformed: ", visited_transformed)
        # print("z1 z2 diffs ", diff_inds)
        # print("visited ", visited)
        origin_ind=visited_transformed.index(origin)
        if len(diff_inds)==1:
            interaction_ind=diff_inds[0]
        else:
            interaction_ind=PathConverter.get_good_interaction_idx(diff_inds, visited_transformed, origin_ind, segment[1])
        # interaction_ind = diff_inds[0]
        # print("diff inds ", diff_inds)
        # print("interaction idx ", interaction_ind)
        diff_inds.remove(interaction_ind)

        # update the set of visited
        for ind in diff_inds:
            PathConverter.update_visited(visited_transformed, interaction_ind, ind)
            # update the mutable path.
            for idx, segment in enumerate(path_mutable):
                # print(f"initial {z1} {z2}")
                PathConverter.update_visited(segment, interaction_ind, ind)
                # segment=[z1,z2]
                # path_mutable[idx]=segment
                # print(f"attached {segment.labels[0]} {segment.labels[1]}")

        # print("updating visited.")
        temp_cx_count+=len(diff_inds)
        # origin_ind = visited.index(origin)
        # print("origin index ", origin_ind)
        control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
        # visited=visited_transformed
            
        # if num_controls % 2==0:
        #     temp_cx_count+=20*num_controls-34
        # else:
        #     temp_cx_count+=20*num_controls-38
        temp_cx_count+=20*len(control_indices)-38
    return temp_cx_count

def _prepare_state_greedy_end(input_basis_sts: list[str]) -> int:
    '''Uses a greedy method. Add the cheapest basis state at a time. Helper function for warm starting
    the greedy insertion.'''
    method = "walks"
    reduce_controls = True
    check_fidelity = True
    remove_leading_cx = True
    add_barriers = False
    optimization_level = 3
    basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # print("num amps: ", num_amplitudes)
    np.random.seed(0)
    # start_idx=np.random.randint(0, num_amplitudes-1)
    basis_sts=deepcopy(input_basis_sts)
    start_idx=basis_sts.index(sorted(basis_sts, key=lambda x: int(x, 2))[0]) #get the smallest Hamming weight element.
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
    return path

def prepare_state_greedy_end(target_state: dict[str, complex], num_amplitudes: int) -> int:
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
    path_finder = PathFinderLinear()
    while len(basis_sts)>1: # the last part of the path should prepare the full state.
        partial_cx=None
        next_basis_idx=None
        for idx, z1 in enumerate(basis_sts):
            coeff=1/np.sqrt(len(path)+1)
            temp_target={k:coeff for k in path+[z1]} #construct fake state (normalized of the partial path).
            # print(temp_target)

            temp_cx_count = prepare_state(temp_target, method, path_finder, basis_gates, optimization_level, False, 
                                    reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
                                    add_barriers=add_barriers) #cost of fake state.
            #cx cost of fake state. cx conjugation + num_controls.
            # convert each bit string to list of zeros and ones.
            # visited=[[int(char) for char in elem] for elem in path]
            # origin = [int(char) for char in path[-1]]
            # destination = [int(char) for char in z1]
            # diff_inds = np.where(np.array(destination) != np.array(origin))[0]
            # interaction_ind = diff_inds[0]
            # visited_transformed=deepcopy(visited)
            # for ind in diff_inds[1::]:
            #     PathConverter.update_visited(visited_transformed, interaction_ind, ind)
            # origin_ind=visited.index(origin)
            # temp_cx_count=len(diff_inds[1::])*2
            # temp_cx_count+=40*len(PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind))
            
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

# def prepare_state_greedy_pframe(target_state: dict[str, complex], num_amplitudes: int) -> int:
#     '''Uses a greedy method. Add the cheapest basis state at a time. 
#     Doesn't update the Pauli frame (don't use without the proper path converter.).'''
#     method = "walks"
#     reduce_controls = True
#     check_fidelity = True
#     remove_leading_cx = True
#     add_barriers = True
#     optimization_level = 0
#     basis_gates = ["rx", "ry", "rz", "h", "cx"]
#     # print("num amps: ", num_amplitudes)
#     np.random.seed(0)
#     start_idx=np.random.randint(0, num_amplitudes-1)
#     basis_sts=list(target_state.keys())
#     basis_sts_original=deepcopy(basis_sts)
#     # start_idx=basis_sts.index(sorted(basis_sts, key=lambda x: int(x, 2))[0]) #get the smallest Hamming weight element.
#     # start_idx=basis_sts.index(GleinigWalk.select_first_string(target_state))
#     path_original=[basis_sts.pop(start_idx)]
#     path=[basis_sts_original.pop(start_idx)]
#     # path_finder = PathFinderLinear()
#     while len(basis_sts)>1: # the last part of the path should prepare the full state.
#         partial_cx=None
#         next_basis_idx=None
#         next_interaction_ind=None
#         next_diff_inds=None
#         iter_basis_sts=enumerate(deepcopy(basis_sts))
#         for idx, _ in iter_basis_sts:
#             z1=basis_sts[idx]
#             # coeff=1/np.sqrt(len(path)+1)
#             # temp_target={k:coeff for k in path+[z1]} #construct fake state (normalized of the partial path).
#             # print(temp_target)

#             # temp_cx_count = prepare_state(temp_target, method, path_finder, basis_gates, optimization_level, False, 
#                                     # reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
#                                     # add_barriers=add_barriers) #cost of fake state.
#             #cx cost of fake state. cx conjugation + num_controls.
#             # convert each bit string to list of zeros and ones.
#             visited=[[int(char) for char in elem] for elem in path]
#             origin = [int(char) for char in path[-1]]
#             destination = [int(char) for char in z1]
#             diff_inds = np.where(np.array(destination) != np.array(origin))[0]
#             interaction_ind = diff_inds[0]
#             visited_transformed=deepcopy(visited)
#             for ind in diff_inds[1::]:
#                 PathConverter.update_visited(visited_transformed, interaction_ind, ind)
#             origin_ind=visited.index(origin)
#             temp_cx_count=len(diff_inds[1::])
#             temp_cx_count+=40*len(PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind))
            
#             if not partial_cx or temp_cx_count<partial_cx:
#                 partial_cx=temp_cx_count
#                 next_basis_idx=idx
#                 next_interaction_ind=interaction_ind
#                 next_diff_inds=diff_inds

#         path.append(basis_sts.pop(next_basis_idx))
#         #update basis
#         # print("basis before: ", basis_sts)
#         # print(next_diff_inds)
#         temp_basis_sts=[[int(char) for char in elem] for elem in basis_sts]
#         temp_path=[[int(char) for char in elem] for elem in path]
#         for ind in next_diff_inds[1::]:
#             PathConverter.update_visited(temp_basis_sts, next_interaction_ind, ind)
#             PathConverter.update_visited(temp_path, next_interaction_ind, ind)
#         temp_basis_sts=[[str(char) for char in elem] for elem in temp_basis_sts]
#         temp_path=[[str(char) for char in elem] for elem in temp_path]
#         basis_sts=["".join(elem) for elem in temp_basis_sts]
#         path=temp_path
#         # print("basis after: ", basis_sts)
#         path_original.append(basis_sts_original.pop(next_basis_idx))
#     path=path+basis_sts # attach the last remaining basis element.
#     path_original=path_original+basis_sts_original
#     # convert path list of indices. prepare actual state.
#     basis_sts=list(target_state.keys())
#     perm=[basis_sts.index(z1) for z1 in path_original]
#     # print(perm)
#     path_finder = PathFinderLinear(perm)
#     cx_count = prepare_state(target_state, method, path_finder, basis_gates, optimization_level, check_fidelity, 
#                                     reduce_controls=reduce_controls, remove_leading_cx=remove_leading_cx,
#                                     add_barriers=add_barriers)

#     return cx_count

def combined_method(m1, m2, out_col_name, num_qubits_all, num_amplitudes_all):
    '''For greedy combined.'''
    assert out_col_name in ["greedy_insertion_mhs_combined", "greedy_insertion_ordered_combined"]
    for n, m in zip(num_qubits_all, num_amplitudes_all):
        data_path = f"data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        column_dat1=df[m1]
        column_dat2=df[m2]
        new_column=[pair[0] if pair[0]<pair[1] else pair[1] for pair in list(zip(column_dat1, column_dat2))]
        df[out_col_name]=new_column
        # print(df)
        df.to_csv(data_path, index=False)
        # print(new_column)

if __name__ == "__main__":
    # generate_states()
    # merge_state_files()
    # init_state={"1000": 1/2, "0110": 1/2, "1011": 1/2, "0101": 1/2}
    amp=1/np.sqrt(5)
    # init_state={"101010": amp, "101001": amp, "110101": amp, "011100": amp, "010110": amp, "000011": amp}
    init_state={"00001": amp, "00110": amp, "00111": amp, "01001": amp, "01011": amp}

    # # print("min cx ", prepare_state_brute(init_state, len(init_state.keys())))
    # num_qubits_all=np.array(list(range(6,7)))
    # file_idxs=[0,1]
    # num_amplitudes_all=num_qubits_all

    # run_bruteforce_order_state(1, "linear", qubits, file_indices)
    # run_bruteforce_order_state(1, "star", qubits, file_indices)
    # path_finder=PathFinderMHSLinear()
    # method="mhs_walks"
    # out_col_name="mhs_linear" #qiskit, linear, linear_reduced, shp_reduced, mst_reduced, shp, mst, random, random_reduced
    #graycode_reudced, gleinig, mhs_linear, mhs_nonlinear

    run_prepare_state()
    # create example circ.
    # reduce_controls = True
    # check_fidelity = True
    # remove_leading_cx = True
    # add_barriers = True
    # optimization_level = 3
    # basis_gates = ["rx", "ry", "rz", "h", "cx"]
    # method="mhs_walks"
    # path_finder=PathFinderMHSNonlinear()
    # prepare_state(init_state, method, path_finder,  basis_gates, optimization_level, check_fidelity,
    #               reduce_controls, remove_leading_cx, add_barriers)
    # run_greedy_order_state(num_workers=6, start_type="mhs")
    # num_qubits_all=np.array(list(range(5,12)))
    # num_amplitudes_all=num_qubits_all
    # combined_method("greedy_insertion_mhs", "mhs_linear", "greedy_insertion_mhs_combined", num_amplitudes_all, num_amplitudes_all)
    # rename_column("greedy_insertion_mhs", "greedy_insertion_mhs_combined")
    # method="merging_states"
    # out_col_name=method
    # run_prepare_state(None, num_qubits_all, file_idxs, num_amplitudes_all, path_finder, out_col_name, method)


    # run_greedy_order_state(num_workers=1)
    # run_bruteforce_order_state(num_workers=6)