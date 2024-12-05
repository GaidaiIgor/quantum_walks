""" Functions that generate quantum walks. """
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import graycode
import networkx as nx
import numpy as np
from networkx import Graph
from copy import deepcopy, copy
# from binarytree import Node
from itertools import permutations
import os
from networkx.algorithms import descendants
from pysat.examples.hitman import Hitman

# currdir=os.getcwd()
# print(currdir)
# os.chdir(currdir+"/src/")
# from src.walks_gates_conversion import PathConverter
# os.chdir(currdir)


@dataclass
class PathSegment:
    """
    Class that stores information about a particular segment in a state preparation path. Incorporates both phase and amplitude walks.
    :var labels: Initial and final basis state labels for this segment.
    :var phase_time: Time of the phase walk.
    :var amplitude_time: Time of the amplitude walk.
    """
    labels: list[str, str]
    phase_time: float
    amplitude_time: float
    interaction_index: int=None

@dataclass
class LeafPathSegment:
    """
    Class that stores information about a particular segment in a state preparation path. Incorporates both phase and amplitude walks.
    :var labels: Initial and final basis state labels for this segment.
    :var phase_time1: Time of the phase walk for origin.
    :var phase_time2: Time of the phase walk for destination.
    :var amplitude_time: Time of the amplitude walk.
    """
    labels: list[str, str]
    phase_time1: float
    phase_time2: float
    amplitude_time: float
    interaction_index: int=None


@dataclass
class PathFinder(ABC):
    """ Base class for implementations of particular traversal orders to prepare a given state. """

    @abstractmethod
    def build_travel_graph(self, bases: list[str]) -> Graph:
        """
        Builds a graph that describes connections between the bases during state preparation. Graph's "start" attribute has to be set to the starting basis of the path.
        :param bases: List of non-zero amplitudes in the target state.
        :return: Basis connectivity graph.
        """
        pass

    def set_graph_attributes(self, graph: Graph, target_state: dict[str, complex]):
        """
        Assigns necessary attributes of the travel graph.
        :param graph: Travel graph for state preparation.
        :param target_state: Target state for state preparation.
        """
        get_attributes = lambda label: {"current_phase": 1,
                                        "target_phase": target_state[label] / abs(target_state[label]),
                                        "current_prob": 0,
                                        "target_prob": abs(target_state[label]) ** 2}
        attributes = {key: get_attributes(key) for key in target_state.keys()}
        nx.set_node_attributes(graph, attributes)
        graph.nodes[graph.graph["start"]]["current_prob"] = 1
        bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))[::-1]
        # print(bfs_edges)

        for edge in bfs_edges:
            graph.nodes[edge[0]]["target_prob"] += graph.nodes[edge[1]]["target_prob"]

    def set_graph_attributes_from_pairs(self, graph: Graph, all_edges, target_state: dict[str, complex]):
        """
        Assigns necessary attributes of the travel graph.
        :param graph: Travel graph for state preparation.
        :param target_state: Target state for state preparation.
        """
        get_attributes = lambda label: {"current_phase": 1,
                                        "target_phase": target_state[label] / abs(target_state[label]),
                                        "current_prob": 0,
                                        "target_prob": abs(target_state[label]) ** 2}
        attributes = {key: get_attributes(key) for key in target_state.keys()}
        nx.set_node_attributes(graph, attributes)
        graph.nodes[graph.graph["start"]]["current_prob"] = 1
        # bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))[::-1]
        bfs_edges=all_edges[::-1] #sum from bottom up

        for edge in bfs_edges:
            graph.nodes[edge[0]]["target_prob"] += graph.nodes[edge[1]]["target_prob"]

    @staticmethod
    def get_hamming_generator(target_str: str) -> Iterator[str]:
        """
        Generates bit strings in the order of the Hamming distance to the target bit string.
        :param target_str: Target bit string.
        :return: Bit string generator.
        """
        all_inds = list(range(len(target_str)))
        for current_distance in range(1, len(target_str) + 1):
            for combo in combinations(all_inds, current_distance):
                # print(f"combo: {combo}")
                next_str = np.array(list(map(int, target_str)))
                # print(f"next string combo: {next_str[combo[0]]}")
                for elem in combo:
                    next_str[elem] = 1 - next_str[elem]
                # next_str[combo[0]] = 1 - next_str[combo[0]]
                next_str = "".join(map(str, next_str))
                yield next_str
        raise StopIteration

    @staticmethod
    def find_closest_zero_amplitude(target_state: dict[str, complex], target_basis: str) -> str | None:
        """
        Finds a basis state that is not a part of the target state.
        :param target_state: Dict of basis states with non-zero amplitude in the target state.
        :param target_basis: Target basis state around which the zero amplitude state will be searched.
        :return: Closest to the target basis state with zero amplitude.
        """
        for state in PathFinder.get_hamming_generator(target_basis):
            # print(f"printing state: {state}")
            if state not in target_state:
                return state
        return None

    def get_path_segments(self, graph: Graph, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param target_state: Dict of non-zero amplitudes in the target state.
        :return: List of path segments.
        """
        tol = 1e-10
        # bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        # print("standard edges bfs :", bfs_edges)

        path = []
        for edge in bfs_edges:
            phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
            graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
            if abs(phase_walk_time) < tol:
                phase_walk_time = 0

            amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
            graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
            path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time))

            if graph.degree(edge[1]) == 1:
                phase_walk_time = (1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                closest_zero_state = PathFinder.find_closest_zero_amplitude(target_state, edge[1])
                path.append(PathSegment(list([edge[1], closest_zero_state]), phase_walk_time, 0))
        # print("standard path ", path)
        return path
    
    def get_path_segments_leafsm(self, graph: Graph, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param target_state: Dict of non-zero amplitudes in the target state.
        :return: List of path segments.
        """
        tol = 1e-10
        bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        path = []
        # print("edges bfs from pairs:", bfs_edges)
        # print("start from pairs", bfs_edges[0][0])
        for edge in bfs_edges:
            if graph.degree(edge[1]) != 1:
                phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
                graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
                if abs(phase_walk_time) < tol:
                    phase_walk_time = 0

                amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
                graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
                path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time))

            else:
                phase_walk_time1 = (-1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
                graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
                if abs(phase_walk_time1) < tol:
                    phase_walk_time1 = 0
                amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
                graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
                phase_walk_time2 = (-1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                path.append(LeafPathSegment(list(edge), phase_walk_time1, phase_walk_time2, amplitude_walk_time))
        # print("path from pairs", path)
        return path


    def get_path_segments_from_pairs_leafsm(self, graph: Graph, pairs: list[list[str, str]], target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param target_state: Dict of non-zero amplitudes in the target state.
        :return: List of path segments.
        """
        tol = 1e-10
        # bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        bfs_edges = pairs
        path = []
        # print("edges bfs from pairs:", bfs_edges)
        # print("start from pairs", bfs_edges[0][0])
        for edge in bfs_edges:
            if len(edge)==3:
                interaction_ind=edge[2]
                edge=edge[:-1:]
            else:
                interaction_ind=None
            if graph.degree(edge[1]) != 1:
                phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
                graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
                if abs(phase_walk_time) < tol:
                    phase_walk_time = 0

                amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
                graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
                path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time, interaction_index=interaction_ind))

            else:
                phase_walk_time1 = (-1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
                graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
                if abs(phase_walk_time1) < tol:
                    phase_walk_time1 = 0
                amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
                graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
                graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
                phase_walk_time2 = (-1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                path.append(LeafPathSegment(list(edge), phase_walk_time1, phase_walk_time2, amplitude_walk_time, interaction_index=interaction_ind))
        # print("path from pairs", path)
        return path
    
    # def get_path_segments_from_pairs_leafsm_test(self, graph: Graph, pairs: list[list[str, str]], target_state: dict[str, complex]) -> list[PathSegment]:
    #     """
    #     Returns a list of path segments describing state preparation path.
    #     :param graph: Travel graph.
    #     :param target_state: Dict of non-zero amplitudes in the target state.
    #     :return: List of path segments.
    #     """
    #     tol = 1e-10
    #     # bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
    #     bfs_edges = pairs[::-1] #we're doing an RXRZ pattern.
    #     path = []
    #     # print("edges bfs from pairs:", bfs_edges)
    #     # print("start from pairs", bfs_edges[0][0])
    #     for edge in bfs_edges:
    #         if len(edge)==3:
    #             interaction_ind=edge[2]
    #             edge=edge[:-1:]
    #         else:
    #             interaction_ind=None
    #         if graph.degree(edge[1]) != 1:
    #             phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
    #             graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
    #             if abs(phase_walk_time) < tol:
    #                 phase_walk_time = 0

    #             amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
    #             graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
    #             graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
    #             graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
    #             path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time, interaction_index=interaction_ind))

    #         else:
    #             phase_walk_time1 = (-1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
    #             graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
    #             if abs(phase_walk_time1) < tol:
    #                 phase_walk_time1 = 0
    #             amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
    #             graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
    #             graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
    #             graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
    #             phase_walk_time2 = (-1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
    #             graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
    #             path.append(LeafPathSegment(list(edge), phase_walk_time1, phase_walk_time2, amplitude_walk_time, interaction_index=interaction_ind))
    #     # print("path from pairs", path)
    #     return path
    
    def get_path_segments_from_pairs(self, graph: Graph, pairs: list[list[str, str]], target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param target_state: Dict of non-zero amplitudes in the target state.
        :return: List of path segments.
        """
        tol = 1e-10
        # bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        bfs_edges = pairs
        path = []
        # print("edges bfs from pairs:", bfs_edges)
        # print("start from pairs", bfs_edges[0][0])
        for edge in bfs_edges:
            phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
            graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
            if abs(phase_walk_time) < tol:
                phase_walk_time = 0

            amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
            graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
            path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time))

            if graph.degree(edge[1]) == 1:
                phase_walk_time = (1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                closest_zero_state = PathFinder.find_closest_zero_amplitude(target_state, edge[1])
                path.append(PathSegment(list([edge[1], closest_zero_state]), phase_walk_time, 0))
        # print("path from pairs", path)
        return path

    def get_path(self, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns state preparation path described by quantum walks.
        :param target_state: Target state to prepare.
        :return: List of path segments.
        """
        travel_graph = self.build_travel_graph(list(target_state.keys()))
        self.set_graph_attributes(travel_graph, target_state)
        return self.get_path_segments(travel_graph, target_state)
    
    def get_path_leafsm(self, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns state preparation path described by quantum walks.
        :param target_state: Target state to prepare.
        :return: List of path segments.
        """
        travel_graph = self.build_travel_graph(list(target_state.keys()))
        self.set_graph_attributes(travel_graph, target_state)
        return self.get_path_segments_leafsm(travel_graph, target_state)
    
    def get_path_from_pairs_leafsm(self, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns state preparation path described by quantum walks.
        :param target_state: Target state to prepare.
        :return: List of path segments.
        """
        travel_graph, basis_pairs = self.build_travel_graph(list(target_state.keys()))
        # print("initial pairs ", basis_pairs)
        self.set_graph_attributes_from_pairs(travel_graph, basis_pairs, target_state)
        # print("after pairs ", basis_pairs)
        return self.get_path_segments_from_pairs_leafsm(travel_graph, basis_pairs, target_state)


class PathFinderRandom(PathFinder):
    """ Connects the states via a random tree. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        random_tree = nx.generators.random_tree(len(bases))
        random_tree = nx.relabel_nodes(random_tree, {i: bases[i] for i in range(len(bases))})
        random_tree.graph["start"] = bases[0]
        return random_tree
    
# @dataclass
class PathFinderFromPairs(PathFinder):
    """ Goes through the states in the same order they are listed in. """

    def __init__(self, basis_pairs: Sequence[Sequence[str, str]]):
        self.basis_pairs=basis_pairs

    def build_travel_graph(self, _: list[str]) -> Graph:
        graph = Graph()
        for b1, b2 in self.basis_pairs:
            # print(b1)
            # print(b2)
            graph.add_edge(b1, b2)
        graph.graph["start"] = self.basis_pairs[0][0]
        return graph


@dataclass
class PathFinderLinear(PathFinder):
    """ Goes through the states in the same order they are listed in. """
    order: Sequence[int] = None

    def build_travel_graph(self, bases: list[str]) -> Graph:
        graph = Graph()
        if self.order is not None:
            bases = np.array(bases)[self.order]
        for i in range(len(bases) - 1):
            graph.add_edge(bases[i], bases[i + 1])
        graph.graph["start"] = bases[0]
        return graph


class PathFinderGrayCode(PathFinder):
    """ Goes through the states in the order of Gray code. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        graph = Graph()
        gray_code = graycode.gen_gray_codes(len(bases[0]))
        gray_code_str = [f"{code:0{len(bases[0])}b}" for code in gray_code]
        last_basis = None
        for code in gray_code_str:
            if code in bases:
                if last_basis is None:
                    graph.graph["start"] = code
                else:
                    graph.add_edge(last_basis, code)
                last_basis = code
        return graph


def build_distance_graph(bases: list[str]) -> Graph:
    """
    Builds the fully connected graph on the nodes in bases where each weight is given by hamming distance.
    :param bases: List of bases to include in the graph.
    :return: Graph.
    """
    get_hamming_distance = lambda str1, str2: sum(c1 != c2 for c1, c2 in zip(str1, str2))
    graph = Graph()
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            graph.add_edge(bases[i], bases[j], weight=get_hamming_distance(bases[i], bases[j]))
    return graph


class PathFinderSHP(PathFinder):
    """ Returns the Shortest Hamiltonian Path throughout the target basis states. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        distance_graph = build_distance_graph(bases)
        shp_nodes = nx.approximation.traveling_salesman_problem(distance_graph, cycle=False)
        travel_graph = Graph()
        travel_graph.graph["start"] = shp_nodes[0]
        for i in range(len(shp_nodes) - 1):
            travel_graph.add_edge(shp_nodes[i], shp_nodes[i + 1])
        return travel_graph


class PathFinderMST(PathFinder):
    """Returns the Minimum Spanning Tree on the basis states. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        distance_graph = build_distance_graph(bases)
        mst = nx.minimum_spanning_tree(distance_graph, algorithm='prim')
        mst.graph["start"] = bases[0]
        return mst


class PathFinderMHSNonlinear(PathFinder):
    """The Minimum Hitting Set method on the basis states. """
    def build_travel_graph(self, states: list[str]) -> Graph:
        # print(states)
        # print(ordered_states)
        graph = Graph()
        path=[]
        basis_original=deepcopy(states)
        basis_mutable=deepcopy(states)
        # indices=list(range(len(basis_original[0])))
        for _ in range(len(states)-1):
            # print("ordered states easiest firs ", self.order_basis_states_mhs(basis_mutable))
            z1_updated=self.order_basis_states_mhs(basis_mutable)[0]
            z1_idx=basis_mutable.index(z1_updated)
            z1_original=basis_original[z1_idx]
            # remaining_basis=deepcopy(basis_mutable)
            # remaining_basis.remove(z1_updated)

            z2_updated, interaction_ind=self._get_partner_node(z1_updated, basis_mutable)
            z2_idx=basis_mutable.index(z2_updated)
            z2_original=basis_original[z2_idx]
            path.append([z1_original, z2_original, interaction_ind])
            basis_mutable.pop(z2_idx)
            basis_original.pop(z2_idx)
            # print("interaction index before update: ", interaction_ind)
            # print("z1 z2 ", z1_updated, z2_updated)
            basis_mutable=self.update_nodes(z1_updated, z2_updated, basis_mutable, interaction_ind)
            
        path=path[::-1] #we worked backwards.
        for pair in path:
            graph.add_edge(pair[0], pair[1])
        graph.graph["start"] = path[0][0]
        return graph, path
    
    @staticmethod
    def _construct_z2_search(intercation_ind, remaining_basis, z1_diffs, z1_mhs):
        '''Filters remaing_basis such that z1_diffs[i] intersects z1_mhs[i] only at interaction_ind.'''
        new_reamining_basis=[remaining_basis[idx] for idx, elem in enumerate(z1_diffs) if set(elem).intersection(set(z1_mhs))==set([intercation_ind])]
        return new_reamining_basis


    def update_nodes(self, z1, z2, visited, interaction_ind):
        '''Updates visited.'''
        z1=[int(char) for char in z1]
        z2=[int(char) for char in z2]
        diff_inds = list(np.where(np.array(z1) != np.array(z2))[0])
        # print(diff_inds)
        diff_inds.remove(interaction_ind)
        # print("interaction for updating ", interaction_ind)
        # print("before updating visited ", visited)
        # print(f"z1 z2 node {z1}, {z2}")
        # print("diff inds ", diff_inds)
        visited=[[int(char) for char in st] for st in visited]
        for target in diff_inds: #update the visited nodes
            update_visited(visited, interaction_ind, target)
        # print("right after ", visited)
        visited=["".join(list(map(str,elem))) for elem in visited]
        # print("after updating visited ", visited)

        return visited

    # @staticmethod
    # def order_by_hamming_dist(origin, remaining_basis):
    #     '''Orders the remaining basis by Hamming distance from origin.'''
    #     # print("origin ", origin)
    #     # print("remaining basis ", remaining_basis)
    #     return sorted(remaining_basis, key=lambda elem: (hamming_dist(origin, elem),))
    #                 # PathFinderMHSLinear.get_mhs_score(elem, remaining_basis)))
    
    @staticmethod
    def get_mhs_score(elem, remaining_basis):
        return len(PathFinderMHSLinear.get_mhs(elem, remaining_basis))
    
    @staticmethod
    def get_mhs(elem, remaining_basis):
        # print("remaining basis ", remaining_basis)
        # print("origin ", elem)
        diffs= [[ind for ind in range(len(elem)) if elem[ind] != z1[ind]] for z1 in remaining_basis]
        # print(f"diffs {diffs}")
        hitman = Hitman()
        for inds_set in diffs:
            hitman.hit(inds_set)
        # print(hitman.get())
        return hitman.get()
    
    # @staticmethod
    # def get_all_mhs_scores(basis:list):
    #     return [PathFinderMHSLinear.get_mhs_score(elem, [elem2 for elem2 in basis if elem2!=elem]) for elem in basis]

    def get_z2_search(self, elem, basis):
        '''Returns the z2 search space and target qubit.
        :param elem: z1
        :param basis: all the basis states including elem.
        :return: z2 and interaction index '''
        indices=range(len(basis[0]))
        remaining_basis1=[elem2 for elem2 in basis if elem2!=elem]
        mhs1=self.get_mhs(elem, remaining_basis1)
        diffs1=[[ind for ind in indices if elem[ind] != z2[ind]] for z2 in remaining_basis1]
        mhs_freq_sorted1=sorted(mhs1, key=lambda idx: sum([1 for block in diffs1 for elem in block if elem==idx]))
        interaction_ind=mhs_freq_sorted1[0]
        z2_search=self._construct_z2_search(interaction_ind, remaining_basis1, diffs1, mhs1)
        return [z2_search, interaction_ind]

    @staticmethod
    def order_basis_states_mhs(basis:list, z1=None):
        '''Orders basis by MHS score of each elem. It includes the cost of MHS of z1 and z2 and takes into
        account the target qubit.'''
        def _count_elements(basis):
            return sum([len(block) for block in basis])
        def _create_remaining_basis(elem, basis):
            return [elem2 for elem2 in basis if elem2!=elem]
        def _get_z2_mhs_score(elem, z2_search):
            '''returns the number of controls required to differentiate z2.'''
            # print("z1 ", elem)
            # print("remaining basis1 ", remaining_basis1)
            # print("diffs1 ", diffs1)
            # print("mhs1 ", mhs1)
            # print("z2 search ", z2_search)
            if len(z2_search)>1:
                z2_search=sorted(z2_search, key=lambda z2:
                                    (PathFinderMHSLinear.get_mhs_score(z2, _create_remaining_basis(z2, z2_search)),
                                    hamming_dist(elem, z2)))
                z2=z2_search[0]
                z2_score=PathFinderMHSLinear.get_mhs_score(z2, _create_remaining_basis(z2, z2_search))
            else:
                z2_score=0
            return z2_score
        
        indices=range(len(basis[0]))
        if z1 is not None:
             z2_search=PathFinderMHSLinear().get_z2_search(z1, basis)[0]
            #  print("new z2_search ", z2_search)
             return sorted(z2_search, key=lambda z2:
                                  (PathFinderMHSLinear.get_mhs_score(z2, _create_remaining_basis(z2, z2_search)),
                                hamming_dist(z2, z1)))
        else:
            return sorted(basis, key=lambda elem:
                                    (PathFinderMHSLinear.get_mhs_score(elem, _create_remaining_basis(elem, basis))
                                     +_get_z2_mhs_score(elem, PathFinderMHSLinear().get_z2_search(elem, basis)[0]),
                    -1*_count_elements([[ind for ind in indices if elem[ind] != z1[ind]] for z1 in _create_remaining_basis(elem, basis)])))
    
    def _get_partner_node(self, z1_updated, basis_mutable):
        '''Gets the partner node for z1_updated.'''
        z2_search_params=self.get_z2_search(z1_updated, basis_mutable)
        interaction_ind=z2_search_params[1]
        z2_updated=self.order_basis_states_mhs(basis_mutable, z1_updated)[0]
        return z2_updated, interaction_ind


class PathFinderMHSLinear(PathFinderMHSNonlinear):
    def build_travel_graph(self, states: list[str]) -> Graph:
        # print(states)
        # print(ordered_states)
        graph = Graph()
        path=[]
        path_mutable=[]
        basis_original=deepcopy(states)
        basis_mutable=deepcopy(states)
        # indices=list(range(len(basis_original[0])))
        for _ in range(len(states)-1):
            # print("ordered states easiest firs ", self.order_basis_states_mhs(basis_mutable))
            if not path_mutable:
                # z2_updated=self.order_basis_states_mhs(basis_mutable)[0]
                # z2_idx=basis_mutable.index(z2_updated)
                # z2_original=basis_original[z2_idx]
                # z1_updated, interaction_ind=self._get_partner_node(z2_updated, basis_mutable)
                # z1_idx=basis_mutable.index(z1_updated)
                # z1_original=basis_original[z1_idx]

                z1_updated=self.order_basis_states_mhs(basis_mutable)[0]
                z1_idx=basis_mutable.index(z1_updated)
                z1_original=basis_original[z1_idx]
                z2_updated, interaction_ind=self._get_partner_node(z1_updated, basis_mutable)
                z2_idx=basis_mutable.index(z2_updated)
                z2_original=basis_original[z2_idx]

            else:
                z2_updated=path_mutable[-1][0]
                z2_idx=basis_mutable.index(z2_updated)
                z2_original=basis_original[z2_idx]
                z1_updated, interaction_ind=self._get_partner_node(z2_updated, basis_mutable)
                z1_idx=basis_mutable.index(z1_updated)
                z1_original=basis_original[z1_idx]
            path.append([z1_original, z2_original, interaction_ind])
            # print("interaction index before update: ", interaction_ind)
            # print("z1 z2 ", z1_updated, z2_updated)
            basis_mutable=self.update_nodes(z1_updated, z2_updated, basis_mutable, interaction_ind)
            path_mutable.append([basis_mutable[z1_idx], basis_mutable[z2_idx], interaction_ind])
            basis_mutable.pop(z2_idx)
            basis_original.pop(z2_idx)
            # print(f"pair: {z1_original} {z2_original}")
            
        path=path[::-1] #we worked backwards.
        for pair in path:
            graph.add_edge(pair[0], pair[1])
        graph.graph["start"] = path[0][0]
        return graph, path

@dataclass
class GleinigPathFinder(PathFinder):
    """Returns the Minimum Spanning Tree on the basis states. """
    def build_travel_graph(self, states: list[str]) -> Graph:
        val=1/np.sqrt(len(states))
        state_dict={k: val for k in states}
        walker=GleinigWalk(state_dict)
        walks=walker.get_gleinig_path()
        bases=walks#list(reversed(walks))
        # print(bases)
        graph = Graph()
        # for i in range(len(bases) - 1):
        #     graph.add_edge(bases[i], bases[i + 1])
        # graph.graph["start"] = bases[0]
        # bases=[list(reversed(pair)) for pair in bases]
        for pair in bases:
            graph.add_edge(pair[0], pair[1])
        graph.graph["start"] = bases[0][0]
        return graph, walks
    
@dataclass
class GleinigPathFinderPframes(PathFinder):
    """Returns the Minimum Spanning Tree on the basis states. """
    def build_travel_graph(self, states: list[str]) -> Graph:
        val=1/np.sqrt(len(states))
        state_dict={k: val for k in states}
        walker=GleinigWalk(state_dict)
        walks, root=walker.get_gleinig_path_pairs_pframes()
        # print(walks)
        basis_pairs=list(reversed(walks))
        # basis_pairs=[('00110', '00111'), ('00111', '00001'), ('00001', '01011'), ('01011', '01001')]
        # basis_pairs=[['10010001', '01110000'], ['10010001', '01110101'], ['10010001', '11001110'], ['10010001', '11011011'], ['10010001', '01101111'], ['10010001', '11010001'], ['10010001', '00001111']]
        # print(basis_pairs)
        # basis_pairs=[['10010001', '01110000'], ['10010001', '01110101'], ['10010001', '11001110'], ['10010001', '11011011'], ['10010001', '01101111'], ['10010001', '11010001'], ['10010001', '00001111']]
        # print(bases)
        graph = Graph()
        for z1, z2 in basis_pairs:
            graph.add_edge(z1, z2)
        graph.graph["start"] = root# basis_pairs[0][0]
        # graph.graph["start"] = basis_pairs[0][0]
        # bases=[list(reversed(pair)) for pair in bases]
        # for pair in bases:
        #     graph.add_edge(pair[0], pair[1])
        # graph.graph["start"] = bases[0][0]
        return graph, basis_pairs

class GleinigWalk():
    def __init__(self, state_dict) -> None:
        self.state_dict=deepcopy(state_dict)

    def get_gleinig_path(self):
        path=[]
        while(len(self.state_dict.keys())>1):
            bitstr1, bitstr2, _, _ = self._select_strings(self.state_dict)
            self.state_dict.pop(bitstr2)
            # path.append([bitstr1, bitstr2])
            path.append([bitstr1, bitstr2])
            # if len(self.state_dict.keys())==1:
            #     path.append(bitstr2)
        # print(path[::-1])
        return path[::-1]
    
    def get_gleinig_path_pframes(self):
        bases_original=list(self.state_dict.keys())
        bases=list(self.state_dict.keys())
        bitstr1, bitstr2, _, _ = self._select_strings(self.state_dict)
        idx1=bases.index(bitstr1)
        idx2=bases.index(bitstr2)
        path=[]
        path+=[bases_original[idx1], bases_original[idx2]]
        remove_idxs=[idx1, idx2]
        bases_original=[elem for i, elem in enumerate(bases_original) if i not in remove_idxs]
        bases=[elem for i, elem in enumerate(bases) if i not in remove_idxs]
        while(len(bases)>1):
            # path.append([bitstr1, bitstr2])
            temp_basis_sts=[[int(char) for char in elem] for elem in bases]
            diff_inds = np.where(np.array(path[-1]) != np.array(path[-2]))[0]
            for ind in diff_inds[1::]:
                update_visited(temp_basis_sts, diff_inds[0], ind)
            temp_basis_sts=[[str(char) for char in elem] for elem in temp_basis_sts]
            bases=["".join(elem) for elem in temp_basis_sts]

            # bitstr1=sorted(bases, key=lambda x: int(x, 2))[0]
            val=1/np.sqrt(len(bases))
            state_dict={k: val for k in bases} #construct a dummy state

            bitstr1, _, _, _ = self._select_strings(state_dict)
            idx1=bases.index(bitstr1)
            path.append(bases_original[idx1])
            bases.pop(idx1)
            bases_original.pop(idx1)
        path+=bases_original
        # print(path)
        return path
    
    def get_gleinig_path_pairs_pframes(self):
        bases_original=list(self.state_dict.keys())
        bases=list(self.state_dict.keys())
        bitstr1, bitstr2, _, _ = self._select_strings_pframes(self.state_dict)
        idx1=bases.index(bitstr1)
        idx2=bases.index(bitstr2)
        path=[]
        path+=[[bases_original[idx1], bases_original[idx2]]]
        remove_idxs=[idx2]
        bases_original=[elem for i, elem in enumerate(bases_original) if i not in remove_idxs]
        bases=[elem for i, elem in enumerate(bases) if i not in remove_idxs]
        while(len(bases)>1):
            # path.append([bitstr1, bitstr2])
            temp_basis_sts=[[int(char) for char in elem] for elem in bases]
            diff_inds = np.where(np.array(path[-1][0]) != np.array(path[-1][1]))[0]
            for ind in diff_inds[1::]:
                update_visited(temp_basis_sts, diff_inds[0], ind)
            temp_basis_sts=[[str(char) for char in elem] for elem in temp_basis_sts]
            bases=["".join(elem) for elem in temp_basis_sts]

            # bitstr1=sorted(bases, key=lambda x: int(x, 2))[0]
            val=1/np.sqrt(len(bases))
            state_dict={k: val for k in bases} #construct a dummy state

            bitstr1, bitstr2, _, _ = self._select_strings_pframes(state_dict)
            idx1=bases.index(bitstr1)
            idx2=bases.index(bitstr2)
            path.append([bases_original[idx1], bases_original[idx2]])
            bases.pop(idx2)
            bases_original.pop(idx2)
        # path+=bases_original
        return path, bases_original[0]
    
    @staticmethod
    def _select_strings_pframes(state_dict):
        """
        Searches for the states described by the bit strings bitstr1 and bitstr2 to be merged
        Args:
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        Returns:
        bitstr1: First binary string
        bitstr2: Second binary string
        dif_qubit: Qubit index to be used as target for the merging operation
        dif_qubits: List of qubit indexes where bitstr1 and bitstr2 must be equal, because the
                    correspondig qubits of those indexes are to be used as control for the
                    merging operation
        """
        # Initialization
        dif_qubits = []
        dif_values = []
        b_strings1 = b_strings2 = list(state_dict.keys())

        # Searching for bitstr1
        (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        dif_qubit = dif_qubits.pop()
        dif_values.pop()
        bitstr1 = b_strings1[0]

        # Searching for bitstr2
        b_strings2.remove(bitstr1)
        b_strings1 = GleinigWalk._build_bit_string_set(b_strings2, dif_qubits, dif_values)
        (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        bitstr2 = b_strings1[0]

        return bitstr1, bitstr2, dif_qubit, dif_qubits
    
    # @staticmethod
    # def select_first_string(state_dict):
    #     """
    #     Searches for the states described by the bit strings bitstr1 and bitstr2 to be merged
    #     Args:
    #     state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
    #                 binary strings as keys e.g.: {'001': <value>, '101': <value>}
    #     Returns:
    #     bitstr1: First binary string
    #     bitstr2: Second binary string
    #     dif_qubit: Qubit index to be used as target for the merging operation
    #     dif_qubits: List of qubit indexes where bitstr1 and bitstr2 must be equal, because the
    #                 correspondig qubits of those indexes are to be used as control for the
    #                 merging operation
    #     """
    #     # Initialization
    #     dif_qubits = []
    #     dif_values = []
    #     b_strings1 = b_strings2 = list(state_dict.keys())

    #     # Searching for bitstr1
    #     (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
    #         b_strings1, dif_qubits, dif_values
    #     )
    #     dif_qubit = dif_qubits.pop()
    #     dif_values.pop()
    #     bitstr1 = b_strings1[0]

    #     # # Searching for bitstr2
    #     # b_strings2.remove(bitstr1)
    #     # b_strings1 = GleinigWalk._build_bit_string_set(b_strings2, dif_qubits, dif_values)
    #     # (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
    #     #     b_strings1, dif_qubits, dif_values
    #     # )
    #     # bitstr2 = b_strings1[0]

    #     return bitstr1#, bitstr2, dif_qubit, dif_qubits

    @staticmethod
    def _select_strings(state_dict):
        """
        Searches for the states described by the bit strings bitstr1 and bitstr2 to be merged
        Args:
        state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                    binary strings as keys e.g.: {'001': <value>, '101': <value>}
        Returns:
        bitstr1: First binary string
        bitstr2: Second binary string
        dif_qubit: Qubit index to be used as target for the merging operation
        dif_qubits: List of qubit indexes where bitstr1 and bitstr2 must be equal, because the
                    correspondig qubits of those indexes are to be used as control for the
                    merging operation
        """
        # Initialization
        dif_qubits = []
        dif_values = []
        b_strings1 = b_strings2 = list(state_dict.keys())

        # Searching for bitstr1
        (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        dif_qubit = dif_qubits.pop()
        dif_values.pop()
        bitstr1 = b_strings1[0]

        # Searching for bitstr2
        b_strings2.remove(bitstr1)
        b_strings1 = GleinigWalk._build_bit_string_set(b_strings2, dif_qubits, dif_values)
        (b_strings1, dif_qubits, dif_values) = GleinigWalk._bit_string_search(
            b_strings1, dif_qubits, dif_values
        )
        bitstr2 = b_strings1[0]

        return bitstr1, bitstr2, dif_qubit, dif_qubits
    
    @staticmethod
    def _bit_string_search_hardest(b_strings, dif_qubits, dif_values):
        """
        Searches for the bit strings with unique qubit values in `dif_values`
        on indexes `dif_qubits`.
        Args:
        b_strings: List of binary strings where the search is to be performed
                    e.g.: ['000', '010', '101', '111']
        dif_qubits: List of indices on a binary string of size N e.g.: [1, 3, 5]
        dif_values: List of values each qubit must have on indexes stored in dif_qubits [0, 1, 1]
        Returns:
        b_strings: One size list with the string found, to have values dif_values on indexes
                    dif_qubits
        dif_qubits: Updated list with new indexes
        dif_values: Updated list with new values
        """
        temp_strings = b_strings
        while len(temp_strings) > 1:
            bit, t_0, t_1 = GleinigWalk._maximizing_difference_bit_search(
                temp_strings, dif_qubits
            )
            dif_qubits.append(bit)
            # if len(t_0) < len(t_1): # original
            if len(t_0) > len(t_1):
                dif_values.append("0")
                temp_strings = t_0
            else:
                dif_values.append("1")
                temp_strings = t_1

        return temp_strings, dif_qubits, dif_values
    
    @staticmethod
    def _bit_string_search(b_strings, dif_qubits, dif_values):
        """
        Searches for the bit strings with unique qubit values in `dif_values`
        on indexes `dif_qubits`.
        Args:
        b_strings: List of binary strings where the search is to be performed
                    e.g.: ['000', '010', '101', '111']
        dif_qubits: List of indices on a binary string of size N e.g.: [1, 3, 5]
        dif_values: List of values each qubit must have on indexes stored in dif_qubits [0, 1, 1]
        Returns:
        b_strings: One size list with the string found, to have values dif_values on indexes
                    dif_qubits
        dif_qubits: Updated list with new indexes
        dif_values: Updated list with new values
        """
        temp_strings = b_strings
        while len(temp_strings) > 1:
            bit, t_0, t_1 = GleinigWalk._maximizing_difference_bit_search(
                temp_strings, dif_qubits
            )
            dif_qubits.append(bit)
            if len(t_0) < len(t_1): # original
            # if len(t_0) > len(t_1):
                dif_values.append("0")
                temp_strings = t_0
            else:
                dif_values.append("1")
                temp_strings = t_1

        return temp_strings, dif_qubits, dif_values
    
    @staticmethod
    def _maximizing_difference_bit_search(b_strings, dif_qubits):
        """
        Splits the set of bit strings into two (t_0 and t_1), by setting
        t_0 as the set of bit_strings with 0 in the bit_index position, and
        t_1 as the set of bit_strings with 1 in the bit_index position.
        Searching for the bit_index not in dif_qubits that maximizes the difference
        between the size of the nonempty t_0 and t_1.
        Args:
        b_string: A list of bit strings eg.: ['000', '011', ...,'101']
        dif_qubits: A list of previous qubits found to maximize the difference
        Returns:
        bit_index: The qubit index that maximizes abs(len(t_0)-len(t_1))
        t_0: List of binary strings with 0 on the bit_index qubit
        t_1: List of binary strings with 1 on the bit_index qubit
        """
        t_0 = []
        t_1 = []
        bit_index = 0
        set_difference = -1
        bit_search_space = list(set(range(len(b_strings[0]))) - set(dif_qubits))

        for bit in bit_search_space:
            temp_t0 = [x for x in b_strings if x[bit] == "0"]
            temp_t1 = [x for x in b_strings if x[bit] == "1"]

            if temp_t0 and temp_t1:
                temp_difference = np.abs(len(temp_t0) - len(temp_t1))
                if temp_difference > set_difference:
                    t_0 = temp_t0
                    t_1 = temp_t1
                    bit_index = bit
                    set_difference = temp_difference

        return bit_index, t_0, t_1
    
    @staticmethod
    def _build_bit_string_set(b_strings, dif_qubits, dif_values):
        """
        Creates a new set of bit strings from b_strings, where the bits
        in the indexes in dif_qubits match the values in dif_values.

        Args:
        b_strings: list of bit strings eg.: ['000', '011', ...,'101']
        dif_qubits: list of integers with the bit indexes
        dif_values: list of integers values containing the values each bit
                    with index in dif_qubits shoud have
        Returns:
        A new list of bit_strings, with matching values in dif_values
        on indexes dif_qubits
        """
        bit_string_set = []
        for b_string in b_strings:
            if [b_string[i] for i in dif_qubits] == dif_values:
                bit_string_set.append(b_string)

        return bit_string_set
    

def hamming_dist(z1, z2):
    return sum([b1 != b2 for b1, b2 in zip(z1,z2)])
    
def get_hamming_weight(bits):
    # tot=0
    # for elem in bits:
    #     if elem=="1":
    #         tot+= 1
    return bits.count("1")
    
def get_shp(bases: list[str]) -> list[str]:
    # print(bases)
    distance_graph = build_distance_graph(bases)
    shp_nodes = nx.approximation.traveling_salesman_problem(distance_graph, cycle=False)
    # print(shp_nodes)
    # travel_graph = Graph()
    # travel_graph.graph["start"] = shp_nodes[0]
    # for i in range(len(shp_nodes) - 1):
    #     travel_graph.add_edge(shp_nodes[i], shp_nodes[i + 1])
    return shp_nodes

def update_visited(visited: list[list[int]], control: int, target: int):
    """
    Updates visited nodes to reflect the action of specified CX gates.
    :param visited: List of basis labels of visited states.
    :param control: Index of the control qubit for the CX operation.
    :param target: Index of the target qubit for the CX operation.
    """
    for label in visited:
        if label[control] == 1:
            label[target] = 1 - label[target]

def find_min_control_set(existing_states: list[list[int]], target_state_ind: int, interaction_ind: int) -> list[int]:
    """
    Finds minimum set of control necessary to select the target state.
    :param existing_states: List of states with non-zero amplitudes.
    :param target_state_ind: Index of the target state in the existing_states.
    :param interaction_ind: Index of target qubit for the controlled operation (to exclude from consideration for the control set).
    :return: Minimum set of control indices necessary to select the target state.
    """
    get_diff_inds = lambda state1, state2: [ind for ind in range(len(state1)) if ind != interaction_ind and state1[ind] != state2[ind]]
    difference_inds = [get_diff_inds(state, existing_states[target_state_ind]) for state_ind, state in enumerate(existing_states) if state_ind != target_state_ind]
    # print(get_diff_inds)
    # print("interaction idx ", interaction_ind)
    # print("difference indices", difference_inds)
    hitman = Hitman()
    for inds_set in difference_inds:
        hitman.hit(inds_set)
    # print("hitman min set ", hitman.get())
    return hitman.get()


if __name__=="__main__":
    amp=1/np.sqrt(6)
    basis=["000001", "000011", "000111", "001111", "011111", "111111"]
    pathf=PathFinderMHSLinear()
    print(pathf.build_travel_graph(basis)[1])
    