""" Module for converting quantum walks to circuit representation. """
import copy

import numpy as np
from pysat.examples.hitman import Hitman
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, RXGate
from qiskit.converters import circuit_to_dag, dag_to_circuit

from src.quantum_walks import PathSegment


class PathConverter:
    """ Converts state preparation paths to the equivalent circuits. """
    @staticmethod
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

    @staticmethod
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
        hitman = Hitman()
        for inds_set in difference_inds:
            hitman.hit(inds_set)
        return hitman.get()

    @staticmethod
    def remove_leading_cx(qc: QuantumCircuit) -> QuantumCircuit:
        """
        Removes leading CX gates whose controls are always false.
        :param qc: Input quantum circuit.
        :return: Optimized quantum circuit where the leading CX gates are removed where possible.
        """
        dag = circuit_to_dag(qc)
        wires = dag.wires
        # Go through the ops in each wire and remove cx ops until we run into a non cx operation.
        for w in wires:
            for node in list(dag.nodes_on_wire(w, only_ops=True)):
                if node.name == "barrier":
                    continue
                elif node.name != "cx":
                    break
                # The control is the current wire
                elif node.qargs[0] == w:
                    dag.remove_op_node(node)
                # CX but with target on qubit wire w.
                else:
                    break
        return dag_to_circuit(dag)

    @staticmethod
    def convert_path_to_circuit(path: list[PathSegment], reduce_controls: bool = True, remove_leading_cx: bool = True, add_barriers: bool = False) -> QuantumCircuit:
        """
        Converts quantum walks to qiskit circuit.
        :param path: List of path segments, describing the state preparation path.
        :param reduce_controls: True to search for minimally necessary state of controls. False to use all n-1 controls (for debug purposes).
        :param remove_leading_cx: True to remove leading CX gates whose controls are never satisfied.
        :param add_barriers: True to insert barriers between path segments.
        :return: Implementing circuit.
        """
        starting_state = path[0].labels[0]
        qc = QuantumCircuit(len(starting_state))
        indices_1 = [ind for ind, elem in enumerate(starting_state) if elem == "1"]
        for ind in indices_1:
            qc.x(ind)
        if add_barriers:
            qc.barrier()

        visited = [[int(char) for char in starting_state]]
        for segment in path:
            origin = [int(char) for char in segment.labels[0]]
            destination = [int(char) for char in segment.labels[1]]
            diff_inds = np.where(np.array(origin) != np.array(destination))[0]
            interaction_ind = diff_inds[0]

            visited_transformed = copy.deepcopy(visited)
            for ind in diff_inds[1:]:
                qc.cx(interaction_ind, ind)
                PathConverter.update_visited(visited_transformed, interaction_ind, ind)

            origin_ind = visited.index(origin)
            if reduce_controls:
                control_indices = PathConverter.find_min_control_set(visited_transformed, origin_ind, interaction_ind)
            else:
                control_indices = [ind for ind in range(len(origin)) if ind != interaction_ind]

            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            rz_angle = 2 * segment.phase_time
            if origin[interaction_ind] == 1:
                rz_angle *= -1
            if rz_angle != 0:
                rz_gate = RZGate(rz_angle)
                if len(control_indices) > 0:
                    rz_gate = rz_gate.control(len(control_indices))
                qc.append(rz_gate, control_indices + [interaction_ind])

            rx_angle = 2 * segment.amplitude_time
            if rx_angle != 0:
                rx_gate = RXGate(rx_angle)
                if len(control_indices) > 0:
                    rx_gate = rx_gate.control(len(control_indices))
                qc.append(rx_gate, control_indices + [interaction_ind])
                visited.append(destination)

            for ind in control_indices:
                if visited_transformed[origin_ind][ind] == 0:
                    qc.x(ind)

            for ind in reversed(diff_inds[1:]):
                qc.cx(interaction_ind, ind)
            if add_barriers:
                qc.barrier()

        if remove_leading_cx:
            qc = PathConverter.remove_leading_cx(qc)

        return qc
