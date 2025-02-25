""" Plots for the quantum walks paper. """
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import sys
sys.path.append("../")
from plots.general import Line, plot_general, save_figure


def plot_cx_count_vs_num_qubits_line(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
    data = []
    data_std_dev=[]
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        column_dat=df[method]
        data.append(np.mean(column_dat))
        data_std_dev.append(np.std(column_dat))

    line = Line(num_qubits, data, color=color_ind, marker=marker_ind, label=label)
    plot_general([line], [data_std_dev], ("n", "CX"), boundaries=(4.75, 11.25, 10, 10 ** 4), figure_id=figure_id)
    plt.yscale("log")

# def plot_cx_count_vs_num_qubits_line_fit(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
#     data = []
#     data_std_dev=[]
#     for n, m in zip(num_qubits, num_amplitudes):
#         data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
#         df = pd.read_csv(data_path)
#         column_dat=df[method]
#         data.append(np.mean(column_dat))
#         data_std_dev.append(np.std(column_dat))

#     line = Line(num_qubits, data, color=color_ind, marker=marker_ind, label=label)
#     plot_general([line], [data_std_dev], ("n", "CX"), boundaries=(4.75, 11.25, 10, 10**4), figure_id=figure_id)
#     plt.yscale("log")
#     # plt.tight_layout()

def plot_cx_count_vs_num_qubits_line_linear(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
    data = []
    data_std_dev=[]
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        column_dat=df[method]
        data.append(np.mean(column_dat))
        data_std_dev.append(np.std(column_dat))

    line = Line(num_qubits, data, color=color_ind, marker=marker_ind, label=label)
    plot_general([line], [data_std_dev], ("n", "CX"), boundaries=(4.75, 11.25, 0, 90), figure_id=figure_id)
    plt.yscale("linear")
    # return line


def plot_control_reduction_effect():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    plot_cx_count_vs_num_qubits_line("random_reduced", num_qubits, num_amplitudes, 0, 0, "With control reduction", figure_id)
    plot_cx_count_vs_num_qubits_line("random", num_qubits, num_amplitudes, 1, 0, "Without control reduction", figure_id)
    save_figure()


def plot_walk_order_comparison():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    methods = ["random_reduced", "mst_reduced", "shp_reduced", "linear_reduced"]
    methods = [method + "_reduced" for method in methods]
    labels = ["Random", "MST", "SHP", "Sorted"]
    for method_ind, method in enumerate(methods):
        plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, method_ind, 0, labels[method_ind], figure_id)
    # plt.yscale("linear")
    plt.ylim(top=175)
    save_figure()


def plot_qiskit_comparison():
    methods_all = ["shp_reduced", "qiskit"]
    densities_all = [lambda n: n, lambda n: n ** 2, lambda n: 2 ** (n - 1)]
    figure_id = 0

    for method_ind, method in enumerate(methods_all):
        for density_ind, density in enumerate(densities_all):
            if method == "shp_reduced" and density_ind == 2:
                num_qubits = np.array(range(5, 10))
            else:
                num_qubits = np.array(range(5, 12))
            num_amplitudes = [densities_all[density_ind](n) for n in num_qubits]
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, density_ind, method_ind, "_nolabel_", figure_id)

    circle_marker = Line2D([0], [0], linestyle="", color="k", marker="o", markersize=5, label="Quantum Walks")
    star_marker = Line2D([0], [0], linestyle="", color="k", marker="*", markersize=8, label="Qiskit")
    blue_line = Line2D([0], [0], color="b", label=r"$m = n$")
    red_line = Line2D([0], [0], color="r", label=r"$m = n^2$")
    green_line = Line2D([0], [0], color="g", label=r"$m = 2^{n-1}$")
    plt.legend(handles=[circle_marker, star_marker, blue_line, red_line, green_line])
    save_figure()

def plot_gleinig_comparison():
    methods_all = ["shp_reduced", "gleinig"]
    densities_all = [lambda n: n, lambda n: n ** 2, lambda n: 2 ** (n - 1)]
    figure_id = 0

    for method_ind, method in enumerate(methods_all):
        for density_ind, density in enumerate(densities_all):
            if method == "shp_reduced" and density_ind == 2:
                num_qubits = np.array(range(5, 10))
            else:
                num_qubits = np.array(range(5, 12))
            num_amplitudes = [densities_all[density_ind](n) for n in num_qubits]
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, density_ind, method_ind, "_nolabel_", figure_id)

    circle_marker = Line2D([0], [0], linestyle="", color="k", marker="o", markersize=5, label="Quantum Walks")
    star_marker = Line2D([0], [0], linestyle="", color="k", marker="*", markersize=8, label="Gleinig")
    blue_line = Line2D([0], [0], color="b", label=r"$m = n$")
    red_line = Line2D([0], [0], color="r", label=r"$m = n^2$")
    green_line = Line2D([0], [0], color="g", label=r"$m = 2^{n-1}$")
    plt.legend(handles=[circle_marker, star_marker, blue_line, red_line, green_line])
    save_figure()

def plot_gleinigwalk_gleinig_comparison():
    methods_all = ["gleinig_qwalk", "gleinig"]
    densities_all = [lambda n: n]#, lambda n: n ** 2, lambda n: 2 ** (n - 1)]
    figure_id = 0

    for method_ind, method in enumerate(methods_all):
        for density_ind, density in enumerate(densities_all):
            if method == "gleinig_qwalk" and density_ind == 2:
                num_qubits = np.array(range(5, 10))
            else:
                num_qubits = np.array(range(5, 12))
            num_amplitudes = [densities_all[density_ind](n) for n in num_qubits]
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, density_ind, method_ind, "_nolabel_", figure_id)

    circle_marker = Line2D([0], [0], linestyle="", color="k", marker="o", markersize=5, label="Gleinig Walks")
    star_marker = Line2D([0], [0], linestyle="", color="k", marker="*", markersize=8, label="Gleinig")
    blue_line = Line2D([0], [0], color="b", label=r"$m = n$")
    red_line = Line2D([0], [0], color="r", label=r"$m = n^2$")
    green_line = Line2D([0], [0], color="g", label=r"$m = 2^{n-1}$")
    plt.legend(handles=[circle_marker, star_marker, blue_line, red_line, green_line])
    save_figure()

def plot_mhs_soa_comparison():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    # plot_cx_count_vs_num_qubits_line_linear("mhs_linear", num_qubits, num_amplitudes, 0, 0, "MHS Linear", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("mhs_nonlinear", num_qubits, num_amplitudes, 0, 0, "MHS Nonlinear", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("merging_states", num_qubits, num_amplitudes, 1, 0, "Merging States", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("qiskit", num_qubits, num_amplitudes, 2, 0, "Qiskit", figure_id)
    save_figure()

def plot_mhs_soa_comparison2():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits**2
    figure_id = 0
    # plot_cx_count_vs_num_qubits_line_fit("mhs_linear", num_qubits, num_amplitudes, 0, 0, "MHS Linear", figure_id)
    plot_cx_count_vs_num_qubits_line("mhs_nonlinear", num_qubits, num_amplitudes, 0, 0, "MHS Nonlinear", figure_id)
    plot_cx_count_vs_num_qubits_line("merging_states", num_qubits, num_amplitudes, 1, 0, "Merging States", figure_id)
    plot_cx_count_vs_num_qubits_line("qiskit", num_qubits, num_amplitudes, 2, 0, "Qiskit", figure_id)

    save_figure()

def plot_walks_comparison():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    plot_cx_count_vs_num_qubits_line_linear("random_reduced", num_qubits, num_amplitudes, 10, 0, "Random", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("linear_reduced", num_qubits, num_amplitudes, 7, 0, "Sorted", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("shp_reduced", num_qubits, num_amplitudes, 8, 0, "SHP", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("mst_reduced", num_qubits, num_amplitudes, 9, 0, "MST", figure_id)
    # plot_cx_count_vs_num_qubits_line_linear("merging_states", num_qubits, num_amplitudes, 2, 0, "Merging States", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("greedy_insertion_ordered", num_qubits, num_amplitudes, 4, 0, "Greedy(Sorted)", figure_id)
    # plot_cx_count_vs_num_qubits_line_linear("greedy_insertion_mhs", num_qubits, num_amplitudes, 4, 0, "Greedy(MHS)", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("greedy_insertion_ordered_combined", num_qubits, num_amplitudes, 6, 0, "Greedy(Sorted) Combined", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("mhs_linear", num_qubits, num_amplitudes, 3, 0, "MHS Linear", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("mhs_nonlinear", num_qubits, num_amplitudes, 0, 0, "MHS Nonlinear", figure_id)
    plot_cx_count_vs_num_qubits_line_linear("greedy_insertion_mhs_combined", num_qubits, num_amplitudes, 5, 0, "Greedy(MHS) Combined", figure_id)

    # plot_cx_count_vs_num_qubits_line_linear("greedy_insertion_mhs", num_qubits, num_amplitudes, 5, 0, "Greedy(MHS)", figure_id)
    
    save_figure()


def _get_avg(method, num_qubits, num_amplitudes):
    data=[]
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        column_dat=df[method]
        data.append(np.mean(column_dat))
    return data

def averages_mhs_merging_comparison():
    num_qubits=np.array((range(5,12)))
    num_amplitudes=num_qubits**2
    methods=["mhs_nonlinear", "merging_states"]
    data=[]
    for method in methods:
        data.append(_get_avg(method, num_qubits, num_amplitudes))

    return [data[1][idx]-data[0][idx] for idx in range(len(data[0]))]


if __name__ == "__main__":
    # plot_control_reduction_effect()
    # plot_walks_comparison()
    # plot_mhs_soa_comparison()
    # plot_mhs_soa_comparison2()
    # plot_greedy_gleinig_comparison()

    # plt.show()

    num_qubits=np.array((range(5,12)))
    num_amplitudes=num_qubits**2
    method="mhs_linear"
    # method="mhs_nonlinear"
    # method="merging_states"
    # method="shp_reduced"
    # method="greedy_insertion_mhs_combined"
    # method="greedy_insertion_ordered_combined"

    print(_get_avg(method, num_qubits, num_amplitudes))
    # print(averages_mhs_merging_comparison())
