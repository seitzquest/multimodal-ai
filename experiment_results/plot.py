import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_column_name(folder):
    """
    Retrieves dataframe column name from the folder name
    """

    def get_scale(folder):
        """
        If scale if in the folder name, retrieves it
        """
        if "2" in folder:
            return "0.2"
        elif "5" in folder:
            return "0.5"
        elif "7" in folder:
            return "0.7"

    if "_trained" in folder:
        return "Seen " + get_scale(folder)
    elif "untrained" in folder:
        return "Unseen " + get_scale(folder)
    elif "unlikely" in folder:
        return "Unlikely"
    elif "origin" in folder:
        return "Origin"
    elif "Notin" in folder:
        return "SimNotRL"
    elif "_in" in folder:
        return "SimInRL"


def extract_from_line(string):
    """
    Extracts evaluation values from a line
    """
    string = string.split("copypaste: ")[-1]
    return [value.strip() for value in string.split(",")]


def extract_data_reitr(file_name):
    """
    Extracts evaluation data from log file
    """
    precision_metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    recall_metrics = ["MR@20", "MR@50", "MR@100"]
    with open(file_name, "r") as f:
        data = f.readlines()
        recall_values = []
        precision_values = []
        for row in data:
            if "Average Precision  (AP)" in row:
                precision_values.append(float(row.split(" ")[-1].strip()))
            elif "mR@" in row:
                recall_values.append(float(row.split(" ")[-1].strip()))
    return recall_metrics, recall_values, precision_metrics, precision_values


def extract_data_speaq(file_name):
    """
    Extracts evaluation data from log file
    """
    with open(file_name, "r") as f:
        data = f.readlines()
        recall_metrics = extract_from_line(data[-2])
        recall_values = [float(value) for value in extract_from_line(data[-1])]
        ap_metrics = extract_from_line(data[-5])
        ap_values = [float(value) for value in extract_from_line(data[-4])]
        recall_values = recall_values[:3]
        recall_metrics = recall_metrics[:3]
        recall_metrics = [
            metric[2:].replace("Mean", "M").replace("Recall", "R")
            for metric in recall_metrics
        ]
        return recall_metrics, recall_values, ap_metrics, ap_values


def save_latex_txt(columns, data, file_name):
    """
    Saves data in a text file in latex format
    """

    table = ""
    for index, column in enumerate(columns):
        table += column
        if index != len(columns) - 1:
            table += " & "
        else:
            table += " \\\\\n\\hline\n"
    for row in data:
        min_value = min([value for value in row if isinstance(value, float)])
        max_value = max([value for value in row if isinstance(value, float)])

        for index, value in enumerate(row):
            if value == min_value:
                table += "\\cellcolor[HTML]{f69f99} "
            if value == max_value:
                table += "\\cellcolor[HTML]{ccf699} "
            table += str(value)
            if index != len(row) - 1:
                table += " & "
            else:
                table += " \\\\\n"

    with open(file_name, "w") as f:
        f.write(table)


def extract_and_plot(result_folder, model):
    """
    Reads evaluation results from the log file and saves it to latex table
    """
    columns = ["Metric"]
    all_recall_values = []
    all_ap_values = []
    for folder in os.listdir(result_folder):
        if os.path.exists(os.path.join(result_folder, folder, "log.txt")):
            file_name = os.path.join(result_folder, folder, "log.txt")
        elif os.path.exists(os.path.join(result_folder, folder, "log(1).txt")):
            file_name = os.path.join(result_folder, folder, "log(1).txt")
        else:
            continue
        columns.append(get_column_name(folder))
        if model == "SpeaQ":
            recall_metrics, recall_values, ap_metrics, ap_values = extract_data_speaq(
                file_name
            )
            ap_values = [round(ap_value / 100, 4) for ap_value in ap_values]
        else:
            recall_metrics, recall_values, ap_metrics, ap_values = extract_data_reitr(
                file_name
            )
        recall_values = [round(recall_value, 4) for recall_value in recall_values]
        all_recall_values.append(recall_values)
        all_ap_values.append(ap_values)

    all_recall_values = list(map(list, zip(*all_recall_values)))
    all_ap_values = list(map(list, zip(*all_ap_values)))
    all_recall_values = [
        [metric] + value for metric, value in zip(recall_metrics, all_recall_values)
    ]

    all_ap_values = [
        [metric] + value for metric, value in zip(ap_metrics, all_ap_values)
    ]

    save_latex_txt(columns, all_ap_values, f"plots/{model.lower()}_ap.txt")
    save_latex_txt(columns, all_recall_values, f"plots/{model.lower()}_recall.txt")

    return columns, all_ap_values, all_recall_values


def create_heat_map(experiments, values1, values2, file_path):
    """
    Plot and save a heat map of RelTR vs SpeaQ
    """
    metrics = [value[0] for value in values1]
    values1 = [values[1:] for values in values1]
    values2 = [values[1:] for values in values2]

    coefficients = np.divide(values1, values2)
    plt.figure(figsize=(len(coefficients[0]), len(coefficients) + 1))
    sns.heatmap(
        coefficients,
        xticklabels=experiments[1:],
        yticklabels=metrics,
        annot=True,
        cmap="viridis",
        fmt=".2f",
    )
    plt.title("SpeaQ / RelTR")
    plt.xlabel("Experiments")
    plt.ylabel("Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_path)


def main():
    columns, reltr_ap_values, reltr_recall_values = extract_and_plot(
        "ExperimentsResults_RelTR", "RelTR"
    )

    columns, speaq_ap_values, speaq_recall_values = extract_and_plot(
        "ExperimentsResults_SpeaQ", "SpeaQ"
    )

    create_heat_map(columns, speaq_ap_values, reltr_ap_values, "plots/heatmap_ap.png")
    create_heat_map(
        columns, speaq_recall_values, reltr_recall_values, "plots/heatmap_recall.png"
    )


if __name__ == "__main__":
    main()
