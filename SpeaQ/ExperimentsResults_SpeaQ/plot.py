import json
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import dataframe_image as dfi
import os


def get_column_name(folder):
    def get_scale(folder):
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
        return "Similar not RL"
    elif "_in" in folder:
        return "Similar in RL"


def plot(columns, data, file_name):
    recall_df = pd.DataFrame(columns=columns, data=data)

    dfi.export(
        recall_df.style.highlight_max(
            axis=1,
            color="lightgreen",
            subset=[col for col in columns if col != "Metric"],
        )
        .highlight_min(
            axis=1,
            color="salmon",
            subset=[col for col in columns if col != "Metric"],
        )
        .hide(axis="index"),
        file_name,
    )


def extract_from_line(string):
    string = string.split("copypaste: ")[-1]
    return [value.strip() for value in string.split(",")]


def extract_data(file_name):
    with open(file_name, "r") as f:
        data = f.readlines()
        recall_metrics = extract_from_line(data[-2])
        recall_values = [float(value) for value in extract_from_line(data[-1])]
        ap_metrics = extract_from_line(data[-5])
        ap_values = [float(value) for value in extract_from_line(data[-4])]

        return recall_metrics, recall_values, ap_metrics, ap_values


def main():

    columns = ["Metric"]
    all_recall_values = []
    all_ap_values = []

    for folder in os.listdir("."):
        if os.path.exists(os.path.join(folder, "log.txt")):
            file_name = os.path.join(folder, "log.txt")
        elif os.path.exists(os.path.join(folder, "log(1).txt")):
            file_name = os.path.join(folder, "log(1).txt")
        else:
            continue
        columns.append(get_column_name(folder))
        recall_metrics, recall_values, ap_metrics, ap_values = extract_data(file_name)
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

    plot(columns, all_ap_values, "plots/speaq_ap.png")
    plot(columns, all_recall_values, "plots/speaq_recall.png")


if __name__ == "__main__":
    main()
