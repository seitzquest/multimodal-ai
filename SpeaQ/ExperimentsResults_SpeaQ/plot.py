import os
import pandas as pd


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


def save_csv(columns, data, file_name):
    """
    Creates a dataframe and saves it as csv
    """
    recall_df = pd.DataFrame(columns=columns, data=data)
    recall_df.to_csv(file_name, index=False)


def extract_from_line(string):
    """
    Extracts evaluation values from a line
    """
    string = string.split("copypaste: ")[-1]
    return [value.strip() for value in string.split(",")]


def extract_data(file_name):
    """
    Extracts evaluation data from log file
    """
    with open(file_name, "r") as f:
        data = f.readlines()
        recall_metrics = extract_from_line(data[-2])
        recall_values = [float(value) for value in extract_from_line(data[-1])]
        ap_metrics = extract_from_line(data[-5])
        ap_values = [float(value) for value in extract_from_line(data[-4])]

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


def main():
    """
    Reads evaluation results from the log file and saves it to csv
    """
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
        ap_values = [round(ap_value / 100, 4) for ap_value in ap_values]
        recall_values = [round(recall_value, 4) for recall_value in recall_values]
        all_recall_values.append(recall_values)
        all_ap_values.append(ap_values)

    recall_metrics = [
        metric[2:].replace("Mean", "M").replace("Recall", "R")
        for metric in recall_metrics
    ]
    all_recall_values = list(map(list, zip(*all_recall_values)))
    all_ap_values = list(map(list, zip(*all_ap_values)))
    all_recall_values = [
        [metric] + value for metric, value in zip(recall_metrics, all_recall_values)
    ]
    all_ap_values = [
        [metric] + value for metric, value in zip(ap_metrics, all_ap_values)
    ]

    save_csv(columns, all_ap_values, "plots/speaq_ap.csv")
    save_csv(columns, all_recall_values, "plots/speaq_recall.csv")
    save_latex_txt(columns, all_ap_values, "plots/speaq_ap.txt")
    save_latex_txt(columns, all_recall_values, "plots/speaq_recall.txt")


if __name__ == "__main__":
    main()
