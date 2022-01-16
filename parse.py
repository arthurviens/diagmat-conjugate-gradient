import pandas as pd

import os
import re

columns = ["nbcores", "rep", "solver", "preconditionner", "time"]

ssor = False


def parse_file(logs_dir, filename):
    print(f"Parsing {filename}")

    splitted = filename.split(".")[0]
    splitted = splitted.split("_")
    np = splitted[0][2:]
    rep = splitted[1][3:]
    solver = splitted[2][6:]

    if "m" in filename: # No preconditionning
        m = splitted[3][1:]
    else:
        m = None
    #print(f"np {np}, rep {rep}, solver {solver}, m {m}")

    with open(os.path.join(logs_dir, filename), "r") as f:
        lines = pd.Series(f.readlines())
    line = lines[lines.str.startswith("That took")].values[0]
    t = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0])
    series = pd.Series(name=filename, index=columns, data=[np, rep, solver, m, t])
    #print(series)

    return series


def parse_file_ssor(logs_dir, filename, columns):
    print(f"Parsing {filename}")
    columns = columns + ["omega"]

    splitted = filename[:-4]
    splitted = splitted.split("_")
    np = splitted[0][2:]
    rep = splitted[1][3:]
    solver = splitted[2][6:]
    m = splitted[3][1:]
    omega = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", splitted[4])[0])


    #print(f"np {np}, rep {rep}, solver {solver}, m {m}, omega {omega}")

    with open(os.path.join(logs_dir, filename), "r") as f:
        lines = pd.Series(f.readlines())
    line = lines[lines.str.startswith("That took")].values[0]
    t = float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0])
    series = pd.Series(name=filename, index=columns, data=[np, rep, solver, m, t, omega])
    #print(series)

    return series



if __name__ == "__main__":
    logs_dir = "logs/"
    files = os.listdir(logs_dir)
    df = pd.DataFrame()
    for file in files:
        if ssor:
            series = parse_file_ssor(logs_dir, file, columns)
        else :
            series = parse_file(logs_dir, file)
        df = df.append(series)
    if ssor:
        df.reset_index(drop=False).to_csv("log_results_ssor.csv", index=False, sep=";")
    else:
        df.reset_index(drop=False).to_csv("log_results.csv", index=False, sep=";")
