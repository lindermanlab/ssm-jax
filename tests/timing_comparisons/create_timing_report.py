"""
Utility functions to create a timing report.
"""

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob

plt.style.use("ggplot")

JAX_BENCHMARK_DIR = "./.benchmarks/Linux-CPython-3.9-64bit"
SSM_V0_BENCHMARK_DIR = "./ssm_v0_benchmark_tests/.benchmarks/Linux-CPython-3.9-64bit"


def parse_json_results_to_df(json_results_fname):
    """Extract useful information from pytest-benchmark json output format.
    Returns a dataframe.
    """

    # read in json from fname
    with open(json_results_fname, "r") as f:
        json_data = json.load(f)

    # construct dataframe
    df = pd.DataFrame(json_data["benchmarks"])
    df = pd.concat([df, df["stats"].apply(pd.Series)], axis=1)
    df["test_name"] = df["name"].apply(lambda x: x.split("[")[0])
    df["param_name"] = df["params"].apply(lambda x: list(x.keys())[0])
    df["param_value"] = df["params"].apply(lambda x: list(x.values())[0])

    # select columns to keep
    columns = [
        "test_name",
        "param_name",
        "param_value",
        "min",
        "max",
        "mean",
        "median",
        "stddev",
        "ops",
        "iterations",
        "rounds",
        "extra_info",
        "params",
    ]
    df = df[columns]
    return df


def plot_series(x, y, color="C0", label=""):
    """Helper function to plot a series of y vs. x.
    Adds a linear trend line and handles NaNs in the runtime.
    """
    params = np.array(x)
    mean_times = np.array(y)
    isnan = np.isnan(mean_times)
    if np.sum(~isnan) > 1:
        m, b = np.polyfit(params[~isnan], mean_times[~isnan], deg=1)
        plt.plot(params, m * params + b, "--", color=color, label=label)
    else:
        m = 0
        b = mean_times[~isnan]
        plt.plot([], [], "--", color=color, label=label)
    ys = np.copy(mean_times)
    # ys[isnan] = m * params[isnan] + b  # replace nans with linear prediction
    ys[isnan] = 0.  # replace nans with zero
    for x, y, isna in zip(params, ys, isnan):
        plt.scatter(
            x, y, marker="x" if isna else "o", s=80 if isna else 40, color=color
        )


def load_multiple_runs(benchmark_runs):
    """Convenience function to load multiple bechmark runs from json and
    return as a single dataframe (with a `run_name`) identifer column.
    """
    runs = []
    for run_name, run_fname in benchmark_runs.items():
        df = parse_json_results_to_df(run_fname)
        cols = df.columns
        cols = cols.insert(0, "run_name")
        df["run_name"] = run_name
        df = df[cols]
        runs.append(df)
    df = pd.concat(runs)
    return df


def generate_report_across_params(
    runs_df, show_plots=False, save_to_pdf: bool or str = False,
    out_units="seconds",
):
    """Main function to generate plotting report of runtime versus params

    Note that different runs are plotted as different series on the plot.
    This is best for comparing how runtime scales with different param values
    across the different runs.
    """
    
    if out_units == "seconds":
        unit_factor = 1.0
    elif out_units == "minutes":
        unit_factor = 60.0
    else:
        raise ValueError(f"out_units {out_units} not recognized.")
        
    colors = [plt.cm.tab20(i) for i in range(20)]
    if save_to_pdf:
        pdf = PdfPages(save_to_pdf)
    for test_name, test_grp in runs_df.groupby(["test_name"]):
        plt.figure(figsize=(10, 6))
        # plot time vs. params for different runs (as different series)
        for i, (run_name, run_grp) in enumerate(test_grp.groupby(["run_name"])):
            plot_series(
                run_grp["param_value"],          # param
                run_grp["median"] / unit_factor, # runtime
                label=run_name,
                color=colors[i % len(colors)],
            )
        plt.title(test_name)
        plt.xlabel(run_grp["param_name"].iloc[0])
        plt.ylabel(f"time ({out_units})")
        ax = plt.gca()
        box = ax.get_position()  # Shink current axis by 20% to fit legend in pdf
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1, 1))
        if save_to_pdf:
            pdf.savefig()
        if show_plots:
            plt.show()
        plt.close()
    if save_to_pdf:
        pdf.close()


def generate_report_across_runs(
    runs_df, show_plots=False, save_to_pdf: bool or str = False
):
    """Main function to generate plotting report directly comparing runtime across runs
    (with params fixed).
    """
    colors = [plt.cm.tab20(i) for i in range(20)]
    if save_to_pdf:
        pdf = PdfPages(save_to_pdf)
    for i, (test_name, test_grp) in enumerate(runs_df.groupby(["test_name"])):
        for param_value, param_grp in test_grp.groupby(["param_value"]):
            plt.figure(figsize=(10, 6))
            run_names = param_grp["run_name"][::-1]
            height = np.nan_to_num(np.array(param_grp["mean"]), -1)[::-1]
            plt.bar(x=run_names, height=height, label=param_grp, color=colors)
            plt.title(f"{test_name} [{param_value}]")
            plt.xticks(rotation=45)
            plt.ylabel("time (s)")
            ax = plt.gca()
            box = ax.get_position()  # Shink current axis by 20% to fit xlabels in pdf
            ax.set_position(
                [box.x0, box.y0 + (box.height * 0.2), box.width, box.height * 0.8]
            )
            if save_to_pdf:
                pdf.savefig()
            if show_plots:
                plt.show()
            plt.close()
    if save_to_pdf:
        pdf.close()


def get_jax_benchmark_files():
    """Helper to get JAX benchmark json files in sorted order."""
    return sorted(glob.glob(f"{JAX_BENCHMARK_DIR}/*.json"))


def get_ssm_v0_benchmark_files():
    """Helper to get SSM_v0 benchmark json files in sorted order."""
    return sorted(glob.glob(f"{SSM_V0_BENCHMARK_DIR}/*.json"))


if __name__ == "__main__":

    # get JAX benchmark files into dict (file number --> full path)
    jax_fnames = get_jax_benchmark_files()
    benchmark_runs = {fname.split("/")[-1].split("_")[0]: fname for fname in jax_fnames}

    # get latest SSM_v0 file and add to dict
    benchmark_runs["ssm_v0"] = get_ssm_v0_benchmark_files()[-1]

    runs_df = load_multiple_runs(benchmark_runs)
    generate_report_across_params(
        runs_df, save_to_pdf="timing_report_A.pdf", show_plots=False
    )
    generate_report_across_runs(
        runs_df, save_to_pdf="timing_report_B.pdf", show_plots=False
    )
