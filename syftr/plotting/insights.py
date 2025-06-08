import json
import re
import tempfile
import textwrap

import altair as alt
import dataframe_image as dfi
import diskcache
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import optuna
import pandas as pd
import paretoset
import seaborn as sns
import tqdm
from adjustText import adjust_text
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from slugify import slugify

from syftr.configuration import cfg
from syftr.llm import AZURE_GPT4O_STD
from syftr.studies import get_response_synthesizer_llm, get_template_name

SHOW_TITLE = False
CACHE = diskcache.Cache(".insights_cache/")


def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Running: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def is_cost_objective(data: pd.DataFrame):
    assert data is not None, "Data must be provided."
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
    assert not data.empty, "Data must not be empty."
    objective_2_name = data.iloc[0]["user_attrs_metric_objective_2_name"]
    print(f"data has {len(data)} rows")
    for i in range(len(data) - 1):
        if pd.isna(objective_2_name):
            objective_2_name = data.iloc[i + 1]["user_attrs_metric_objective_2_name"]
        obj_2_name = data.iloc[i + 1]["user_attrs_metric_objective_2_name"]
        if (not pd.isna(obj_2_name)) and (obj_2_name != objective_2_name):
            print("Multiple objective 2 names found in the data.")
            return None
    if objective_2_name == "p80_time":
        return False
    return True


def is_retriever_study(data: pd.DataFrame):
    """Check if the study is a retriever study based on the presence of specific parameters."""
    assert data is not None, "Data must be provided."
    assert isinstance(data, pd.DataFrame), "Data must be a DataFrame"
    assert not data.empty, "Data must not be empty."
    objective_1_name = data.iloc[0]["user_attrs_metric_objective_1_name"]
    if objective_1_name == "retriever_recall":
        return True
    return False


def get_objective_2_name(data: pd.DataFrame = None, is_cost: bool | None = None):
    if is_cost is None:
        if data is None or data.empty:
            return "Objective 2"
        if is_cost_objective(data):
            return "Cost"
        return "Latency"
    if is_cost:
        return "Cost"
    return "Latency"


def get_objective_2_unit(data: pd.DataFrame = None, is_cost: bool | None = None):
    if is_cost is None:
        if data is None:
            return "Unit"
        if is_cost_objective(data):
            return "¢ per 100 calls"
        return "seconds"
    if is_cost:
        return "¢ per 100 calls"
    return "seconds"


discrete_cmap = plt.get_cmap("tab20")
baseline_cmap = plt.get_cmap("copper")
continuous_cmap = plt.get_cmap("copper_r")
grey_face_color = "0.9"
grey_edge_color = "0.7"
fore_color = "0.0"

normal_to_red = LinearSegmentedColormap.from_list(
    "normal_to_red", ["white", "red", "darkred"]
)
normal_to_yellow = LinearSegmentedColormap.from_list(
    "normal_to_green", ["white", "gold", "goldenrod"]
)
normal_to_green = LinearSegmentedColormap.from_list(
    "normal_to_green", ["white", "green", "darkgreen"]
)
normal_to_bad = LinearSegmentedColormap.from_list(
    "normal_to_bad", ["white", "orange", "red", "darkred"]
)
normal_to_good = LinearSegmentedColormap.from_list(
    "normal_to_good", ["white", "gold", "green", "darkgreen"]
)

red_to_normal = normal_to_red.reversed()
yellow_to_normal = normal_to_yellow.reversed()
green_to_normal = normal_to_green.reversed()
bad_to_normal = normal_to_bad.reversed()
good_to_normal = normal_to_good.reversed()


def safe_paretoset(df: pd.DataFrame, sense):
    """Handle nan and inf values when computing the Pareto-frontier."""
    assert df.shape[1] == 2
    df = df.replace([np.inf, -np.inf], np.nan)
    if len(df.dropna(how="any")) == 0:
        return pd.Series(False, index=df.index)
    pareto = paretoset.paretoset(df, sense=sense, use_numba=False)
    pareto = pd.Series(pareto, index=df.index)
    return pareto


def descriptive_name(param_col, is_cost=None):
    """Returns a human-readable description of a parameter column."""
    column_name_map = {
        "values_0": "Accuracy (%)",
        "values_1": f"{get_objective_2_name(is_cost=is_cost)} ({get_objective_2_unit(is_cost=is_cost)})",
        "study_name": "Study Name",
        "trial_minutes": "Build+Evaluate Time",
        "pareto": "New Pareto Trial",
        "pruned_eval": "Evaluations Pruned",
        "params_no_rag_llm_name": "No RAG Response Synthesizer LLM",
        "params_additional_context_enabled": "Neighboring Retrieval Results Enabled",
        "params_additional_context_num_nodes": "Number of Neighboring Retrieval Results",
        "params_few_shot_enabled": "Dynamic Few-shot Example Retrieval",
        "params_few_shot_embedding_model": "Embedding Model for Few-Shot Retrieval",
        "params_few_shot_method": "Retrieval Method for Dynamic Few-Shot Retrieval",
        "params_few_shot_top_k": "Number of Examples Used in Few-Shot Retrieval",
        "params_hyde_enabled": "HyDE Retrieval Enabled in RAG",
        "params_hyde_llm_name": "HyDE Retrieval Generative Model",
        "params_rag_embedding_model": "RAG Embedding Model",
        "params_rag_fusion_llm_name": "LLM for RAG Fusion Retriever",
        "params_rag_fusion_mode": "Result Merge Strategy for RAG Fusion Retriever",
        "params_rag_fusion_num_queries": "Number of Prompt Queries for RAG Fusion Retriever",
        "params_rag_method": "RAG Retriever Type",
        "params_rag_mode": "RAG Strategy",
        "params_rag_top_k": "Number of RAG Retrieval Results",
        "params_reranker_enabled": "RAG Reranker Enabled",
        "params_reranker_llm_name": "RAG Reranker LLM",
        "params_reranker_top_k": "Max Results from RAG Reranker",
        "params_splitter_chunk_overlap_frac": "RAG Chunk Overlap Fraction",
        "params_splitter_chunk_exp": "RAG Chunk Size Exponent",
        "params_splitter_method": "RAG Splitter Type",
        "params_rag_query_decomposition_enabled": "Query Decomposition for RAG",
        "params_rag_query_decomposition_num_queries": "Number of Queries to generate for RAG",
        "params_rag_query_decomposition_llm_name": "LLM for Query Decomposition",
        "params_few_shot_query_decomposition_enabled": "Query Decomposition for Few-Shot Retrieval",
        "params_rag_hybrid_bm25_weight": "Relative weight for BM25 results in Hybrid retriever",
        # old param name
        "params_splitter_chunk_size": "RAG Chunk Size",
        # old param names (bench8 or before December 12, 2024)
        "params_chunk_overlap": "RAG Chunk Overlap Count",
        "params_chunk_size": "RAG Chunk Size",
        "params_embedding_model": "RAG Embedding Model",
        "params_fusion_mode": "Result Merge Strategy for RAG Fusion Retriever",
        "params_num_queries": "Number of Prompt Queries for RAG Fusion Retriever",
        "params_reranker_top_n": "Max Results from RAG Reranker",
        "params_retriever": "RAG Retriever Type",
        "params_similarity_top_k": "Number of RAG Retrieval Results",
        "params_splitter": "RAG Splitter Type",
        "params_use_hyde": "HyDE Retrieval Enabled in RAG",
        "params_use_reranker": "RAG Reranker Enabled",
        "params_agent_retriever": "Agent Retriever Type",
        # new names after search space restructuring
        "params_critique_rag_agent_critique_agent_llm": "Critique Agent LLM",
        "params_critique_rag_agent_reflection_agent_llm": "Reflection Agent LLM",
        "params_critique_rag_agent_response_synthesizer_llm": "Critique RAG Agent Response Synthesizer LLM",
        "params_critique_rag_agent_subquestion_engine_llm": "Subquestion Engine LLM",
        "params_critique_rag_agent_subquestion_response_synthesizer_llm": "Subquestion Response Synthesizer LLM",
        "params_critique_rag_agent_template_name": "Critique Agent Template",
        "params_lats_rag_agent_max_rollouts": "LATS Agent Max Rollouts",
        "params_lats_rag_agent_num_expansions": "LATS Agent Num Expansions",
        "params_lats_rag_agent_response_synthesizer_llm": "LATS Agent Response Synthesizer LLM",
        "params_lats_rag_agent_template_name": "LATS Agent Template",
        "params_no_rag_response_synthesizer_llm": "No RAG Response Synthesizer LLM",
        "params_no_rag_template_name": "No RAG Template",
        "params_rag_response_synthesizer_llm": "RAG Response Synthesizer LLM",
        "params_rag_template_name": "RAG Template",
        "params_react_rag_agent_response_synthesizer_llm": "ReAct RAG Agent Response Synthesizer LLM",
        "params_react_rag_agent_subquestion_engine_llm": "ReAct RAG Agent Subquestion Engine LLM",
        "params_react_rag_agent_subquestion_response_synthesizer_llm": "ReAct RAG Agent Subquestion Response Synthesizer LLM",
        "params_react_rag_agent_template_name": "ReAct RAG Agent Template",
        "params_sub_question_rag_response_synthesizer_llm": "Subquestion RAG Response Synthesizer LLM",
        "params_sub_question_rag_subquestion_engine_llm": "Subquestion RAG Subquestion Engine LLM",
        "params_sub_question_rag_subquestion_response_synthesizer_llm": "Subquestion RAG Subquestion Response Synthesizer LLM",
        "params_sub_question_rag_template_name": "Subquestion RAG Template",
        # recombined parameter colums
        "params_llm_name": "Response Synthesizer LLM",
        "params_sub_question_engine_llm": "Subquestion Engine LLM",
        "params_sub_question_response_synthesizer_llm": "Subquestion Response Synthesizer LLM",
        "params_template_name": "Prompt Template",
    }
    if isinstance(param_col, str):
        if param_col not in column_name_map:
            print(
                f'WARNING: Please add a descriptive title for column "{param_col}" inside the descriptive_name() function.'
            )
            return param_col.replace("params_", "").replace("_", " ").title()
        return column_name_map[param_col]
    else:
        return " x ".join([descriptive_name(col, is_cost) for col in param_col])


def get_name(study_name: str, titles=None) -> str:
    if titles is not None:
        if study_name in titles:
            return titles[study_name]

    name = ""
    if "synthetic" in study_name:
        name = "Synthetic "

    if "financebench" in study_name:
        name += "FinanceBench"
    elif "hotpot" in study_name:
        name += "HotpotQA"
    elif "infinit" in study_name:
        name += "InfiniteBench"
    elif "drdocs" in study_name:
        name += "DRDocs"
    elif "crag-" in study_name:
        name += "CRAG3 "
        name += study_name.split("-")[-1].capitalize()
    else:
        print(f"WARNING: please add '{study_name}' to the benchmark_name() function.")
        if "--" in study_name:
            name += study_name.split("--")[-1].capitalize()
        elif "-" in study_name:
            name += study_name.split("-")[-1].capitalize()
        else:
            name += study_name
    return name


def groupby_to_series(df: pd.DataFrame, by: list[str] | str, func):
    n_groups = df.groupby(by).ngroups
    if n_groups > 1:
        return df.groupby(by, group_keys=False).apply(func, include_groups=False)
    else:
        return func(df)


def download_study(study_name: str) -> pd.DataFrame:
    cache_key = ("download_study", study_name, "v0")
    if cache_key in CACHE:
        return CACHE[cache_key]

    study = optuna.load_study(
        study_name=study_name, storage=cfg.database.get_optuna_storage()
    )
    df: pd.DataFrame = study.trials_dataframe()
    if len(df) == 0:
        raise ValueError(f"Study '{study_name}' contains no trials")

    # clean-up datetimes
    df["duration"] = pd.to_timedelta(df["duration"])
    df["datetime_start"] = pd.to_datetime(df["datetime_start"], utc=True)
    df["datetime_complete"] = pd.to_datetime(df["datetime_complete"], utc=True)

    def my_to_datetime(x):
        try:
            return pd.Timestamp.fromtimestamp(x, tz="utc") if np.isfinite(x) else pd.NaT
        except (TypeError, ValueError, OSError):
            print(f"ERROR: Failed to parse timestamp: {x}")
            return pd.NaT

    if "user_attrs_metric_flow_duration" in df.columns:
        df["duration"] = pd.to_timedelta(
            df["user_attrs_metric_flow_duration"], unit="s"
        )
    if "user_attrs_metric_flow_start" in df.columns:
        df["datetime_start"] = df["user_attrs_metric_flow_start"].apply(my_to_datetime)
    if "user_attrs_metric_flow_end" in df.columns:
        df["datetime_complete"] = df["user_attrs_metric_flow_end"].apply(my_to_datetime)

    # cache and return
    time_since_last = pd.Timestamp.now(tz="utc") - df["datetime_complete"].max()
    if time_since_last > pd.Timedelta(days=7):
        expire = 60 * 60 * 24 * 7
    elif time_since_last > pd.Timedelta(days=3):
        expire = 60 * 60 * 24
    else:
        # expire = 60 * 15
        expire = 60 * 60 * 4
    CACHE.set(cache_key, df, expire=expire)
    return df


def recombine_split_parameters(
    df: pd.DataFrame, get_func, source_cols: list[str], new_col: str
):
    # combine subquestion LLM into a single parameter
    if new_col not in df.columns:
        # make sure the underlying cols exist
        df = df.copy()
        for col in source_cols:
            if col not in df.columns:
                df[col] = np.nan

        # fill in the new column and delete the source cols
        df[new_col] = df.apply(get_func, axis=1)
        df = df.drop(columns=source_cols)
    return df


def set_from_flow(df: pd.DataFrame):
    for index, row in df.iterrows():
        if isinstance(row["user_attrs_flow"], str):
            flow = json.loads(row["user_attrs_flow"])
            params_flow = {"params_" + param: value for param, value in flow.items()}
            for param in params_flow:
                if params_flow[param] != np.nan:
                    if param not in df.columns:
                        inferred_type = type(params_flow[param])
                        if inferred_type is float:
                            df[param] = pd.Series(dtype="float64")
                        elif inferred_type is int:
                            df[param] = pd.Series(dtype="int64")
                        elif inferred_type is str:
                            df[param] = pd.Series(dtype="string")
                        elif inferred_type is bool:
                            df[param] = pd.Series(dtype="bool")
                        else:
                            df[param] = pd.Series(dtype="object")
                    df.loc[index, param] = params_flow[param]


def load_study(study_name: str) -> pd.DataFrame:
    # load the study
    df: pd.DataFrame = download_study(study_name)
    set_from_flow(df)

    # ensure columns "values_0" and "values_1" are float dtypes
    df["values_0"] = df["values_0"].astype(float)
    df["values_1"] = df["values_1"].astype(float)

    is_cost = is_cost_objective(df)
    if is_cost:
        df["values_1"] = 100 * 100 * df["values_1"]

    # clean-up unused parameters to be N/A
    if "params_retriever" in df.columns:
        df.loc[df["params_retriever"] == "sparse", "params_embedding_model"] = np.nan

    # clean-up missing trials by re-numbering them
    df.sort_values(by="datetime_complete", inplace=True)
    df["number"] = range(1, len(df) + 1)

    # clean up 0% accuracy bug on dense retriever for a few runs
    if study_name in ["run9", "run10"]:
        df = df[df["values_0"] > 0]

    # remember which study this is
    df["study_name"] = study_name

    # pre-calculate some useful columns
    df["compute_hours"] = (df["duration"].dt.total_seconds() / 3600).cumsum()
    df["wallclock_hours"] = (
        df["datetime_complete"] - df["datetime_start"].min()
    ) / pd.Timedelta(hours=1)

    # ensure exception columns exist
    for col in [
        "user_attrs_metric_exception_class",
        "user_attrs_metric_exception",
        "user_attrs_metric_exception_message",
    ]:
        if col not in df.columns:
            df[col] = "None"

    # ensure run status columns exist
    for col in [
        "user_attrs_is_seeding",
        "user_attrs_metric_is_pruned",
        "user_attrs_metric_failed",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # ensure error counts exist
    for col in ["user_attrs_metric_num_errors", "user_attrs_metric_num_total"]:
        if col not in df.columns:
            df[col] = 0

    # clean-up synthesizer parameters
    if (
        "params_no_rag_response_synthesizer_llm" in df.columns
        and "params_no_rag_llm_name" in df.columns
    ):
        df["params_response_synthesizer_llm"] = df[
            "params_no_rag_response_synthesizer_llm"
        ].fillna(df["params_no_rag_llm_name"])
        df = df.drop(columns=["params_no_rag_llm_name"])
    elif "params_no_rag_llm_name" in df.columns:
        df = df.rename(
            columns={"params_no_rag_llm_name": "params_response_synthesizer_llm"}
        )

    # remove invalid data (but why are they missing?)
    # Frequent restarting left the DB with some cases of missing data that had to be excluded.
    bad_rag_mode_mas = df["params_rag_mode"].isna()
    n_bad_rag_mode = bad_rag_mode_mas.sum()
    if n_bad_rag_mode > 0:
        print(f"WARNING: {n_bad_rag_mode} trials with missing RAG mode were excluded.")
        df = df[~bad_rag_mode_mas]

    return df


@log_function_call
def create_exceptions_stats_table(df: pd.DataFrame):
    df = df.copy()

    # empty means no exception
    df["user_attrs_metric_exception_class"] = (
        df["user_attrs_metric_exception_class"].fillna("None").astype(str)
    )
    df["user_attrs_metric_exception_message"] = (
        df["user_attrs_metric_exception_message"].fillna("None").astype(str)
    )

    # truncate
    df["user_attrs_metric_exception_message"] = df[
        "user_attrs_metric_exception_message"
    ].apply(lambda x: truncate_label(x, 1000))

    # ignore some numbers in exception so they group together
    df["user_attrs_metric_exception_message"] = df[
        "user_attrs_metric_exception_message"
    ].str.replace(r"\=\d+\.\d+\b", "=[number]", regex=True)

    # count up the exceptions
    table = df.groupby(
        [
            "user_attrs_metric_exception_class",
            "user_attrs_metric_exception_message",
        ],
        dropna=False,
    ).size()

    # order and style
    table = table.sort_values(ascending=False)
    table.name = "Count"
    table = table.reset_index()
    table.columns = [col.replace("user_attrs_", "") for col in table.columns]
    return table


def load_studies(study_names=None, only_successful_trials=True):
    # load all studies by default
    if study_names is None:
        study_names = optuna.get_all_study_names(
            storage=cfg.database.get_optuna_storage()
        )
    elif isinstance(study_names, str):
        study_names = [study_names]

    df_all = []
    for study_name in tqdm.tqdm(study_names, desc="Loading Studies"):
        try:
            df_i = load_study(study_name)
            if len(df_i) == 0:
                raise ValueError(f"Study '{study_name}' contains no trials")
            df_all.append(df_i)
        except Exception as e:
            print(f"Error loading study '{study_name}': {e}")
            continue
    print("")

    df_all = pd.concat(df_all, ignore_index=True)

    # create stats before filtering to successful trials only
    study_stats_table = create_study_stats_table(df_all)
    exceptions_table = create_exceptions_stats_table(df_all)

    # filter to successful trials
    if only_successful_trials:
        df_all = df_all[df_all.state == "COMPLETE"].reset_index(drop=True)

    # filter out trials with low success rates
    if "user_attrs_metric_num_success" in df_all:
        success_percent = groupby_to_series(
            df_all,
            "study_name",
            lambda g: g["user_attrs_metric_num_success"]
            / g["user_attrs_metric_num_total"].max(),
        )
        is_pruned = groupby_to_series(
            df_all, "study_name", lambda g: g["user_attrs_metric_is_pruned"] == True
        )
        success_mask = (
            success_percent > 0.9
        ) | is_pruned  # keep pruned trials below the Pareto-frontier
        n_unsuccessful = len(df_all) - success_mask.sum()
        if n_unsuccessful > 0:
            print(
                f"!!!\nWARNING: {n_unsuccessful:,} trials with low success rates were seen by the optimizer but will be filtered out here ({n_unsuccessful / len(df_all):.0%}).\n!!!\n"
            )
            df_all = df_all[success_mask]

    # handle trials with zero cost or latency
    zero_trials = ~(df_all["values_1"] > 0)
    n_zero_trials = zero_trials.sum()
    if n_zero_trials > 0:
        print(f"WARNING: there are {n_zero_trials} trials with zero objective 2.")
        df_all.loc[zero_trials, "values_1"] = 0.000001

    # compute Pareto-frontiers after filtering
    df_all["pareto"] = groupby_to_series(
        df_all,
        "study_name",
        lambda g: safe_paretoset(g[["values_0", "values_1"]], sense=["max", "min"]),
    )

    # check that pareto trials were not pruned
    if "user_attrs_metric_num_success" in df_all:
        incomplete_pareto_trial = groupby_to_series(
            df_all,
            "study_name",
            lambda g: g["pareto"]
            & (
                g["user_attrs_metric_num_success"]
                < g["user_attrs_metric_num_total"].max()
            ),
        )
        if incomplete_pareto_trial.sum() > 0:
            print(
                f"!!!\nWARNING: {incomplete_pareto_trial.sum()} of {df_all['pareto'].sum()} pareto trials have incomplete evaluations.\n!!!"
            )

    return df_all, study_stats_table, exceptions_table


def generate(prompt):
    cache_key = ("generate", prompt)
    if cache_key in CACHE:
        return CACHE[cache_key]
    response = AZURE_GPT4O_STD.complete(prompt=prompt, temperature=0)
    CACHE.set(cache_key, response.text, expire=60 * 60 * 24 * 7)
    return response.text


def generate_trial_description(trial, is_cost):
    param_cols = [col for col in trial.index if col.startswith("params_")]
    param_strings = []
    raw_param_strings = []
    for col in param_cols:
        col_name = descriptive_name(col, is_cost)
        # if col_name.startswith("RAG") and not trial['params_to_index'] and col != 'params_to_index':
        #     continue
        if isinstance(trial[col], float) and np.isfinite(trial[col]):
            value_str = f"{trial[col]:g}"
        else:
            value_str = str(trial[col])
            if value_str in ("nan", "NaN", "None", "", "N/A", "NA"):
                value_str = "Unspecified"
                continue

        raw_param_strings.append(f"{col.replace('params_', '')}={value_str}")
        param_strings.append(f"{col_name}: {value_str}")
    param_string = "\n".join(sorted(param_strings))
    prompt = f"""You are a generative AI expert who writes descriptive titles of generative AI workflows based on the parameters and settings used in the generative AI workflow. 
Only use plain text, no bold or markdown. 
Respond only with the prefix "Title:" followed by the workflow title as a single line. 
Emphasize the RAG Strategy the most prominently in the title, especially between no RAG, RAG, and agent strategies.
The LLM mentioned in the title should always correspond to the Response Synthesizer LLM and not the retriever LLMs or other auxiliary LLMs like the HyDE or the few-shot LLM. 
Valid parameter values for the LLM that should be mentioned are:

- Critique RAG Agent Response Synthesizer LLM
- LATS RAG Agent Response Synthesizer LLM
- No RAG Response Synthesizer LLM
- RAG Response Synthesizer LLM
- ReAct RAG Agent Response Synthesizer LLM
- Subquestion RAG Response Synthesizer LLM

Always mention every LLM in the title and what it's used for.
Below are a few input and output examples for parameters and their corresponding title outputs.

Example #1:

No RAG Response Synthesizer LLM: gpt-4o-std
RAG Strategy: no_rag

Title: Baseline GPT-4o Workflow and No Retrieval

Example #2:

Dynamically Retrieved Few-Shot Prompt Top K: 3
No RAG Response Synthesizer LLM: anthropic-sonnet-35
Prompting Strategy: few-shot
RAG Embedding Model: BAAI/bge-small-en-v1.5
RAG Strategy: no_rag

Title: Claude Sonnet 3.5 with Dynamically Retrieved Few-shot Examples with BGE-small-en Embeddings 

Example #3:

React RAG Agent Response Synthesizer LLM: llama-31-70B
Prompting Strategy: default
RAG Chunk Overlap Percentage: 20
RAG Chunk Size: 1024
RAG Chunk Splitter: sentence
RAG Embedding Model: BAAI/bge-small-en-v1.5
RAG Strategy: react_rag_agent
Reranker Enabled: False

Title: Llama 3.1 70B ReAct RAG Agent with Sentence Splitting and BGE-small-en Embeddings

Example #4:

No RAG Response Synthesizer LLM: gemini-pro
Prompting Strategy: CoT
RAG Chunk Overlap Percentage: 33
RAG Chunk Size: 512
RAG Chunk Splitter: token
RAG Embedding Model: sentence-transformers/all-MiniLM-L12-v2
RAG Strategy: rag
Reranker Enabled: False

Title: Gemini-Pro RAG with Token Splitting, MiniLM-L12-v2 Embeddings, and CoT Prompting


Generate a title for the generative AI strategy with the following parameters:

{param_string}"""
    # print('='*100)
    # print(f'PROMPT: {prompt}')
    response = generate(prompt)
    # print(f'{param_string}')
    # print(f'{response}')
    # print('-'*100, flush=True)

    # extract title and description from the response with regex
    title = re.search(r"^\W*Title:\W*(.*)\W*$", response).group(1).strip()
    description = ", ".join(sorted(raw_param_strings))
    return title, description


@log_function_call
def generate_trial_description_table(df):
    is_cost = is_cost_objective(df)
    objective_2_name = get_objective_2_name(is_cost=is_cost)
    df_output = pd.DataFrame(
        index=df.index, columns=["Accuracy", objective_2_name, "Title", "Description"]
    )
    for i, trial in df.iterrows():
        title, description = generate_trial_description(trial, is_cost)
        df_output.loc[i] = (trial.values_0, trial.values_1, title, description)
    return df_output


@log_function_call
def style_pareto_table(df_pareto_descriptions, is_cost):
    if is_cost:
        obj2_fmt = "{:.4f}¢"
    else:
        obj2_fmt = "{:.2f}s"

    objective_2_name = get_objective_2_name(is_cost=is_cost)

    df_desc_styled = (
        df_pareto_descriptions.style.set_table_styles(
            [
                {
                    "selector": "td, th",
                    "props": [
                        ("text-align", "left"),
                        # ('font-size','12pt'),
                        ("white-space", "pre-wrap"),
                        ("word-wrap", "break-word"),
                        # ('font-family','Arial')
                    ],
                },
                {"selector": "td.col0, td.col1", "props": [("text-align", "center")]},
                {
                    "selector": "td.col2",
                    "props": [("width", "500px"), ("font-weight", "bold")],
                },
                {"selector": "td.col3", "props": [("width", "1000px")]},
            ]
        )
        .format(
            {
                "Accuracy": "{:.1%}",
                objective_2_name: obj2_fmt,
            }
        )
        .hide(axis="index")
        .background_gradient(
            subset=["Accuracy"],
            cmap=normal_to_good,
            vmin=0,
            vmax=1,
        )
        .background_gradient(
            subset=[objective_2_name],
            cmap=normal_to_bad,
            vmin=np.log(0.1),
            vmax=np.log(15),
            gmap=df_pareto_descriptions[objective_2_name].apply(
                lambda x: np.log(max(x, 0.01))
            ),
        )
    )
    return df_desc_styled


def get_pareto_square_points(xs, ys):
    i_sorted = np.argsort(ys.values)
    xs, ys = xs.iloc[i_sorted], ys.iloc[i_sorted]
    px, py = [], []
    for i in range(len(xs)):
        px.append(xs.iloc[i])
        py.append(ys.iloc[i])
        if i + 1 < len(xs):
            px.append(0.50 * xs.iloc[i] + 0.50 * xs.iloc[i + 1])
            py.append(ys.iloc[i])

            px.append(xs.iloc[i + 1])
            py.append(ys.iloc[i])

            px.append(xs.iloc[i + 1])
            py.append(0.5 * ys.iloc[i] + 0.5 * ys.iloc[i + 1])
    return px, py


def get_baselines(df):
    is_cost = is_cost_objective(df)
    # Define the baseline by filtering for specifc parameters
    baseline_params = [
        {
            "params_response_synthesizer_llm": "gemini-flash",
            "params_rag_mode": "rag",
            "params_template_name": "default",
            "params_splitter_method": "sentence",
            "params_splitter_chunk_exp": 10,
            "params_splitter_chunk_overlap_frac": 0.25,
            "params_rag_method": "sparse",
            "params_rag_top_k": 50,
            "params_rag_query_decomposition_enabled": False,
            "params_reranker_enabled": False,
            "params_hyde_enabled": False,
            "params_additional_context_enabled": False,
        },
        {
            "params_rag_mode": "rag",
            "params_template_name": "default",
            "params_response_synthesizer_llm": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "params_few_shot_enabled": False,
            "params_reranker_enabled": False,
            "params_hyde_enabled": False,
            "params_additional_context_enabled": False,
            "params_rag_method": "dense",
            "params_rag_query_decomposition_enabled": False,
            "params_rag_top_k": 4,
            "params_rag_embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
            "params_splitter_method": "sentence",
            "params_splitter_chunk_exp": 10,
            "params_splitter_chunk_size": 1024,
            "params_splitter_chunk_overlap_frac": 0.22,
        },
    ]
    # find the best trial for each baseline filter
    results = []
    key_errors = []
    for params in baseline_params:
        try:
            # filter to the baseline's parameters
            df_baseline = df
            for col, val in params.items():
                df_baseline = df_baseline[df_baseline[col] == val]
            n_baselines = len(df_baseline)
            if n_baselines == 0:
                print("Found no baselines for parameters:", params)
                continue

            print(f"Found {n_baselines} baselines for parameters: {params}")

            # perfer baselines that use gpt-* as the synthesizer
            # df_baseline = df_baseline.copy()
            # df_baseline['primary_llm'] = df_baseline.apply(
            #     lambda row: row['params_llm_name'] if 'params_llm_name' in row.index else row["response_synthesizer_llm"],
            #     axis=1,
            # )
            # df_baseline["preferred_llm"] = (
            #     df_baseline["primary_llm"] == "gpt-4o-std"
            # ) + df_baseline["primary_llm"].str.startswith("gpt")
            # df_baseline = df_baseline.sort_values(
            #     by=["preferred_llm", "values_0", "values_1"],
            #     ascending=[False, False, True],
            # )

            # # debug what the baselines look like
            # display(df_baseline[["values_0", "values_1", "params_llm_name"] + list(params.keys())])

            # take the first ranked trial that matches the baseline params and generate a title
            trial = df_baseline.loc[
                df_baseline["user_attrs_metric_num_success"].idxmax(), :
            ].copy()
            title, description = generate_trial_description(trial, is_cost)
            trial["title"] = f"Baseline: {title}"
            trial["description"] = description

            results.append(trial)

            # display(trial.loc[
            #     ['values_0', 'values_1', 'title', 'description'] + [c for c in df.columns if c.startswith('params_')]
            # ])
        except KeyError as e:
            key_errors.append(e)
    if len(results) == 0:
        missing_keys = ", ".join({str(e.args[0]) for e in key_errors})
        print(f"WARNING: No baselines found. {missing_keys}")
    results = pd.DataFrame(
        results, columns=df.columns.tolist() + ["title", "description"]
    ).reset_index(drop=True)
    return results


def get_comparison_accuracies(study_names):
    if isinstance(study_names, str):
        study_names = [study_names]
    comparisons = {
        "financebench": [
            ("Amazon Q", 0.592),
            ("Azure Assistant (GPT-4o-Mini)", 0.551),
            ("Azure Assistant (GPT-4o)", 0.653),
            # ("BAM Embeddings (Anderson, 2024)", 0.55),
            # ("OpenSSA DANA-NK-NP (Luong, 2024)", 0.90),
            # ("KodeXv0.1 (Rajani, 2024)", 0.78),
            # ("OODA Agent (Nguyen, 2024)", 0.85),
        ],
        "crag-private": [
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.478),
            # ("WeKnow-RAG (Xie, 2024)", 0.41),
        ],
        "crag-finance": [
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.278),
        ],
        "crag-sports": [
            ("Amazon Q", 0.63),
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.542),
        ],
        "crag-music": [
            ("Amazon Q", 0.735),
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.41),
        ],
        "crag-movie": [
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.384),
        ],
        "crag-open": [
            # ("KDD Cup'24 Winner (Xia, 2024)", 0.23),
        ],
        "hotpot": [
            # ("AutoReason (Sevinc, 2024)", 0.766)
        ],
        "infinitebench": [
            # ("ChatQA 2 (Xu, 2025)", 0.48),
        ],
    }
    # results = pd.DataFrame(index=study_names, columns=['Title', 'Accuracy'])
    results = []
    for study_name in study_names:
        if "synthetic_" in study_name:
            continue
        for bench_name, entries in comparisons.items():
            if bench_name in study_name:
                for title, accuracy in entries:
                    results.append([study_name, title, accuracy])
    return pd.DataFrame(results, columns=["study_name", "Title", "Accuracy"])


def wrap(s, width=50):
    return "\n".join(textwrap.wrap(s, width))


@log_function_call
def plot_pareto_plot_altair(
    df_pareto_desc,
    study_name,
    is_cost,
    df_trials=None,
    color_dict=None,
    ax=None,
    titles=None,
):
    charts = []
    objective_2_name = get_objective_2_name(is_cost=is_cost)

    # historical trials
    if df_trials is not None:
        all_trials = (
            alt.Chart(df_trials[["values_0", "values_1"]])
            .mark_circle(
                size=100,
            )
            .encode(
                x=alt.X(
                    "values_1", scale=alt.Scale(type="log"), title=objective_2_name
                ),
                y=alt.Y("values_0", title="Accuracy"),
                color=alt.value("lightgrey"),
            )
        )
        charts.append(all_trials)

    # Pareto-frontier line
    px, py = get_pareto_square_points(
        df_pareto_desc[objective_2_name], df_pareto_desc["Accuracy"]
    )
    df_front = pd.DataFrame({"values_1": px, "values_0": py})
    pareto_front = (
        alt.Chart(df_front)
        .mark_line()
        .encode(
            x=alt.X("values_1", scale=alt.Scale(type="log"), title=objective_2_name),
            y=alt.Y("values_0", title="Accuracy"),
            color=alt.value("black"),
        )
    )
    charts.append(pareto_front)

    # Pareto-frontier points
    pareto_trials = (
        alt.Chart(df_pareto_desc[["Accuracy", objective_2_name, "Title"]])
        .mark_circle(
            size=100,
        )
        .encode(
            x=alt.X(objective_2_name, scale=alt.Scale(type="log")),
            y="Accuracy",
            color=alt.Color("Title"),
        )
    )
    charts.append(pareto_trials)

    # plot the baselines
    if df_trials is not None:
        baselines = get_baselines(df_trials)
        all_trials = (
            alt.Chart(baselines[["values_0", "values_1", "title"]])
            .mark_square(
                size=100,
            )
            .encode(
                x=alt.X(
                    "values_1", scale=alt.Scale(type="log"), title=objective_2_name
                ),
                y=alt.Y("values_0", title="Accuracy"),
                color=alt.Color("title"),
            )
        )
        charts.append(all_trials)

    # final combined chart
    title = f"{get_name(study_name, titles=titles)}"
    combined_chart = alt.layer(*charts).properties(
        width=800,
        height=600,
        title=title if SHOW_TITLE else None,
    )
    combined_chart.show()
    return combined_chart, title


def apply_altair_like_styles(ax):
    # add grid lines
    ax.grid(True, which="major", linestyle="-", color="lightgrey", linewidth=0.5)

    # make border on left and bottom only
    ax.spines["top"].set_linewidth(False)
    ax.spines["top"].set_color("lightgrey")
    ax.spines["top"].set_linewidth(0.5)

    ax.spines["right"].set_linewidth(False)
    ax.spines["right"].set_color("lightgrey")
    ax.spines["right"].set_linewidth(0.5)

    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    # make axis labels bold
    ax.set_xlabel(ax.get_xlabel(), fontweight="bold", fontsize=8)
    ax.set_ylabel(ax.get_ylabel(), fontweight="bold", fontsize=8)


def plot_pareto_plot(
    df_pareto_desc,
    study_name,
    is_cost,
    df_trials=None,
    color_dict=None,
    ax=None,
    titles=None,
    show_title=True,
):
    objective_2_name = get_objective_2_name(is_cost=is_cost)
    df_pareto_desc = df_pareto_desc.sort_values(
        ["Accuracy", objective_2_name], ascending=[False, True]
    )
    px, py = get_pareto_square_points(
        df_pareto_desc[objective_2_name], df_pareto_desc["Accuracy"]
    )

    # testing out alternative plot with Altair
    # chart = plot_pareto_plot_altair(df_pareto_desc, study_name, df_trials, color_dict, ax)

    is_subplot = ax is not None
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    else:
        fig = None

    markersize = 6 if is_subplot else 10

    # plot all historical trials
    if df_trials is not None:
        ax.plot(
            df_trials["values_1"],
            df_trials["values_0"],
            "o",
            color=grey_face_color,
            markeredgecolor=grey_edge_color,
            markeredgewidth=0.5,
            alpha=0.25,
            label="All Trials",
            markersize=markersize,
        )

    # plot each individual trial on the Pareto-frontier
    ax.plot(px, py, "-", color=grey_edge_color, label="Pareto-Frontier")
    labels = df_pareto_desc["Title"].unique()
    for i, label in enumerate(labels):
        if color_dict is None:
            color = discrete_cmap(i / max(10, len(labels)))
        else:
            color = color_dict[label]

        filter = df_pareto_desc["Title"] == label
        ax.plot(
            df_pareto_desc.loc[filter, objective_2_name],
            df_pareto_desc.loc[filter, "Accuracy"],
            "o",
            color=color,
            markeredgewidth=0.5,
            markeredgecolor=(0, 0, 0, 0.9),
            label=wrap(label, 55 if is_subplot else 100),
            markersize=markersize,
            zorder=2,
        )

    # plot the baseline trials
    if df_trials is not None:
        baselines = get_baselines(df_trials)
        for i, (idx, trial) in enumerate(baselines.iterrows()):
            ax.plot(
                trial["values_1"],
                trial["values_0"],
                "s",
                alpha=1,
                color="none",  # Transparent center
                markeredgewidth=0.5,  # Bigger edge
                markeredgecolor="black",  # Black edge
                label=wrap(trial["title"], 55 if is_subplot else 100),
                markersize=markersize,
            )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # if is_subplot:
    ax.tick_params(axis="both", which="major", labelsize=12)
    title = f"{get_name(study_name, titles=titles)}"
    if show_title:
        ax.set_title(title, fontsize=18)
    ax.set_xscale("log")

    # constrain axis limits and expand a bit for the legend
    xlim = ax.get_xlim()
    new_xlim_right = 10 ** (
        1 + np.floor(np.log10(xlim[1]))
    )  # round xlim[1] to the next highest power of 10
    ax.set_xlim([xlim[0], new_xlim_right])

    if not is_subplot:
        # plot sota comparisons as horizontal lines
        comparisons = get_comparison_accuracies([study_name])
        linestyles = [":"]  # "--", "-.", ":", "-"

        for i, (_, (_, competitor, accuracy)) in enumerate(comparisons.iterrows()):
            if np.isnan(accuracy):
                continue
            linestyle = linestyles[i % len(linestyles)]
            ax.axhline(
                accuracy,
                color="black",
                linestyle=linestyle,
                linewidth=1.0,
                zorder=1,
            )
            # --------------------
            # align right
            # --------------------
            ax.text(
                0.85 * ax.get_xlim()[1],
                accuracy,
                competitor,
                color="black",
                fontsize=9,
                va="center",
                ha="right",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
            # --------------------
            # align left
            # --------------------
            # ax.text(
            #     1.3 * ax.get_xlim()[0],
            #     accuracy,
            #     competitor,
            #     color="black",
            #     fontsize=9,
            #     va="center",
            #     ha="left",
            #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            # )

    ylim = ax.get_ylim()
    new_ylim_top = min(
        1, np.ceil(ylim[1] * 5) / 5
    )  # round ylim[1] to the next highest 20%
    ax.set_ylim([max(0, ylim[0]), new_ylim_top])

    apply_altair_like_styles(ax)

    if is_subplot:
        ax.margins(y=100)
        title = "Workflow Family" if show_title else f"{title}"
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            fontsize=9,
            title=title,
            title_fontsize=10,
            alignment="left",
        )
    else:
        title = "Workflow Family" if show_title else f"{title} - Workflows"
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10,
            title=title,
            title_fontsize=12,
            alignment="left",
        )

    legend.get_title().set_fontweight("bold")

    label_fontsize = 10 if is_subplot else 14
    ax.set_xlabel(
        descriptive_name("values_1", is_cost=is_cost), fontsize=label_fontsize
    )
    ax.set_ylabel(descriptive_name("values_0"), fontsize=label_fontsize)

    return fig, title


@log_function_call
def pareto_plot_and_table(df, study_name, ax=None, titles=None):
    df_trials = df[df["study_name"] == study_name]
    is_cost = is_cost_objective(df_trials)
    df_pareto = df_trials[df_trials["pareto"]].sort_values(
        ["values_0", "values_1"], ascending=[False, True]
    )
    df_desc = generate_trial_description_table(df_pareto)
    if is_retriever_study(df_trials):
        fig, title = plot_retriever_pareto(
            df_trials,
            study_name,
            ax=ax,
            titles=titles,
            show_title=SHOW_TITLE,
        )
    else:
        fig, title = plot_pareto_plot(
            df_desc,
            study_name,
            is_cost,
            df_trials,
            ax=ax,
            titles=titles,
            show_title=SHOW_TITLE,
        )
    table = style_pareto_table(df_desc, is_cost=is_cost)
    return fig, table, title


def labeled_multi_line_plot(
    df,
    study_name,
    x_func,
    y_func,
    xlabel=None,
    ylabel=None,
    xscale=None,
    yscale=None,
    xpct=False,
    ypct=False,
    titles=None,
):
    study_names = df["study_name"].unique()
    labels = []
    fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
    for study_i, study_name_i in enumerate(study_names):
        df_study = df[df["study_name"] == study_name_i]
        color = continuous_cmap(study_i / len(study_names))
        highlight = study_name_i == study_name
        x = x_func(df_study)
        y = y_func(df_study)
        ax.plot(
            x,
            y,
            "-",
            color=color,
            label=get_name(study_name_i, titles=titles),
            alpha=0.75 if highlight else 0.3,
            lw=3 if highlight else 1,
        )
        ax.plot(
            [x.iloc[-1]], [y.iloc[-1]], ".", color=color, alpha=1 if highlight else 0.5
        )
        labels.append(
            ax.text(
                x.iloc[-1],
                y.iloc[-1],
                f"  {get_name(study_name_i, titles=titles)}",
                ha="left",
                va="center",
                fontsize=5,
                color=color,
            )
        )

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xpct:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if ypct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.legend(loc="best", fontsize=5)
    fig.tight_layout()
    adjust_text(
        labels,
        expand_axes=True,
        iter_lim=1000,
    )
    return fig


@log_function_call
def pareto_comparison_plot(df, study_name, titles=None):
    is_cost = is_cost_objective(df)
    if is_cost is None:
        print("Not plotting Pareto comparison because of mixed objectives.")
        return None
    fig, ax = plt.subplots(dpi=300, figsize=(8, 5))
    study_names = df["study_name"].unique()
    labels = []
    for study_i, study_name_i in enumerate(study_names):
        df_study = df[df["study_name"] == study_name_i]
        objective_2_name = get_objective_2_name(is_cost=is_cost)
        objective_2_unit = get_objective_2_unit(is_cost=is_cost)
        df_pareto = df_study[df_study["pareto"]].sort_values(
            ["values_0", "values_1"], ascending=[True, False]
        )
        color = continuous_cmap(study_i / len(study_names))
        highlight = study_name_i == study_name
        ax.plot(
            df_pareto["values_1"],
            df_pareto["values_0"],
            "-",
            color=color,
            alpha=0.75 if highlight else 0.35,
            lw=2 if highlight else 1,
        )
        ax.plot(
            df_pareto["values_1"],
            df_pareto["values_0"],
            ".",
            color=color,
            alpha=0.95 if highlight else 0.5,
            label=get_name(study_name_i, titles=titles),
        )
        # labels.append(ax.text(df_pareto['values_1'].iloc[-1], df_pareto['values_0'].iloc[-1], f'  {study_name_i}', ha='left', va='center', fontsize=5, color=color))
        labels.append(
            ax.text(
                df_pareto["values_1"].iloc[-1],
                df_pareto["values_0"].iloc[-1],
                f"  {get_name(study_name_i, titles=titles)}",
                ha="left",
                va="center",
                fontsize=5,
                color=color,
            )
        )

    ax.legend(loc="lower right", fontsize=5)
    ax.set_xlabel(f"{objective_2_name} ({objective_2_unit})")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xscale("log")
    adjust_text(
        labels,
        expand_axes=True,
        iter_lim=1000,
    )
    return fig


def pareto_area(df_values, limits, sense):
    xmin, xmax = limits[1]
    ymin, ymax = limits[0]
    total_area = (xmax - xmin) * (ymax - ymin)

    df_pareto = df_values[safe_paretoset(df_values, sense=sense)].copy()
    df_pareto = df_pareto[
        (df_pareto.iloc[:, 1] >= xmin)
        & (df_pareto.iloc[:, 1] <= xmax)
        & (df_pareto.iloc[:, 0] >= ymin)
        & (df_pareto.iloc[:, 0] <= ymax)
    ]
    if len(df_pareto) == 0:
        return 0.0
    df_pareto.sort_values(by=[df_pareto.columns[1], df_pareto.columns[0]], inplace=True)

    dominated_area = 0.0
    for i in range(1, len(df_pareto)):
        y1, x1 = df_pareto.iloc[i - 1]
        y2, x2 = df_pareto.iloc[i]
        x1, x2 = np.clip([x1, x2], xmin, xmax)
        y1, y2 = np.clip([y1, y2], ymin, ymax)
        dominated_area += (x2 - x1) * (y1 - ymin)
    last_y, last_x = df_pareto.iloc[-1]
    last_y = np.clip(last_y, ymin, ymax)
    last_x = np.clip(last_x, xmin, xmax)
    dominated_area += (xmax - last_x) * (last_y - ymin)
    return dominated_area / total_area


def cumulative_pareto_area(df, limits, sense):
    df_values = df.sort_values(by="datetime_complete")[["values_0", "values_1"]]
    areas = pd.Series(index=df_values.index, dtype=float)
    for i, idx in enumerate(df_values.index):
        areas.loc[idx] = pareto_area(df_values.iloc[: i + 1, :], limits, sense)
    return areas


@log_function_call
def pareto_area_plot(df, study_name, titles=None):
    finite_0 = df["values_0"].apply(np.isfinite)
    finite_1 = df["values_1"].apply(np.isfinite)
    limits = [
        df.loc[finite_0, "values_0"].quantile([0.1, 1]).values,
        df.loc[finite_1, "values_1"].quantile([0, 0.9]).values,
    ]
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: cumulative_pareto_area(df, limits, sense=["max", "min"]),
        xlabel="Optimizer Trials",
        ylabel="Pareto Area %",
        xscale="log",
        ypct=True,
        titles=titles,
    )


@log_function_call
def accuracy_plot(df, study_name, titles=None):
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: df["values_0"].cummax(),
        xlabel="Optimizer Trials",
        ylabel=descriptive_name("values_0"),
        xscale="log",
        ypct=True,
        titles=titles,
    )


@log_function_call
def latency_plot(df, study_name, titles=None):
    is_cost = is_cost_objective(df)
    if is_cost is None:
        print("Not plotting latency because of mixed objectives.")
        return None
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: df["values_1"].cummin(),
        xlabel="Optimizer Trials",
        ylabel=descriptive_name("values_1", is_cost=is_cost),
        xscale="log",
        yscale="log",
        titles=titles,
    )


@log_function_call
def compute_plot(df, study_name, titles=None):
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: df["compute_hours"],
        xlabel="Optimizer Trials",
        ylabel="Cumulative Trial Time (Hours)",
        titles=titles,
    )


@log_function_call
def cost_plot(df, study_name, titles=None):
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: df["user_attrs_metric_llm_cost_total"].cumsum(),
        xlabel="Optimizer Trials",
        ylabel="Cumulative Trial Cost (Dollars)",
        titles=titles,
    )


@log_function_call
def compute_trial_rate_plot(df, study_name, titles=None):
    return labeled_multi_line_plot(
        df,
        study_name,
        x_func=lambda df: pd.Series(np.arange(1, len(df) + 1), index=df.index),
        y_func=lambda df: df["wallclock_hours"],
        xlabel="Optimizer Trials",
        ylabel="Wallclock Time (Hours)",
        titles=titles,
    )


def truncate_label(label, max_length=30):
    label = str(label)
    if len(label) > max_length and "/" in label:
        label = label.split("/")[-1]
    if len(label) > max_length:
        part_length = (max_length - 3) // 2
        label = label[:part_length] + "..." + label[-part_length:]
    return label


def consider_categorical(series: pd.Series) -> bool:
    return (not pd.api.types.is_numeric_dtype(series)) or series.nunique(
        dropna=False
    ) <= 15


def insert_bin_col(df, param_col):
    param_cols = [col for col in df.columns if col.startswith("params_")]
    numeric_cols = df[param_cols].select_dtypes(include=["int", "float"]).columns
    int_cols = [
        col
        for col in numeric_cols
        if df[col].dropna().apply(lambda x: float(x) == float(int(x))).all()
    ]
    bin_col = f"{param_col}_binned"
    if (
        pd.api.types.is_numeric_dtype(df[param_col])
        and df[param_col].nunique(dropna=False) > 15
    ):
        bins = pd.cut(df[param_col], bins=10, duplicates="drop")
        if param_col in int_cols:
            bin_edges = bins.cat.categories.left.round(0).astype(int)
            max_edge = bins.cat.categories.right.max().round(0).astype(int) + 1
            bins = pd.cut(
                df[param_col], bins=list(bin_edges) + [max_edge], duplicates="drop"
            )
            df[bin_col] = bins.apply(lambda x: f"({int(x.left):g}, {int(x.right):g}]")
        else:
            bins = pd.cut(df[param_col], bins=10, duplicates="drop")
            df[bin_col] = bins.apply(lambda x: f"({x.left:g}, {x.right:g}]")
        df[bin_col] = df[bin_col].cat.add_categories("N/A")
        df[bin_col] = df[bin_col].fillna("N/A")
    elif pd.api.types.is_numeric_dtype(df[param_col]):
        df[bin_col] = df[param_col].apply(lambda x: f"{x:g}")
        df[bin_col] = df[bin_col].str.replace("nan", "N/A").fillna("N/A")
    else:
        df[bin_col] = df[param_col].astype(str).str.replace("nan", "N/A").fillna("N/A")
        df[bin_col] = df[bin_col].apply(truncate_label)
    return bin_col


@log_function_call
def param_plot(df, study_name, param_cols, titles=None):
    if isinstance(param_cols, str):
        param_cols = [param_cols]

    # add columns to group the parameter columns by and bin the numeric colums where necessary
    df: pd.DataFrame = df[df["study_name"] == study_name].copy()
    is_cost = is_cost_objective(df)
    bin_cols = [insert_bin_col(df, param) for param in param_cols]

    # calculate stats of the parameter groups to order their display by
    stats = df.groupby(bin_cols, observed=False)[["values_0", "values_1"]].agg(
        ["min", "max", "mean", "size"]
    )
    stats = stats[stats[("values_0", "size")] > 0]
    stats.sort_values(
        by=[("values_0", "max"), ("values_1", "min")],
        ascending=[True, False],
        inplace=True,
        na_position="first",
    )
    n_lines = min(50, len(stats))

    # create the accuracy and latency plots the same
    fig, axes = plt.subplots(
        ncols=4,
        dpi=300,
        figsize=(8, 2 + 0.1 * n_lines),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    for ax, value_col, color in zip(axes[:2], ["values_0", "values_1"], ["C0", "C1"]):
        for i, (param_bins, row) in enumerate(stats.tail(n_lines).iterrows()):
            p_bins = (
                param_bins.copy() if isinstance(param_bins, tuple) else (param_bins,)
            )

            # plot range bar
            ax.plot(
                [row[(value_col, "min")], row[(value_col, "max")]],
                [i, i],
                "-",
                color=grey_face_color,
                lw=5,
            )

            # filter trials to matching the current parameter
            temp_i = df.copy()
            for bin_col, param_bin in zip(bin_cols, p_bins):
                temp_i = temp_i[temp_i[bin_col] == param_bin]

            # plot all trials as dots
            jitter = np.random.uniform(-0.25, 0.25, len(temp_i))
            alpha = 0.25 * 200 / (200 + len(temp_i))
            ax.plot(
                temp_i[value_col],
                [i] * len(temp_i) + jitter,
                "o",
                color=grey_face_color,
                markeredgecolor=grey_edge_color,
                alpha=alpha,
                markersize=5,
                markeredgewidth=0.5,
            )

            # plot pareto trials as dots
            temp_pareto = temp_i[temp_i["pareto"]]
            ax.plot(
                temp_pareto[value_col],
                [i] * len(temp_pareto),
                "o",
                color=color,
                alpha=0.75,
                markersize=5,
                markeredgewidth=0.5,
            )

        ax.set_title(
            descriptive_name(value_col, is_cost=is_cost).capitalize(), fontsize=8
        )
        if value_col == "values_0":
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if value_col == "values_1":
            ax.set_xscale("log")

    # create the total trial time plot
    ax = axes[2]
    for i, (param_bins, row) in enumerate(stats.tail(n_lines).iterrows()):
        p_bins = param_bins.copy() if isinstance(param_bins, tuple) else (param_bins,)
        temp_i = df
        for bin_col, param_bin in zip(bin_cols, p_bins):
            temp_i = temp_i[temp_i[bin_col] == param_bin]
        if len(temp_i) == 0:
            continue
        log_durations = np.log10((temp_i["duration"] / pd.Timedelta(minutes=1)) + 1)
        parts = ax.violinplot(
            log_durations,
            [i],
            widths=0.9,
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.1,
        )
        for body in parts["bodies"]:
            body.set_facecolor(grey_edge_color)
            body.set_edgecolor(grey_edge_color)
            body.set_linewidth(0.5)
            body.set_alpha(0.8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$"))
    ax.set_title("Trial Minutes", fontsize=8)

    # create the trial count plot
    ax = axes[3]
    bars = ax.barh(
        range(n_lines),
        stats.tail(n_lines)[("values_0", "size")],
        color=grey_edge_color,
        edgecolor=grey_edge_color,
        alpha=0.8,
        linewidth=0.5,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:g}",
            va="center",
            fontsize=5,
        )
    ax.set_xlim(0, stats.tail(n_lines)[("values_0", "size")].max() * 1.2)
    ax.set_title("Trial Count", fontsize=8)

    # apply common formating
    for i, ax in enumerate(axes):
        ax.set_yticks(range(n_lines))
        ax.set_yticklabels([label for label in stats.tail(n_lines).index])
        ax.set_ylabel("")
        ax.set_ylim([-0.5, n_lines - 0.5])
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=5)
        if i > 0:
            ax.tick_params(left=False)
    title = descriptive_name(param_cols, is_cost=is_cost)
    suptitle = f"{title} ({get_name(study_name, titles=titles)})"
    if SHOW_TITLE:
        fig.suptitle(suptitle)
    fig.tight_layout(h_pad=0)
    return fig, suptitle


def get_common_study_name_prefix(study_names):
    uniques = {s.split("-")[0] for s in study_names}
    return "+".join(uniques)


@log_function_call
def param_plot_all_studies(df: pd.DataFrame, study_name, param_cols):
    is_cost = is_cost_objective(df)
    if is_cost is None:
        print("Not plotting parameter plot because of mixed objectives.")
        return None

    if isinstance(param_cols, str):
        param_cols = [param_cols]

    # add columns to group the parameter columns by and bin the numeric colums where necessary
    df = df.copy().reset_index(drop=True)
    bin_cols = [insert_bin_col(df, param) for param in param_cols]

    # normalize the accuracy to be percent of maximum per study
    df["values_0"] = df.groupby("study_name")["values_0"].transform(
        lambda x: x / x.max()
    )

    # take the pareto trials (where param is used) for each unique parameter group in each study
    df_best = (
        df.groupby(["study_name"] + list(bin_cols), observed=True)
        .apply(
            lambda x: x[
                safe_paretoset(x[["values_0", "values_1"]], sense=["max", "min"])
            ],
            include_groups=False,
        )
        .reset_index()
    )

    # calculate stats of the parameter groups to order their display by
    stats = df_best.groupby(bin_cols, observed=True)[["values_0", "values_1"]].agg(
        ["min", "max", "median", "size"]
    )
    stats = stats[stats[("values_0", "size")] > 0]
    stats.sort_values(
        by=[
            ("values_0", "max"),
            ("values_0", "median"),
            ("values_1", "min"),
            ("values_1", "median"),
        ],
        ascending=[True, True, False, False],
        inplace=True,
        na_position="first",
    )
    n_lines = min(50, len(stats))

    # create the accuracy and latency plots the same
    fig, axes = plt.subplots(
        ncols=4,
        dpi=300,
        figsize=(8, 2 + 0.1 * n_lines),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    for ax, value_col, color in zip(axes[:2], ["values_0", "values_1"], ["C0", "C1"]):
        # plot each bar position manually
        for i, (param_bins, _) in enumerate(stats.tail(n_lines).iterrows()):
            p_bins = param_bins if isinstance(param_bins, tuple) else (param_bins,)

            # filter trials to matching the current parameter
            df_param = df_best.copy()
            for bin_col, param_bin in zip(bin_cols, p_bins):
                df_param = df_param[df_param[bin_col] == param_bin]
            if len(df_param) == 0:
                continue

            # calculate the point estimate using only studies that used the parameter
            finite_only = df_param[value_col].apply(np.isfinite)
            df_param = df_param[finite_only]
            param_min, param_median, param_max = df_param[value_col].quantile(
                [0, 0.5, 1]
            )

            q1, q3 = df_param[value_col].quantile([0.25, 0.75])

            # plot for this parameter group
            ax.plot(
                [param_min, param_max],
                [i, i],
                "-",
                color=grey_face_color,
                lw=5,
            )
            jitter = np.random.uniform(-0.25, 0.25, len(df_param))
            alpha = 0.25 * 200 / (200 + len(df_param))
            ax.plot(
                df_param[value_col],
                [i] * len(df_param) + jitter,
                "o",
                color=grey_face_color,
                markeredgecolor=grey_edge_color,
                alpha=alpha,
                markersize=5,
                markeredgewidth=0.5,
            )
            df_param_pareto = df_param[df_param["pareto"]]
            ax.plot(
                df_param_pareto[value_col],
                [i] * len(df_param_pareto),
                "o",
                color=color,
                alpha=0.75,
                markersize=5,
                markeredgewidth=0.5,
            )

        if value_col == "values_0":
            ax.set_title("Normalized Accuracy\n(Pareto Trials)", fontsize=8)
        elif value_col == "values_1":
            ax.set_title(
                f"{get_objective_2_unit(is_cost=is_cost).capitalize()}\n(Pareto Trials)",
                fontsize=8,
            )
        else:
            ax.set_title(descriptive_name(value_col, is_cost=is_cost), fontsize=8)
        if value_col == "values_0":
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if value_col == "values_1":
            ax.set_xscale("log")
    del df_best

    # create the frequency on pareto plot
    ax = axes[2]
    on_pareto = df.groupby(["study_name"] + list(bin_cols), observed=True)[
        "pareto"
    ].max()
    pareto_freq = (
        on_pareto.groupby(bin_cols, observed=True).sum() / df["study_name"].nunique()
    )
    pareto_freq = pareto_freq.loc[stats.index]
    assert pareto_freq.index.to_list() == stats.index.to_list(), "Indexes do not match"
    bars = ax.barh(
        range(n_lines), pareto_freq.tail(n_lines), color="C3", alpha=0.8, linewidth=0.5
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.0%}",
            va="center",
            fontsize=5,
        )
    ax.set_xlim(0, pareto_freq.max() * 1.25)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title("Frequency on\nOverall Pareto", fontsize=8)

    # create the total samples plot
    ax = axes[3]
    counts = df.groupby(bin_cols, observed=True).size()
    counts = counts.loc[stats.index]
    assert counts.index.to_list() == stats.index.to_list(), "Indexes do not match"
    bars = ax.barh(
        range(n_lines),
        counts.tail(n_lines),
        color=grey_edge_color,
        edgecolor=grey_edge_color,
        alpha=0.8,
        linewidth=0.5,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:g}",
            va="center",
            fontsize=5,
        )
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title("Total\nTrial Count", fontsize=8)

    # apply common formating
    for i, ax in enumerate(axes):
        ax.set_yticks(range(n_lines))
        ax.set_yticklabels([label for label in stats.tail(n_lines).index])
        ax.set_ylabel("")
        ax.set_ylim([-0.5, n_lines - 0.5])
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=5)
        if i > 0:
            ax.tick_params(left=False)
    title = descriptive_name(param_cols)
    title = f"{title} (Across {len(df['study_name'].unique())} Studies)"
    if SHOW_TITLE:
        fig.suptitle(title)
    fig.tight_layout(h_pad=0)
    return fig, title


@log_function_call
def param_pareto_plot(df: pd.DataFrame, study_name, param_col, titles=None):
    df = df[np.isfinite(df["values_1"])]
    df = df.dropna(subset=["values_0", "values_1"])

    temp = df[df["study_name"] == study_name].copy()
    is_cost = is_cost_objective(temp)

    bin_col = "bin"
    if (
        pd.api.types.is_numeric_dtype(temp[param_col])
        and temp[param_col].nunique(dropna=False) > 15
    ):
        bins = pd.cut(temp[param_col], bins=10, duplicates="drop")
        temp[bin_col] = bins.cat.add_categories("N/A")
        temp[bin_col] = temp[bin_col].fillna("N/A")
    else:
        temp[bin_col] = (
            temp[param_col].astype(str).str.replace("nan", "N/A").fillna("N/A")
        )
    bin_values = temp[bin_col].unique()

    fig, axes = plt.subplots(
        nrows=1, ncols=2, dpi=300, figsize=(10, 3), sharex=True, sharey=True
    )

    # plot Pareto-frontier for each individual parameter value
    plot_param_pareto_fronts(temp, bin_col, bin_values, axes[0])

    if SHOW_TITLE:
        axes[0].set_title(f"{descriptive_name(param_col)} Pareto-Frontier")

    # plot the overall pareto front with family labels
    df_trials = df[df["study_name"] == study_name]
    objective_2_name = get_objective_2_name(data=df_trials, is_cost=is_cost)
    df_pareto = df_trials[df_trials["pareto"]].sort_values(
        ["values_0", "values_1"], ascending=[False, True]
    )
    df_pareto = df_pareto.rename(
        columns={"values_0": "Accuracy", "values_1": objective_2_name}
    )
    if len(df_pareto) > 0:
        df_pareto["Title"] = df_pareto.apply(
            title_by_rag_mode_minimal, axis=1
        )  # title_by_rag_mode
    else:
        df_pareto["Title"] = ""
    plot_pareto_plot(
        df_pareto, study_name, is_cost, df_trials, ax=axes[1], show_title=SHOW_TITLE
    )

    if SHOW_TITLE:
        axes[1].set_title("Overall Pareto-Frontier")

    if param_col in ["params_rag_mode"]:
        suptitle = get_name(study_name)
    else:
        suptitle = f"{get_name(study_name)} ({descriptive_name(param_col)})"

    if SHOW_TITLE:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, suptitle


def calc_distances_to_pareto(x1, y1, x2, y2):
    index = x1.index
    x2, y2 = np.log(np.asarray(x2)), np.asarray(y2)
    x1, y1 = np.log(np.asarray(x1)), np.asarray(y1)
    x_scale, y_scale = max(1e-9, np.std(x1)), max(1e-9, np.std(y1))
    x1, y1 = x1 / x_scale, y1 / y_scale
    x2, y2 = x2 / x_scale, y2 / y_scale
    distances = pd.Series(
        [
            np.min(np.sqrt((x2 - x1[i]) ** 2 + (y2 - y1[i]) ** 2))
            for i in range(len(x1))
        ],
        index=index,
    )
    distances = distances.replace(np.inf, 1e9)
    return distances


def plot_param_pareto_fronts(temp, bin_col, bin_values, ax):
    is_cost = is_cost_objective(temp)

    labels = []
    for i, bin in enumerate(bin_values):
        color = discrete_cmap(i / max(10, len(bin_values)))
        temp_bin = temp[temp[bin_col] == bin]
        temp_bin.loc[:, "values_0"] = temp_bin.loc[:, "values_0"].replace(
            [np.inf, -np.inf], np.nan
        )
        temp_bin.loc[:, "values_1"] = temp_bin.loc[:, "values_1"].replace(
            [np.inf, -np.inf], np.nan
        )
        temp_bin = temp_bin.dropna(subset=["values_0", "values_1"])
        if len(temp_bin) == 0:
            continue

        x, y = temp_bin["values_1"], temp_bin["values_0"]
        jitter = np.random.uniform(-0.001, 0.001, len(y))

        mask = safe_paretoset(temp_bin[["values_0", "values_1"]], sense=["max", "min"])
        temp_pareto = temp_bin[mask].sort_values(
            by=["values_0", "values_1"], ascending=[True, True]
        )
        px, py = get_pareto_square_points(
            temp_pareto["values_1"], temp_pareto["values_0"]
        )

        dists = calc_distances_to_pareto(x, y, px, py)
        dists_max = dists.replace([np.inf, -np.inf], np.nan).max()
        alphas = 1.0 - np.minimum(1.0, dists / max(1e-9, dists_max))
        alphas = (1 - np.cos(np.pi * alphas**4)) / 2
        alphas = np.clip(0.95 * alphas + 0.05, 0, 1)
        alphas[~np.isfinite(alphas)] = 1.0

        ax.plot(px, py, "-", alpha=0.75, color=color)
        labels.append(
            ax.text(
                temp_pareto["values_1"].iloc[-1],
                temp_pareto["values_0"].iloc[-1],
                f"{bin}\n",
                ha="center",
                va="bottom",
                fontsize=5,
                color=color,
                linespacing=0.75,
            )
        )
        ax.scatter(x, y + jitter, c=[color] * len(x), alpha=alphas, s=2.5)

    ax.set_xlabel(descriptive_name("values_1", is_cost=is_cost))
    ax.set_ylabel(descriptive_name("values_0"))
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="both", which="major", labelsize=8)
    # ax.legend(loc="lower right", fontsize=4)

    apply_altair_like_styles(ax)


@log_function_call
def trial_duration_hist(df, study_name, titles=None):
    study_names = df["study_name"].unique()
    log_durations = [
        np.log10(
            (
                df.loc[df["study_name"] == study_name_i, "duration"]
                / pd.Timedelta(minutes=1)
            )
            + 1
        )
        for study_name_i in study_names
    ]
    fig, ax = plt.subplots(dpi=300, figsize=(8, 2 + 0.2 * len(study_names)))
    parts = ax.violinplot(
        log_durations,
        positions=range(len(study_names)),
        widths=0.9,
        vert=False,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        bw_method=0.05,
    )
    for i, (study_name_i, body) in enumerate(zip(study_names, parts["bodies"])):
        highlight = study_name_i == study_name
        color = continuous_cmap(i / len(study_names))
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.75 if highlight else 0.25)
    for element in [parts["cbars"], parts["cmins"], parts["cmaxes"], parts["cmedians"]]:
        element.set_color(fore_color)
        element.set_linewidth(0.5)
        element.set_alpha(0.25)

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$"))
    ax.set_yticks(range(len(study_names)))
    ax.set_yticklabels([get_name(s, titles=titles) for s in study_names])
    if SHOW_TITLE:
        ax.set_title("Trial Durations (Minutes)")
    ax.tick_params(axis="both", which="major", labelsize=8)
    fig.tight_layout()
    return fig


@log_function_call
def focus_over_time_plot(df, study_name, titles=None):
    df_study = df[df["study_name"] == study_name].copy().reset_index(drop=True)
    is_cost = is_cost_objective(df_study)
    df_study["trial_minutes"] = np.log10(
        df_study["duration"] / pd.Timedelta(minutes=1) + 0.0001
    )
    df_study["pruned_eval"] = df_study["user_attrs_metric_is_pruned"]
    n_window = 100
    n_step = 10

    cols = ["values_0", "values_1", "pareto", "pruned_eval", "trial_minutes"]
    colors = ["C0", "C1", "C2", "C3", "grey"]

    fig, axes = plt.subplots(
        nrows=len(cols), dpi=300, figsize=(8, 1 + 1.25 * len(cols)), sharex=True
    )
    for ax, col, color in zip(axes, cols, colors):
        if col in ["values_0", "values_1", "trial_minutes"]:
            for i, window in enumerate(df_study[col].rolling(n_window)):
                w = window.copy()
                if i > 0 and (i % n_step) == 0:
                    if col == "values_1":
                        w = w.replace(np.inf, 100000)
                        w = np.log10(np.maximum(w, 0.00001))
                    parts = ax.violinplot(
                        w,
                        positions=[i + 1],
                        widths=0.9 * n_step,
                        vert=True,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                        bw_method=0.05,
                    )
                    for body in parts["bodies"]:
                        body.set_facecolor(color)
                        body.set_edgecolor(color)
                        body.set_alpha(0.75)
        elif col in ["pruned_eval"]:
            # plot the rolling mean of col
            window = (
                df_study[col]
                .rolling(
                    n_window,
                    min_periods=0,
                )
                .mean()
            )
            ax.plot(range(1, 1 + len(df_study)), window, color=color, alpha=0.75, lw=1)
        else:
            ax.plot(
                range(1, 1 + len(df_study)),
                df_study[col],
                color=color,
                alpha=0.75,
                lw=1,
            )

        if col in ["values_1", "trial_minutes"]:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$")
            )
        ax.set_ylabel(descriptive_name(col, is_cost=is_cost), fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        if ax is axes[-1]:
            ax.set_xlabel("Optimizer Trials")
        if ax is axes[0]:
            ax.set_title(
                f"Rolling {n_window} Trials Search Focus ({get_name(study_name, titles=titles)})"
            )
        if col in ["values_0", "pruned_eval"]:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    fig.tight_layout()
    return fig


@log_function_call
def param_pair_plot(df, study_name, param_cols, titles=None):
    df = df[df["study_name"] == study_name].copy()
    bin_cols = [insert_bin_col(df, param) for param in param_cols]
    stats = df.groupby(bin_cols, observed=False)[["values_0", "values_1"]].agg(
        ["min", "max", "size"]
    )
    stats = stats[stats[("values_0", "size")] > 0]
    stats.sort_values(
        by=[("values_0", "max")], ascending=True, inplace=True, na_position="first"
    )

    fig, axes = plt.subplots(
        ncols=4,
        dpi=300,
        figsize=(8, 2 + 0.1 * len(stats)),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1]},
    )
    for ax, value_col, color in zip(axes[:2], ["values_0", "values_1"], ["C0", "C1"]):
        for i, (param_bins, row) in enumerate(stats.iterrows()):
            ax.plot(
                [row[(value_col, "min")], row[(value_col, "max")]],
                [i, i],
                "-",
                color=grey_face_color,
                lw=5,
            )

            temp_i = df
            for bin_col, param_bin in zip(bin_cols, param_bins):
                temp_i = temp_i[temp_i[bin_col] == param_bin]

            jitter = np.random.uniform(-0.25, 0.25, len(temp_i))
            ax.plot(
                temp_i[value_col],
                [i] * len(temp_i) + jitter,
                "o",
                color=grey_face_color,
                markeredgecolor=grey_edge_color,
                alpha=0.25,
                markersize=5,
                markeredgewidth=0.5,
            )

            temp_pareto = temp_i[temp_i["pareto"]]
            ax.plot(
                temp_pareto[value_col],
                [i] * len(temp_pareto),
                "o",
                color=color,
                alpha=0.75,
                markersize=5,
                markeredgewidth=0.5,
            )

        ax.set_title(descriptive_name(value_col), fontsize=8)
        if value_col == "values_0":
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if value_col == "values_1":
            ax.set_xscale("log")

    ax = axes[2]
    for i, (param_bins, row) in enumerate(stats.iterrows()):
        temp_i = df
        for bin_col, param_bin in zip(bin_cols, param_bins):
            temp_i = temp_i[temp_i[bin_col] == param_bin]
        if len(temp_i) == 0:
            continue
        log_durations = np.log10((temp_i["duration"] / pd.Timedelta(minutes=1)) + 1)
        parts = ax.violinplot(
            log_durations,
            [i],
            widths=0.9,
            vert=False,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.1,
        )
        for body in parts["bodies"]:
            body.set_facecolor(grey_edge_color)
            body.set_edgecolor(grey_edge_color)
            body.set_linewidth(0.5)
            body.set_alpha(0.8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$"))
    ax.set_title("Trial Minutes", fontsize=8)

    ax = axes[3]
    bars = ax.barh(
        range(len(stats)),
        stats[("values_0", "size")],
        color=grey_edge_color,
        edgecolor=grey_edge_color,
        alpha=0.8,
        linewidth=0.5,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:g}",
            va="center",
            fontsize=5,
        )
    ax.set_xlim(0, stats[("values_0", "size")].max() * 1.2)
    ax.set_title("Trial Count", fontsize=8)

    for i, ax in enumerate(axes):
        ax.set_yticks(range(len(stats)))
        ax.set_yticklabels([label for label in stats.index])
        ax.set_ylabel("")
        ax.set_ylim([-0.5, len(stats) - 0.5])
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=5)
        if i > 0:
            ax.tick_params(left=False)
    title = " x ".join([f"({descriptive_name(param)})" for param in param_cols])
    suptitle = f"{title} ({get_name(study_name, titles=titles)})"
    if SHOW_TITLE:
        fig.suptitle()
    fig.tight_layout(h_pad=0)
    return fig, suptitle


def create_mergeable_params_df(df, strict=False):
    df = df.copy()
    param_cols = [col for col in df.columns if col.startswith("params_")]
    numeric_cols = df[param_cols].select_dtypes(include=["int", "float"]).columns
    int_cols = [
        col
        for col in numeric_cols
        if df[col].dropna().apply(lambda x: x == int(x)).all()
    ]

    # convert object columns
    for col in df[param_cols].select_dtypes(include=["object", "int"]).columns:
        n_bools = df[col].isin([True, False]).sum()
        n_nans = df[col].isna().sum()
        n_ints = df[col].apply(lambda x: isinstance(x, int)).sum()
        n_floats = df[col].apply(lambda x: isinstance(x, float)).sum()
        n_strings = df[col].apply(lambda x: isinstance(x, str)).sum()
        if n_bools > 0 and n_nans > 0 and n_bools + n_nans == len(df):
            # print(f'Filling missing values in {col} with False')
            df[col] = df[col] == True
        elif n_ints > 0 and n_ints + n_floats == len(df):
            # print(f'Converting ints in {col} to floats')
            df[col] = df[col].astype(float)
        elif n_nans > 0 and n_strings > 0:
            # print(f'Filling missing values in {col} with "N/A"')
            df[col] = df[col].fillna("N/A").astype(str)

    # bin numeric columns
    for col in df[param_cols].select_dtypes(include=["number"]).columns:
        # combine similar floats into bins
        n_ints = df[col].apply(lambda x: isinstance(x, int)).sum()
        n_floats = df[col].apply(lambda x: isinstance(x, float)).sum()
        n_unique = df[col].nunique(dropna=True)
        n_bins = 10
        if strict == False and n_unique > n_bins and n_floats + n_ints == len(df):
            # print(f"Converting {col} to bins")
            bins = pd.cut(df[col], bins=n_bins, duplicates="drop")
            if col in int_cols:
                df[col] = bins.apply(lambda x: f"({int(x.left):g}, {int(x.right):g}]")
            else:
                df[col] = bins.apply(lambda x: f"({x.left:g}, {x.right:g}]")
            df[col] = df[col].cat.add_categories("N/A")
            df[col] = df[col].fillna("N/A")
        else:
            df[col] = df[col].apply(lambda x: "N/A" if np.isnan(x) else f"{x:g}")

    # ignore columns with only 1 unique value
    merge_cols = [col for col in param_cols if df[col].nunique(dropna=False) > 1]

    # # convert object columns to categoricals to reduce memory usage
    # for col in df.select_dtypes(include=["object"]).columns:
    #     df[col] = df[col].astype("category")
    return df, merge_cols


def study_similarity(df, study_name_1, study_name_2, merge_cols):
    value_col = "values_0"

    # clean for merging
    def prep(df):
        df = df.copy()
        df[merge_cols] = df[merge_cols].astype(str)
        df = df.groupby(merge_cols, dropna=False)[value_col].median().reset_index()
        return df

    df1 = prep(df[df["study_name"] == study_name_1])
    df2 = prep(df[df["study_name"] == study_name_2])
    # df1 = df1[df1['values_0'] > df1['values_0'].quantile(0.25)]
    # df2 = df2[df2['values_0'] > df2['values_0'].quantile(0.25)]

    merge_cols_i = [
        col
        for col in merge_cols
        if (df1[col].nunique(dropna=False) > 1) and (df2[col].nunique(dropna=False) > 1)
    ]
    if len(merge_cols_i) == 0:
        return np.nan

    df_merged = pd.merge(df1, df2, how="inner", on=merge_cols_i, suffixes=("_1", "_2"))
    df_merged.dropna(subset=[value_col + "_1", value_col + "_2"], inplace=True)
    corr = df_merged[value_col + "_1"].corr(
        df_merged[value_col + "_2"], min_periods=3, method="kendall"
    )
    return corr


@log_function_call
def study_similarity_plot(df, study_name, titles=None):
    df, merge_cols = create_mergeable_params_df(df)
    df["study_name"] = df["study_name"].apply(lambda x: get_name(x, titles=titles))

    study_names = df["study_name"].unique()
    similarities = pd.Series(0, index=study_names, dtype=float)
    for study_name_i in study_names:
        similarities[study_name_i] = study_similarity(
            df, study_name, study_name_i, merge_cols
        )
    similarities.sort_values(ascending=True, inplace=True)

    fig, ax = plt.subplots(dpi=300, figsize=(8, 2 + 0.1 * len(similarities)))
    ax.barh(
        similarities.index,
        similarities,
        color=grey_edge_color,
        edgecolor=grey_edge_color,
        alpha=0.8,
        linewidth=0.5,
    )
    if SHOW_TITLE:
        ax.set_title(
            f"Similarity of trials between {get_name(study_name, titles=titles)} and other studies"
        )
    ax.set_xlabel("Correlation of Accuracy Across Matching Trials")
    fig.tight_layout()
    return fig


@log_function_call
def study_similarity_all_pairs_scatters_plot(df, study_name, titles=None):
    df, merge_cols = create_mergeable_params_df(df)
    df["study_name"] = df["study_name"].apply(lambda x: get_name(x, titles=titles))

    study_names = df["study_name"].unique()
    study_names = sorted(study_names, key=lambda x: x[::-1])

    x_names = [name for name in study_names if "synthetic_" in name]
    y_names = [name for name in study_names if "synthetic_" not in name]
    synthetic_comparison = len(x_names) > 1 and len(y_names) > 1
    if not synthetic_comparison:
        x_names = study_names
        y_names = study_names

    # clean for merging
    def prep(df):
        df = df.copy()
        df[merge_cols] = df[merge_cols].astype(str)
        df = df.groupby(merge_cols, dropna=False)[value_col].median().reset_index()
        return df

    value_col = "values_0"
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(
        nrows=len(x_names),
        ncols=len(y_names),
        dpi=300,
        figsize=(8, 5),
        sharex=True,
        sharey=True,
    )
    for i, study_name_1 in enumerate(x_names):
        for j, study_name_2 in enumerate(y_names):
            df1 = prep(df[df["study_name"] == study_name_1])
            df2 = prep(df[df["study_name"] == study_name_2])

            # df1 = df1[df1['values_0'] > df1['values_0'].quantile(0.25)]
            # df2 = df2[df2['values_0'] > df2['values_0'].quantile(0.25)]

            merge_cols_i = [
                col
                for col in merge_cols
                if (df1[col].nunique(dropna=False) > 1)
                and (df2[col].nunique(dropna=False) > 1)
            ]
            if len(merge_cols_i) == 0:
                continue

            df_merged = pd.merge(
                df1, df2, how="inner", on=merge_cols_i, suffixes=("_1", "_2")
            )
            df_merged = df_merged.replace(np.inf, np.nan)
            df_merged.dropna(subset=[value_col + "_1", value_col + "_2"], inplace=True)
            corr = df_merged[value_col + "_1"].corr(
                df_merged[value_col + "_2"], min_periods=3, method="kendall"
            )
            df_merged[value_col + "_1_rank"] = df_merged[value_col + "_1"].rank(
                pct=True
            )
            df_merged[value_col + "_2_rank"] = df_merged[value_col + "_2"].rank(
                pct=True
            )

            sns.regplot(
                y=value_col + "_1",  # "_1_rank",
                x=value_col + "_2",  # "_2_rank",
                data=df_merged,
                ax=axes[i, j],
                lowess=True,
                # ci=False,
                line_kws={"color": "black", "alpha": 0.8},
                scatter_kws={"alpha": 0.5, "s": 5},
            )
            if i == 0:
                axes[i, j].set_title(study_name_2, fontsize=4)
            else:
                axes[i, j].set_title("", fontsize=4)
            axes[i, j].set_xlabel("", fontsize=4)
            if j == 0:
                axes[i, j].set_ylabel(study_name_1, fontsize=4)
            else:
                axes[i, j].set_ylabel("", fontsize=4)
            axes[i, j].tick_params(axis="both", which="major", labelsize=4)
            axes[i, j].tick_params(axis="both", which="minor", labelsize=2)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            # set axis border color
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(cmap((corr + 1) / 2))
                # spine.set_linewidth(0.5)

    if synthetic_comparison:
        suptitle = (
            "Accuracy Similarty Scatters of Matching Workflows (synthetic vs original)"
        )
    else:
        suptitle = "Accuracy Similarty Scatters of Matching Workflows"

    if SHOW_TITLE:
        fig.suptitle(suptitle)

    fig.tight_layout()
    return fig, suptitle


@log_function_call
def study_similarity_all_pairs_plot(df, study_name, titles=None):
    df, merge_cols = create_mergeable_params_df(df)
    df["study_name"] = df["study_name"].apply(lambda x: get_name(x, titles=titles))

    study_names = df["study_name"].unique()
    # study_names = sorted(study_names, key=lambda x: x.replace('synthetic_',''))
    study_names = sorted(study_names, key=lambda x: x[::-1])

    y_names = [name for name in study_names if "synthetic_" in name]
    x_names = [name for name in study_names if "synthetic_" not in name]
    if len(x_names) <= 1 or len(y_names) <= 1:
        # if True:
        x_names = study_names
        y_names = study_names

    # iterate over all pairs of study_names
    corrs = pd.DataFrame(index=x_names, columns=y_names, dtype=float)
    for study1 in x_names:
        for study2 in y_names:
            if study1 == study2:
                corrs.loc[study1, study2] = 0
                continue
            corr = study_similarity(df, study1, study2, merge_cols)
            corrs.loc[study1, study2] = corr

    # use matplotlib to plot the correlation matrix
    fig, ax = plt.subplots(dpi=300, figsize=(8, 5))
    im = ax.imshow(corrs, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(len(corrs)):
        im.axes.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1, fill=True, color="white", edgecolor="white"
            )
        )

    # highlight the largest value in each row
    for i in range(corrs.shape[0]):
        sorted_indices = np.argsort(corrs.iloc[i, :].values)
        max_j = sorted_indices[-1]
        # second_max_j = sorted_indices[-2]
        # ax.add_patch(plt.Rectangle((second_max_j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='yellow', lw=2, alpha=0.5))
        ax.add_patch(
            plt.Rectangle(
                (max_j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                lw=1,
                alpha=1,
            )
        )

    # display the correlation value on each rectangle
    for i in range(corrs.shape[0]):
        for j in range(corrs.shape[1]):
            if i == j:
                continue
            ax.text(
                j,
                i,
                f"{corrs.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    ax.set_xticks(range(len(corrs.columns)))
    ax.set_yticks(range(len(corrs.index)))
    ax.set_xticklabels(corrs.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corrs.index, fontsize=8)
    if SHOW_TITLE:
        ax.set_title("Accuracy Similarty of Matching Workflows")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(
        "Rank Correlation (Kendall)", rotation=-90, va="bottom", fontsize=8
    )
    cbar.ax.tick_params(labelsize=5)
    fig.tight_layout()
    return fig


@log_function_call
def slowest_components_plot(df, study_name, limit=100):
    # df, merge_cols = create_mergeable_params_df(df[df['study_name'] == study_name])
    df, merge_cols = create_mergeable_params_df(df)

    stat_col = "trial_minutes"
    df[stat_col] = df["duration"] / pd.Timedelta(minutes=1)
    log_scale = True

    # stat_col = 'trial_time_to_eval_time_ratio'
    # df[stat_col] =  df['user_attrs_metric_flow_duration'] / df['user_attrs_metric_eval_duration']
    # log_scale = True

    # stat_col = 'trial_duration_to_latency_ratio'
    # df[stat_col] = (df['duration'] / pd.Timedelta(seconds=1)) / (df['values_1'] + 0.0001)
    # log_scale = True

    # stat_col = 'error_rate'
    # df[stat_col] = df['user_attrs_metric_num_errors'] / (df['user_attrs_metric_num_errors'] + df['user_attrs_metric_num_evaluations'])
    # log_scale = False

    if log_scale:
        # assert df[stat_col].min() > 0
        df[stat_col] = np.log10(np.maximum(df[stat_col], 0.0001))

    melted = df.melt(
        id_vars=stat_col,
        value_vars=merge_cols,
        var_name="component_name",
        value_name="component_value",
    )
    melted["component"] = (
        melted["component_name"] + ": " + melted["component_value"].astype(str)
    )
    melted.drop(columns=["component_name", "component_value"], inplace=True)

    grouped = melted.groupby(["component"], dropna=False)
    # stats = grouped[stat_col].mean().sort_values(ascending=True)
    stats = grouped[stat_col].agg(["max", "mean"])
    stats = stats.sort_values(by=["max", "mean"], ascending=[True, True])
    stats = stats.tail(limit)

    components = [melted.loc[melted["component"] == c, stat_col] for c in stats.index]

    fig, ax = plt.subplots(dpi=300, figsize=(8, 2 + 0.07 * len(stats)))
    parts = ax.violinplot(
        components,
        widths=0.9,
        vert=False,
        showmeans=True,
        showmedians=False,
        showextrema=True,
        bw_method=0.05,
    )
    for body in parts["bodies"]:
        body.set_facecolor(grey_edge_color)
        body.set_edgecolor(grey_edge_color)
        body.set_linewidth(0.5)
        body.set_alpha(0.5)
    for element in [parts["cbars"], parts["cmins"], parts["cmaxes"], parts["cmeans"]]:
        element.set_color(fore_color)
        element.set_linewidth(0.5)
        element.set_alpha(0.9)
    if log_scale:
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, pos: f"$10^{{{x:g}}}$")
        )
    ax.set_ylim(0.5, len(stats) + 1.5)
    ax.set_yticks(range(1, len(stats.index) + 1))
    ax.set_yticklabels(
        [truncate_label(c.replace("params_", ""), max_length=80) for c in stats.index]
    )
    if SHOW_TITLE:
        ax.set_title(f"{descriptive_name(stat_col)} (all studies)")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="y", which="major", labelsize=6)
    fig.tight_layout()
    return fig


def format_timedelta(td: pd.Timedelta):
    if pd.isna(td):
        return "N/A"
    elif abs(td) > pd.Timedelta(days=31):
        return f"{td / pd.Timedelta(weeks=1):.1f} weeks"
    elif abs(td) > pd.Timedelta(days=1):
        return f"{td / pd.Timedelta(days=1):.1f} days"
    elif abs(td) > pd.Timedelta(hours=1):
        return f"{td / pd.Timedelta(hours=1):.1f} hours"
    elif abs(td) > pd.Timedelta(minutes=1):
        return f"{td / pd.Timedelta(minutes=1):.1f} minutes"
    else:
        return f"{td / pd.Timedelta(seconds=1):.1f} seconds"


@log_function_call
def create_study_stats_table(df):
    now = pd.Timestamp.now(tz="utc")
    extra_groups = {}
    if "user_attrs_metric_llm_cost_total" in df:
        extra_groups = {
            "LLM Costs ($)": ("user_attrs_metric_llm_cost_total", "sum"),
        }
    study_stats = df.groupby("study_name").agg(
        **{
            "Study Name": ("study_name", "first"),
            "Successful": ("state", lambda x: np.nan),
            "Failed": ("state", lambda x: np.nan),
            "Running": (
                "state",
                lambda x: (x == optuna.trial.TrialState.RUNNING.name).sum(),
            ),
            "Waiting": (
                "state",
                lambda x: (x == optuna.trial.TrialState.WAITING.name).sum(),
            ),
            "Pruned": (
                "state",
                lambda x: (x == optuna.trial.TrialState.PRUNED.name).sum(),
            ),
            "Total": ("state", "size"),
            "Seeds": ("user_attrs_is_seeding", "sum"),
            "Start Date": ("datetime_start", "min"),
            "Last Update": ("datetime_complete", lambda x: now - x.max()),
            "Trials/hr": ("duration", lambda x: np.nan),
            "Sample Errors": (
                "user_attrs_metric_num_errors",
                lambda x: x.sum()
                / max(1, df.loc[x.index, "user_attrs_metric_num_total"].sum()),
            ),
            "Pruned Evals": ("user_attrs_metric_is_pruned", "mean"),
            **extra_groups,
        },
    )
    study_stats["Failed"] = df.groupby("study_name").apply(
        lambda x: (
            (x["state"] == optuna.trial.TrialState.FAIL.name)
            | (x["user_attrs_metric_failed"] == True)
        ).sum(),
        include_groups=False,
    )
    study_stats["Successful"] = df.groupby("study_name").apply(
        lambda x: (
            (x["state"] == optuna.trial.TrialState.COMPLETE.name)
            & (x["user_attrs_metric_failed"] != True)
        ).sum(),
        include_groups=False,
    )
    study_stats = study_stats.sort_values(by="Study Name")
    study_stats["Start Date"] = study_stats["Start Date"].dt.strftime("%Y-%m-%d")
    study_stats["Last Update"] = (
        df.groupby("study_name")
        .apply(
            lambda x: now
            - max(x["datetime_complete"].max(), x["datetime_start"].max()),
            include_groups=False,
        )
        .apply(format_timedelta)
    )
    study_stats["Trials/hr"] = df.groupby("study_name").apply(
        lambda x: 60
        * 60
        * len(x)
        / (
            max(x["datetime_complete"].max(), x["datetime_start"].max())
            - x["datetime_start"].min()
        ).total_seconds(),
        include_groups=False,
    )

    # Show stats for each trial failure type
    fail_reasons = [
        str(x) for x in df["user_attrs_metric_exception_class"].dropna().unique()
    ]
    fail_names = [
        s.replace("EmbeddingPreemptiveTimeoutError", "EmbedPreTimeout")
        for s in fail_reasons
    ]
    for fail_value, fail_name in zip(fail_reasons, fail_names):
        study_stats[fail_name] = df.groupby("study_name")[
            "user_attrs_metric_exception_class"
        ].apply(lambda x: (x == fail_value).sum())

    total_trials = df.groupby("study_name")["state"].size()
    running_rate = study_stats["Running"] / total_trials
    waiting_rate = study_stats["Waiting"] / total_trials
    pruned_rate = study_stats["Pruned"] / total_trials
    fail_rate = study_stats["Failed"] / total_trials

    # set styles
    table = (
        study_stats.style.set_properties(
            subset=["Study Name", "Start Date"], **{"text-align": "left"}
        )
        .hide(axis="index")
        .background_gradient(
            subset=["Successful"],
            cmap=normal_to_green,
            vmin=0,
            vmax=max(500, total_trials.max()),
        )
        .background_gradient(
            subset=["Running"],
            cmap=normal_to_yellow,
            gmap=running_rate,
            vmin=0,
            vmax=0.1,
        )
        .background_gradient(
            subset=["Waiting"],
            cmap=normal_to_yellow,
            gmap=waiting_rate,
            vmin=0,
            vmax=0.1,
        )
        .background_gradient(
            subset=["Pruned"],
            cmap=normal_to_yellow,
            gmap=pruned_rate,
            vmin=0,
            vmax=0.1,
        )
        .background_gradient(
            subset=["Failed"],
            cmap=normal_to_red,
            gmap=fail_rate,
            vmin=0,
            vmax=0.5,
        )
        .background_gradient(
            subset=["Sample Errors"],
            cmap=normal_to_bad,
            vmin=0,
            vmax=0.05,
        )
        .format("{:.0f}", subset=["Trials/hr"])
        .format("{:.2f}", subset=["LLM Costs ($)"])
        .format("{:.2%}", subset=["Sample Errors", "Pruned Evals"])
    )
    for fail_value, fail_name in zip(fail_reasons, fail_names):
        fail_rate = df.groupby("study_name")["user_attrs_metric_exception_class"].apply(
            lambda x: (x == fail_value).mean()
        )
        table = table.background_gradient(
            subset=[fail_name],
            axis=0,
            cmap=normal_to_bad,
            gmap=fail_rate,
            vmin=0,
            vmax=0.1,
        )
    return table


@log_function_call
def create_benchmark_plot_and_table(df, titles=None):
    df = df.copy()
    is_cost = is_cost_objective(df)
    if is_cost is None:
        print("Not creating benchmark plot and table because of mixed objectives.")
        return None, None

    objective_2 = get_objective_2_unit(is_cost=is_cost)
    df["study_name"] = df["study_name"].apply(lambda x: get_name(x, titles=titles))
    study_names = df["study_name"].unique()
    accuracy_cols = [
        "Baseline Accuracy",
        f"Best Accuracy (at Baseline {objective_2})",
        "Best Accuracy (Overall)",
        # "Best 3rd Party Accuracy",
    ]
    latency_cols = [
        f"Baseline {objective_2}",
        f"Best {objective_2} (at Baseline Accuracy)",
        f"Best {objective_2} (at Highest Accuracy)",
    ]
    stat_cols = accuracy_cols + latency_cols

    results = pd.DataFrame(index=study_names, columns=["Study Name"] + stat_cols)
    for study_name in study_names:
        study = df[df["study_name"] == study_name]
        idx = study["values_0"].idxmax()
        results.loc[study_name, "Best Accuracy (Overall)"] = study.loc[idx, "values_0"]
        results.loc[study_name, f"Best {objective_2} (at Highest Accuracy)"] = (
            study.loc[idx, "values_1"]
        )
        results.loc[study_name, "Study Name"] = study_name

        baselines = get_baselines(study)
        if baselines.shape[0] > 0:
            baseline_idx = baselines["values_0"].idxmax()
            results.loc[study_name, "Baseline Accuracy"] = baselines.loc[
                baseline_idx, "values_0"
            ]
            results.loc[study_name, f"Baseline {objective_2}"] = baselines.loc[
                baseline_idx, "values_1"
            ]

            # filter to best accuracy within X% of baseline latency
            filter = study["values_1"] <= 1.01 * baselines.loc[baseline_idx, "values_1"]
            idx = study.loc[filter, "values_0"].idxmax()
            results.loc[study_name, f"Best Accuracy (at Baseline {objective_2})"] = (
                study.loc[idx, "values_0"]
            )

            # filter to best latency within X% of baseline accuracy
            filter = study["values_0"] >= 0.99 * baselines.loc[baseline_idx, "values_0"]
            idx = study.loc[filter, "values_1"].idxmin()
            results.loc[study_name, f"Best {objective_2} (at Baseline Accuracy)"] = (
                study.loc[idx, "values_1"]
            )

        # add comparisons
        # comparisons = get_comparison_accuracies(study_name)
        # if comparisons.shape[0] > 0:
        #     results.loc[study_name, "Best 3rd Party Accuracy"] = comparisons[
        #         "Accuracy"
        #     ].max()

    # include bottom row of the average across all studies
    averages = results[stat_cols].mean(axis=0, skipna=False)
    averages.loc["Study Name"] = "Average"
    results.loc["average", :] = averages

    # style the table
    results_styled = (
        results.style.set_properties(subset=["Study Name"], **{"text-align": "left"})
        .set_table_styles(
            [
                {
                    "selector": "th:not(:first-child), td:not(:first-child)",
                    "props": [
                        ("width", "100px"),
                        ("white-space", "normal"),
                        ("word-wrap", "break-word"),
                    ],
                },
                {
                    "selector": "tr:last-child",
                    "props": [
                        ("border-top", "2px solid black"),
                    ],
                },
            ]
        )
        .hide(axis="index")
        .format(
            lambda x: "-" if np.isnan(x) else f"{x:.1%}",
            subset=accuracy_cols,
        )
        .format(
            lambda x: "-" if np.isnan(x) else f"{x:.2f}",
            subset=latency_cols,
        )
        .highlight_max(
            subset=accuracy_cols,
            axis=1,
            color="black",
            props="font-weight: bold",
        )
        .highlight_min(
            subset=latency_cols,
            axis=1,
            color="black",
            props="font-weight: bold",
        )
        .set_properties(subset=accuracy_cols, **{"background-color": "honeydew"})
        .set_properties(subset=latency_cols, **{"background-color": "bisque"})
    )

    # create accuracy bar chart
    fig, axes = plt.subplots(
        nrows=2, dpi=300, figsize=(2 + 1.25 * len(results), 2 * 3), sharex=False
    )
    ax = axes[0]
    results[accuracy_cols].plot(
        kind="bar",
        ax=ax,
        color=["silver", "limegreen", "forestgreen", "turquoise"],
        edgecolor="white",
        rot=0,
    )
    for i, p in enumerate(ax.patches):
        col = accuracy_cols[i // len(results)]
        if col == "Best 3rd Party Accuracy" and p.get_height() == 0:
            continue
        label = f"{p.get_height():.0%}"
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_title("Benchmark Accuracy (%) vs Baseline")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xticklabels([wrap(label.get_text(), 20) for label in ax.get_xticklabels()])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, results[accuracy_cols].replace(np.inf, np.nan).max().max() * 1.5)
    ax.legend(loc="best", fontsize=8)

    # Creat the latency bar chart
    ax: plt.Axes = axes[1]
    results[latency_cols].plot(
        kind="bar",
        ax=ax,
        color=["silver", "indianred", "brown"],
        edgecolor="white",
        rot=0,
    )
    for i, p in enumerate(ax.patches):
        col = latency_cols[i // len(results)]
        label = f"{p.get_height():.2f}"
        ax.annotate(
            label,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_title(f"Benchmark {objective_2} vs Baseline")
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xticklabels([wrap(label.get_text(), 20) for label in ax.get_xticklabels()])
    ax.set_yscale("log")
    _, ymax = ax.get_ylim()
    ax.set_ylim(top=np.exp(np.log(ymax) * 1.25))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    return fig, results_styled


@log_function_call
def plot_metric_variability(df, study_name):
    df, merge_cols = create_mergeable_params_df(df, strict=True)

    is_cost = is_cost_objective(df)
    if is_cost is None:
        print("Not creating metric variability plot because of mixed objectives.")
        return None

    merge_cols.append("study_name")
    df = df[merge_cols + ["values_0", "values_1"]].copy()
    df = df.replace(np.inf, np.nan).dropna(subset=["values_0", "values_1"])
    df = df[(df["values_0"] > 0) & (df["values_1"] > 0)]

    def variability(x):
        return x.std()

    # Perform the groupby operation
    stats = df.groupby(merge_cols, observed=True)[["values_0", "values_1"]].agg(
        **{
            "values_0": ("values_0", variability),
            "values_1": ("values_1", variability),
            "N": ("values_0", "size"),
        }
    )
    stats = stats[stats["N"] > 1]

    # plot the accuracy variability as a histogram
    fig, axes = plt.subplots(nrows=2, dpi=300, figsize=(8, 5))
    for col, ax in zip(stats.columns, axes):
        name = descriptive_name(col, is_cost=is_cost)
        stats.plot.hist(
            y=col, bins=50, ax=ax, color="red", edgecolor="darkred", label=name
        )
        ax.set_title(f"{name} Variability (std of {len(stats)} identical workflows)")
        ax.tick_params(axis="both", which="major", labelsize=8)
        if name in ["Accuracy"]:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


@log_function_call
def create_exceptions_table(df: pd.DataFrame, exceptions_table: pd.DataFrame):
    # limit to 25 rows
    table = exceptions_table.head(25)

    # set style props to word wrap long strings in 'user_attrs_metric_exception_message'
    styled_table = table.style.set_table_styles(
        [
            {
                "selector": "td.col1, th.col1",
                "props": [("max-width", "1000px"), ("min-width", "750px")],
            },
            {
                "selector": "td.col0, th.col0, td.col1, th.col1",
                "props": [
                    ("white-space", "normal"),
                    ("word-wrap", "break-word"),
                    ("text-align", "left"),
                ],
            },
        ]
    ).hide(axis="index")
    return styled_table


def title_by_rag_mode_minimal(row):
    title = ""
    title += (
        row["params_rag_mode"].replace("params_", "").replace("_", " ").capitalize()
    )
    llms_used = set(
        [
            row[col]
            for col in row.index
            if (col.endswith("_llm") or col.endswith("_llm_name"))
            and str(row[col]) not in ["", "nan", "None"]
        ]
    )
    if len(title) > 0:
        title += " using " + ", ".join(llms_used)
    return title


def title_by_rag_mode(row):
    title = row["params_rag_mode"].replace("_", " ").capitalize()
    if "params_llm_name" in row:
        title += f" ({row['params_llm_name']}"
    else:
        title += f" ({get_response_synthesizer_llm(row, prefix='params_')})"
    if "params_template_name" in row:
        template_name = get_template_name(row, prefix="params_")
        if not pd.isna(template_name) and template_name != "default":
            title += f", {template_name.capitalize()}"
    if "params_hyde_enabled" in row and row["params_hyde_enabled"] == True:
        title += ", HyDE"
    if "params_reranker_enabled" in row and row["params_reranker_enabled"] == True:
        title += ", Reranking"
    if (
        "params_additional_context_enabled" in row
        and row["params_additional_context_enabled"] == True
    ):
        title += ", Padding"
    if "params_few_shot_enabled" in row and row["params_few_shot_enabled"] == True:
        title += ", Few-shot"
    if (
        "params_rag_query_decomposition_enabled" in row
        and row["params_rag_query_decomposition_enabled"] == True
    ):
        title += ", Decomposition"
    return title


@log_function_call
def plot_all_paretos(df, study_names, titles=None):
    n = len(study_names)
    ncols = min(int(np.ceil(np.sqrt(n))), 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        dpi=300,
        figsize=(1 + 13 * ncols / 2, 1 + 6 * nrows / 2),
    )

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # pre-compute the pareto frontiers for each study
    paretos = []
    for k, (ax, study_name) in enumerate(zip(axes, study_names)):
        df_trials = df[df["study_name"] == study_name]
        objective_2_name = get_objective_2_name(data=df_trials)
        df_pareto = df_trials[df_trials["pareto"]].sort_values(
            ["values_0", "values_1"], ascending=[False, True]
        )
        df_pareto = df_pareto.rename(
            columns={"values_0": "Accuracy", "values_1": objective_2_name}
        )
        if len(df_pareto) > 0:
            df_pareto["Title"] = df_pareto.apply(title_by_rag_mode, axis=1)
        else:
            df_pareto["Title"] = ""
        paretos.append(df_pareto)

    # decide on the colors for all plots so they match
    df_all_paretos = pd.concat(paretos, axis=0, ignore_index=True)
    unique_titles = df_all_paretos["Title"].unique()
    color_dict = {}
    for i, title in enumerate(unique_titles):
        color_dict[title] = discrete_cmap(i / max(10, len(unique_titles)))

    # don't try to use consistent colors across subplots if there's too many
    if len(color_dict) > 20:
        color_dict = None

    assert len(paretos) == n
    assert len(axes) >= n

    for k, (ax, study_name, df_pareto) in enumerate(zip(axes, study_names, paretos)):
        df_trials = df[df["study_name"] == study_name]
        is_cost = is_cost_objective(df_trials)

        plot_pareto_plot(
            df_pareto,
            study_name,
            is_cost,
            df_trials,
            color_dict=color_dict,
            ax=ax,
            titles=titles,
            show_title=SHOW_TITLE,
        )

        if SHOW_TITLE:
            ax.set_title(get_name(study_name, titles=titles), fontsize=8)
        i = k // ncols
        j = k % ncols
        if j != 0:
            ax.set_ylabel("")
        if i != nrows - 1:
            ax.set_xlabel("")

    # hide unused axes
    for k in range(n, nrows * ncols):
        axes[k].axis("off")

    suptitle = f"Pareto-Frontiers of {n} Studies"
    if SHOW_TITLE:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, suptitle


@log_function_call
def all_parameters_all_studies_plot(
    df: pd.DataFrame,
    param_cols,
    group_col=None,
    group_quantiles=None,
    group_labels=None,
    titles=None,
):
    df, merge_cols = create_mergeable_params_df(df.copy())
    df["study_name"] = df["study_name"].apply(lambda x: get_name(x, titles=titles))
    param_cols = [c for c in param_cols if c in merge_cols]

    # convert param_cols from categorical to string values
    for col in param_cols + ["study_name"]:
        df[col] = df[col].astype(str)

    # prep column for stacked bar chart binning if necessary
    if group_col is None:
        group_col = "DUMMY GROUP"
        df[group_col] = 0
    elif group_quantiles is not None or group_labels is not None:
        df[group_col] = df.groupby("study_name")[group_col].transform(
            lambda x: np.nan
            if group_labels is not None and x.nunique() <= len(group_labels)
            else pd.qcut(x, group_quantiles, labels=group_labels, duplicates="drop")
        )
    n_groups = df[group_col].nunique(dropna=False)

    n = len(param_cols)
    ncols = min(int(np.floor(np.sqrt(n))), 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        dpi=300,
        figsize=(1 + 8 * ncols / 2, 1 + 4 * nrows / 2),
        sharex=True,
        sharey=False,
    )
    axes = axes.flatten()

    for ax, param_col in zip(axes, param_cols):
        p_col = [param_col] if isinstance(param_col, str) else list(param_col)

        # calculate frequency of pareto appearance across all pareto front solutions and all studies binned by accuracy
        on_pareto: pd.DataFrame
        on_pareto = (
            df[df["pareto"]]
            .groupby(p_col + [group_col], dropna=True, observed=False)["pareto"]
            .sum()
            / df["pareto"].sum()
        ).reset_index()
        on_pareto = on_pareto.pivot_table(
            index=p_col,
            columns=group_col,
            values="pareto",
            fill_value=0,
            observed=False,
        )
        on_pareto = on_pareto.loc[
            [label for label in on_pareto.index if label != "N/A"]
        ]  # ignore 'N/A' values

        if len(on_pareto) > 0:
            # sort by total
            sorted_totals = on_pareto.sum(axis=1).sort_values()
            on_pareto = on_pareto.loc[sorted_totals.index, :]
            on_pareto = on_pareto.tail(50)

            # stacked bar shart
            on_pareto.plot(
                kind="barh",
                stacked=True,
                ax=ax,
                color=[discrete_cmap(i / max(10, n_groups)) for i in range(n_groups)],
            )

            # add labels for bar totals
            for i, value in enumerate(sorted_totals):
                ax.text(
                    value,
                    i,
                    f" {value:.0%}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    alpha=0.7,
                    color=fore_color,
                )

            # clean-up axies
            labels = ax.get_yticklabels()
            labels = [truncate_label(label.get_text(), 25) for label in labels]
            ax.set_yticklabels(labels)
            if group_col == "DUMMY GROUP":
                ax.legend().remove()
            else:
                leg = ax.legend(loc="best", fontsize=5)
                leg.set_alpha(0.8)

        ax.set_ylabel("")
        ax.set_title(", ".join([descriptive_name(c) for c in p_col]), fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # study_prefix = df['study_name'].iloc[0].split('-')[0]
    n_studies = df["study_name"].nunique()
    suptitle = f"Parameter Pareto Appearance Rates (Across {n_studies} Studies)"
    if group_col != "DUMMY GROUP":
        suptitle += f" grouped by {descriptive_name(group_col)}"
    if SHOW_TITLE:
        fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    return fig, suptitle


@log_function_call
def what_correlates_with(df: pd.DataFrame, target: pd.Series, method="kendall"):
    assert (df.index == target.index).all()
    corrs = pd.DataFrame(columns=["Parameter", "Value", "Correlation", "Count"])

    for col in df.columns:
        n_uniques = df[col].nunique(dropna=False)

        if n_uniques <= 1:
            idx = len(corrs)
            count = df[col].count()
            corrs.loc[idx, :] = (col, "*", np.nan, count)
        elif n_uniques > 3 and pd.api.types.is_numeric_dtype(df[col].dtype):
            idx = len(corrs)
            count = df[col].count()
            corr = df[col].corr(target, method=method)
            corrs.loc[idx, :] = (col, "*", corr, count)
        else:
            values = df[col].astype(str)
            labels = values.unique()
            for label in labels:
                idx = len(corrs)
                onehot = values == label
                count = onehot.value_counts().min()
                corr = onehot.corr(target, method=method)
                corrs.loc[idx, :] = (col, label, corr, count)

    corrs["Strength"] = corrs.groupby("Parameter", group_keys=False).apply(
        lambda g: pd.Series(
            np.average(g["Correlation"].abs(), weights=g["Count"]), index=g.index
        ),
        include_groups=False,
    )

    corrs["Parameter"] = corrs["Parameter"].str.replace("params_", "")
    corrs["Value"] = corrs["Value"].transform(truncate_label)

    corrs = corrs.set_index(["Parameter", "Value"], drop=True)
    corrs = corrs.sort_values(
        by=["Strength", "Correlation", "Count"],
        ascending=[False, False, False],
        key=np.abs,
    )

    table = (
        corrs.style.background_gradient(subset=["Strength"], axis=0)
        .background_gradient(
            subset=["Correlation"],
            cmap=LinearSegmentedColormap.from_list(
                "red_white_green",
                [
                    (0.0, "darkred"),
                    (0.25, "red"),
                    (0.5, "white"),
                    (0.75, "green"),
                    (1.0, "darkgreen"),
                ],
            ),
            vmin=min(corrs["Correlation"].min(), -1 + (1 - corrs["Correlation"].max())),
            vmax=max(corrs["Correlation"].max(), 1 - (1 + corrs["Correlation"].min())),
            axis=0,
        )
        .format("{:+.1%}", subset=["Correlation"])
        .format("{:.1%}", subset=["Strength"])
        .set_table_styles(
            [
                {
                    "selector": ".row_heading, .col_heading, .index_name",
                    "props": [("text-align", "left")],
                }
            ],
        )
    )
    return table


def escape_latex(filename):
    filename = filename.replace("%", "pct")
    return filename


async def append_table(
    pdf: PdfPages, styled_df: pd.io.formats.style.Styler, show=True, title=None
):
    path = cfg.paths.results_dir / "insights"
    path.mkdir(parents=True, exist_ok=True)
    path = path.resolve()

    # save for latex
    if title is not None:
        tex_filename = str(path / f"Table {escape_latex(title)}.tex")
        print(f'Saved: "{tex_filename}"')
        tex = styled_df.to_latex(caption=title, convert_css=True)
        tex = re.sub(r"(\d)%", r"\1\\%", tex, flags=re.MULTILINE)
        tex = re.sub(r"^\\\s*(?:tr|td|th).*$", "", tex, flags=re.MULTILINE)
        with open(tex_filename, "w") as f:
            f.write(tex)

    # put in pdf
    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        try:
            await dfi.export_async(
                styled_df,
                temp.name,
                dpi=300,
                table_conversion="playwright_async",
                max_rows=1000,
                max_cols=100,
            )
            fig = plt.figure(dpi=300, figsize=(8, 5))
            img = plt.imread(temp.name)
            if title is not None and SHOW_TITLE:
                plt.gca().set_title(title)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            pdf.savefig(fig)

            # save as an image also
            if title is not None:
                image_filename = str(path / f"Table {title}.pdf")
                print(f'Saved: "{image_filename}"')
                fig.savefig(image_filename, format="pdf")

            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception as e:
            display(styled_df)
            print(f"Error saving table as image: {e}\n\n")


def get_title(fig: plt.Figure):
    title = fig.get_suptitle()
    if len(title) == 0:
        title = " - ".join([ax.get_title() for ax in fig.get_axes() if ax.get_title()])
    if len(title) == 0:
        title = ", ".join(
            [f"{ax.get_ylabel()} over {ax.get_xlabel()}" for ax in fig.get_axes()]
        )
    return title[:200].strip()


def append_figure(
    pdf: PdfPages, fig: plt.Figure, insights_prefix: str, show=True, title=None
):
    # save as image for latex
    title = title or get_title(fig)
    if len(title) > 0:
        path = cfg.paths.results_dir / "insights"
        path.mkdir(parents=True, exist_ok=True)
        path = path.resolve()
        for extension in ["pdf"]:  # , 'png'
            image_filename = (
                insights_prefix + slugify(f"{escape_latex(title)}") + f".{extension}"
            )
            image_filepath = str(path / image_filename)
            try:
                fig.savefig(
                    image_filepath, format=extension, dpi=300, bbox_inches="tight"
                )
                print(f'Saved: "{image_filepath}"')
            except Exception as e:
                print(f"Error saving figure: {e}")

    # put in pdf
    pdf.savefig(fig)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_retriever_pareto(
    df_trials, study_name, color_dict=None, ax=None, titles=None, show_title=True
):
    df_pareto_desc = df_trials[df_trials["pareto"] == True]
    df_pareto_desc = df_pareto_desc.sort_values(
        ["user_attrs_metric_obj1_value", "user_attrs_metric_obj2_value"],
        ascending=[False, True],
    )
    px, py = get_pareto_square_points(
        df_pareto_desc["user_attrs_metric_obj2_value"],
        df_pareto_desc["user_attrs_metric_obj1_value"],
    )

    is_subplot = ax is not None
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    else:
        fig = None

    markersize = 6 if is_subplot else 10

    # plot all historical trials
    if df_trials is not None:
        ax.plot(
            df_trials["user_attrs_metric_obj2_value"],
            df_trials["user_attrs_metric_obj1_value"],
            "o",
            color=grey_face_color,
            markeredgecolor=grey_edge_color,
            markeredgewidth=0.5,
            alpha=0.25,
            label="All Trials",
            markersize=markersize,
        )

    # plot each individual trial on the Pareto-frontier
    ax.plot(px, py, "-", color=grey_edge_color, label="Pareto-Frontier")
    df_pareto_desc["Title"] = df_pareto_desc.apply(title_by_rag_mode, axis=1)
    labels = df_pareto_desc["Title"].unique()
    for i, label in enumerate(labels):
        if color_dict is None:
            color = discrete_cmap(i / max(10, len(labels)))
        else:
            color = color_dict[label]

        filter = df_pareto_desc["Title"] == label
        ax.plot(
            df_pareto_desc.loc[filter, "user_attrs_metric_obj2_value"],
            df_pareto_desc.loc[filter, "user_attrs_metric_obj1_value"],
            "o",
            color=color,
            markeredgewidth=0.5,
            markeredgecolor=(0, 0, 0, 0.9),
            label=wrap(label, 55 if is_subplot else 100),
            markersize=markersize,
            zorder=2,
        )

    # plot the baseline trials
    if df_trials is not None:
        baselines = get_baselines(df_trials)
        for i, (idx, trial) in enumerate(baselines.iterrows()):
            ax.plot(
                trial["user_attrs_metric_obj2_value"],
                trial["user_attrs_metric_obj1_value"],
                "s",
                alpha=1,
                color="none",  # Transparent center
                markeredgewidth=0.5,  # Bigger edge
                markeredgecolor="black",  # Black edge
                label=wrap(trial["title"], 55 if is_subplot else 100),
                markersize=markersize,
            )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # if is_subplot:
    ax.tick_params(axis="both", which="major", labelsize=12)
    title = f"{get_name(study_name, titles=titles)}"
    if show_title:
        ax.set_title(title, fontsize=18)

    if not is_subplot:
        # plot sota comparisons as horizontal lines
        comparisons = get_comparison_accuracies([study_name])
        linestyles = [":"]  # "--", "-.", ":", "-"

        for i, (_, (_, competitor, accuracy)) in enumerate(comparisons.iterrows()):
            if np.isnan(accuracy):
                continue
            linestyle = linestyles[i % len(linestyles)]
            ax.axhline(
                accuracy,
                color="black",
                linestyle=linestyle,
                linewidth=1.0,
                zorder=1,
            )
            # --------------------
            # align right
            # --------------------
            ax.text(
                0.85 * ax.get_xlim()[1],
                accuracy,
                competitor,
                color="black",
                fontsize=9,
                va="center",
                ha="right",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            )
            # --------------------
            # align left
            # --------------------
            # ax.text(
            #     1.3 * ax.get_xlim()[0],
            #     accuracy,
            #     competitor,
            #     color="black",
            #     fontsize=9,
            #     va="center",
            #     ha="left",
            #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
            # )

    ylim = ax.get_ylim()
    new_ylim_top = min(
        1, np.ceil(ylim[1] * 5) / 5
    )  # round ylim[1] to the next highest 20%
    ax.set_ylim([max(0, ylim[0]), new_ylim_top])

    apply_altair_like_styles(ax)

    if is_subplot:
        ax.margins(y=100)
        title = "Workflow Family" if show_title else f"{title} - Workflow Family"
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            fontsize=9,
            title=title,
            title_fontsize=10,
            alignment="left",
        )
    else:
        title = "Workflow Family" if show_title else f"{title} - Workflows"
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10,
            title=title,
            title_fontsize=12,
            alignment="left",
        )

    legend.get_title().set_fontweight("bold")

    ax.set_xlabel(df_pareto_desc["user_attrs_metric_objective_2_name"].unique()[0])
    ax.set_ylabel(df_pareto_desc["user_attrs_metric_objective_1_name"].unique()[0])

    return fig, title


async def create_pdf_report(
    study_name,
    all_study_names=None,
    pdf_filename=None,
    titles=None,
    insights_prefix=None,
    show=True,
):
    # load all studies
    df, study_stats_table, exceptions_table = load_studies(all_study_names)
    df_study = df[df["study_name"] == study_name]

    # show most interesting parameters first then append others
    param_cols = [
        (
            "params_rag_mode",
            "params_llm_name",
            "params_template_name",
        ),
        "params_rag_mode",
        "params_llm_name",
        "params_template_name",
        "params_rag_method",
        "params_rag_embedding_model",
        "params_splitter_method",
        "params_splitter_chunk_exp",
        "params_splitter_chunk_overlap_frac",
        "params_rag_top_k",
        "params_reranker_enabled",
        "params_hyde_enabled",
    ]
    param_cols += [
        c for c in df_study.columns if c.startswith("params_") and c not in param_cols
    ]

    def valid_param_set(cols, df):
        if isinstance(cols, str):
            cols = [cols]
        for c in cols:
            if c not in df.columns:
                return False
            if df[c].nunique(dropna=False) <= 1:
                return False
        return True

    param_cols = [c for c in param_cols if valid_param_set(c, df)]

    # start writing a pdf report
    if pdf_filename is None:
        cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.paths.results_dir.resolve()
        pdf_filename = str(path / f"insights_{study_name}.pdf")
    with PdfPages(pdf_filename) as pdf:
        # study stats table
        await append_table(pdf, study_stats_table, show=show, title="Study Stats")

        # exceptions table
        table = create_exceptions_table(df, exceptions_table)
        await append_table(pdf, table, show=show, title="Top Exceptions")

        # error here if the focus study isn't valid
        if len(df_study) == 0:
            raise ValueError(f"The focus study '{study_name}' contains 0 trials.")

        # benchmark accuracy table
        fig, table = create_benchmark_plot_and_table(df)
        if fig is not None:
            append_figure(pdf, fig, insights_prefix, show=show)
        if table is not None:
            await append_table(pdf, table, show=show, title="Benchmark Performance")

        # metric variability chart
        fig = plot_metric_variability(df, study_name)
        if fig is not None:
            append_figure(pdf, fig, insights_prefix, show=show)

        # all pareto fronts
        fig, title = plot_all_paretos(df, all_study_names, titles=titles)
        append_figure(pdf, fig, insights_prefix, title=title, show=show)

        # pareto front with descriptions
        fig, table, fig_title = pareto_plot_and_table(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, title=fig_title, show=show)
        await append_table(
            pdf,
            table,
            show=show,
            title=f"Pareto Frontier ({get_name(study_name, titles)})",
        )

        # optimization focus over time
        fig = focus_over_time_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # compute trial rate plot
        fig = compute_trial_rate_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # compute plot
        fig = compute_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # cost plot
        fig = cost_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # hist of trials
        fig = trial_duration_hist(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # historical pareto plots
        fig = pareto_comparison_plot(df, study_name, titles=titles)
        if fig is not None:
            append_figure(pdf, fig, insights_prefix, show=show)

        # pareto area plot
        fig = pareto_area_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # accuracy plot
        fig = accuracy_plot(df, study_name, titles=titles)
        append_figure(pdf, fig, insights_prefix, show=show)

        # latency plot
        fig = latency_plot(df, study_name)
        if fig is not None:
            append_figure(pdf, fig, insights_prefix, show=show)

        # plot summary of all params
        fig, title = all_parameters_all_studies_plot(
            df, param_cols, group_col="study_name", titles=titles
        )
        append_figure(pdf, fig, insights_prefix, title=title, show=show)

        # plots for each parameter
        for param_col in param_cols:
            # param stats across all studies
            # if df["study_name"].nunique() > 1:
            #     fig, title = param_plot_all_studies(df, study_name, param_col)
            #     if fig is not None:
            #         append_figure(pdf, fig, insights_prefix, title=title)

            # param stats of invidual study
            fig, title = param_plot(df, study_name, param_col, titles=titles)
            append_figure(pdf, fig, insights_prefix, title=title, show=show)

            # parameter pareto frontier for individual study
            if (
                isinstance(param_col, str)
                and df[param_col].dtype == "O"
                and df[param_col].nunique(dropna=False) <= 10
            ):
                fig, title = param_pareto_plot(df, study_name, param_col, titles=titles)
                append_figure(pdf, fig, insights_prefix, title=title, show=show)

        if df["study_name"].nunique() > 1:
            # single study similar to all others
            fig = study_similarity_plot(df, study_name, titles=titles)
            append_figure(pdf, fig, insights_prefix, show=show)

            # correlation matrix of all studies
            fig = study_similarity_all_pairs_plot(df, study_name, titles=titles)
            append_figure(pdf, fig, insights_prefix, show=show)

            # matrix display of scatter plots
            fig, title = study_similarity_all_pairs_scatters_plot(
                df, study_name, titles=titles
            )
            append_figure(pdf, fig, insights_prefix, title=title, show=show)

        # plot slowest components and params
        fig = slowest_components_plot(df, study_name)
        append_figure(pdf, fig, insights_prefix, show=show)

    return pdf_filename
