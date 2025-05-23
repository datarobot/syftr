import json

import numpy as np
import pandas as pd
from paretoset import paretoset
from sklearn.cluster import KMeans

from syftr.configuration import cfg
from syftr.huggingface_helper import get_embedding_model
from syftr.logger import logger
from syftr.optuna_helper import get_completed_trials
from syftr.studies import StudyConfig


def get_n_pareto_fronts(df_tmp, max_n: int = 1) -> pd.DataFrame:
    assert max_n > 0, "Wrong number of Pareto fronts"
    dfs = []
    df_tmp = df_tmp.copy()
    df_tmp["front"] = 1
    mask = paretoset(df_tmp[["values_0", "values_1"]], sense=["max", "min"])
    dfs.append(df_tmp[mask])
    if max_n > 1:
        df_next = df_tmp[~mask]
        df_pareto = get_n_pareto_fronts(df_next, max_n - 1)
        df_pareto["front"] += 1
        dfs.append(df_pareto)
    return pd.concat(dfs, ignore_index=True)


def get_top_performing_trials(
    df_trials: pd.DataFrame, max_pareto_fronts: int = 1
) -> pd.DataFrame:
    logger.info("Getting top-performing flows")
    df_result = df_trials.groupby("study_name").apply(
        lambda x: get_n_pareto_fronts(x, max_pareto_fronts)
    )
    df_result = df_result.groupby("user_attrs_flow", as_index=False).first()
    return df_result.sort_index()


def get_selected_trials(
    df_top: pd.DataFrame, embedding_model: str, max_total: int
) -> pd.DataFrame:
    logger.info(f"Clustering flows for selection using '{embedding_model}'")
    embedder, _ = get_embedding_model(embedding_model)
    assert embedder, "No embedding model"
    flows = list(df_top["user_attrs_flow"].values)
    embeddings = np.array([embedder.get_query_embedding(text) for text in flows])
    kmeans = KMeans(n_clusters=max_total, random_state=0)
    df_top["cluster"] = kmeans.fit_predict(embeddings)
    return df_top.loc[df_top.groupby("cluster")["front"].idxmin()].sort_index()


def get_examples(study_config: StudyConfig):
    study_names = study_config.transfer_learning.studies
    assert study_names, "No studies for transfer learning"
    max_fronts = study_config.transfer_learning.max_fronts
    max_total = study_config.transfer_learning.max_total
    success_rate = study_config.transfer_learning.success_rate

    storage = cfg.database.get_optuna_storage()
    df_trials = get_completed_trials(
        study=study_names, storage=storage, success_rate=success_rate
    )
    df_top = get_top_performing_trials(df_trials, max_fronts)

    if len(df_top) <= max_total:
        return [json.loads(flow) for flow in df_top["user_attrs_flow"]]

    embedding_model = study_config.transfer_learning.embedding_model
    df_selected = get_selected_trials(df_top, embedding_model, max_total)
    return [json.loads(flow) for flow in df_selected["user_attrs_flow"]]
