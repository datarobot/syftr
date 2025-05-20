from pathlib import Path

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.e2e.optimization import run_and_test_optimization

PARAM_LIST = [
    "llm_name",
    "rag_embedding_model",
    "rag_fusion_llm_name",
    "rag_fusion_mode",
    "rag_fusion_num_queries",
    "rag_method",
    "rag_mode",
    "rag_top_k",
    "splitter_chunk_overlap_frac",
    "splitter_chunk_exp",
    "splitter_method",
    "template_name",
]


def test_fusion_rag_seeding_local(request):
    study_config_file = Path(
        cfg.paths.test_studies_dir / "test-fusion-rag-seeding.yaml"
    )
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    if request.config.getoption("--gpu"):
        study_config.optimization.embedding_device = "cuda"
        study_config.optimization.gpus_per_trial = 0.25

    param_list = study_config.search_space.param_names(params=PARAM_LIST)

    df_trials = run_and_test_optimization(study_config, param_list)

    assert len(df_trials)
