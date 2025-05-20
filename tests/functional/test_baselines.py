from syftr.baselines import set_baselines
from syftr.storage import HotPotQAHF, PartitionMap
from syftr.studies import StudyConfig


def test_set_baselines():
    study_config = StudyConfig(
        name="test-embedding-models",
        dataset=HotPotQAHF(
            partition_map=PartitionMap(test="sample"),
        ),
    )
    study_config.optimization.raise_on_invalid_baseline = True
    set_baselines(study_config)
