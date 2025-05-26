import optuna

from syftr.configuration import cfg

# STUDY_NAME = "run9"
STUDY_NAME = "hotpot-test-fullwiki-toy"

storage = cfg.database.get_optuna_storage()
optuna.delete_study(study_name=STUDY_NAME, storage=storage)
