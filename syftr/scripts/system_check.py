from pathlib import Path
from sqlalchemy.exc import OperationalError

from syftr import __version__ as version
from syftr.configuration import SYFTR_CONFIG_FILE_ENV_NAME, cfg
from syftr.optuna_helper import get_study_names
from syftr.studies import ALL_LLMS


def print_into():
    print()
    print(rf"""Welcome to
 ___  _  _  ____  ____  ____ 
/ __)( \/ )( ___)(_  _)(  _ \
\__ \ \  /  )__)   )(   )   /
(___/ (__) (__)   (__) (_)\_)

version {version}.
Running system check...""")
    print()
    return True


def check_config():
    file_locations = cfg.model_config["yaml_file"]
    for file_location in reversed(file_locations):
        if str(file_location) != ".":
            if Path(file_location).is_file():
                print(
                    "Your effetive config.yaml file is:", Path(file_location).absolute()
                )
                print()
                return True
    print("No config.yaml file found.")
    print("Please create a config.yaml file in one of the following locations:")
    for file_location in reversed(file_locations):
        if str(file_location) != ".":
            print(f" - {file_location}")
    print(
        f"or specify its path using the environment variable {(SYFTR_CONFIG_FILE_ENV_NAME)}."
    )
    print("The README.md file contains an example config.yaml file.")
    return False


def check_database():
    try:
        study_names = get_study_names(".*")
        print(f"Database connection successful. We found {len(study_names)} studies.")
        print()
    except OperationalError:
        print("Postgres database connection failed.")
        print("Please check your database settings.")

        print("Once you have installed PostgreSQL,")
        print("you can do the following setup from a Linux bash:")
        print()
        print("  sudo -u postgres psql")
        print("  CREATE USER syftr WITH PASSWORD 'your_password';")
        print("  CREATE DATABASE syftr WITH OWNER syftr;")
        print(r"  \q")
        print()
        print("In your config.yaml file, set the following:")
        print()
        print("postgres:")
        print('  dsn: "postgresql://syftr:your_password@localhost:5432/syftr"')
        print()
        print("You may need to adjust hostname and port depending on your setup.")
        return False
    return True


# def check_llms():
#     for name, llm in ALL_LLMS.items():
#         llm.


CHECKS = [
    print_into,
    check_config,
    check_database,
    # check_llms
]


def check():
    for check in CHECKS:
        if not check():
            print()
            print("You can run this script again to check your progress.")
            print()
            return False
    print("All checks passed.")
    print("You are good to go!")
    print()


if __name__ == "__main__":
    check()
