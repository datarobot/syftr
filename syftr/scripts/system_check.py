import logging
import queue
import threading
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.text import Text
from sqlalchemy.exc import OperationalError
from transformers import AutoConfig

from syftr import __version__
from syftr.configuration import SYFTR_CONFIG_FILE_ENV_NAME, cfg
from syftr.llm import get_llm
from syftr.optuna_helper import get_study_names
from syftr.studies import ALL_LLMS, DEFAULT_EMBEDDING_MODELS
from syftr.studyconfig_helper import build_example_config

llms = []
embedding_models = []

console = Console()


def print_into():
    ascii_art = rf"""
Welcome to
 ___  _  _  ____  ____  ____
/ __)( \/ )( ___)(_  _)(  _ \
\__ \ \  /  )__)   )(   )   /
(___/ (__) (__)   (__) (_)\_)

version {__version__}.
Running system check..."""
    # Print the ASCII art as a Text object, ensuring it's treated literally
    # and doesn't get misinterpreted by console markup.
    console.print(Text(ascii_art))
    return True


def check_config():
    file_locations_str = [
        str(location) for location in reversed(cfg.model_config["yaml_file"])
    ]

    potential_paths = [Path(loc) for loc in file_locations_str if loc != "."]

    existing_config_files = [str(p) for p in potential_paths if p.is_file()]

    if existing_config_files:
        console.print("Syftr will load configuration from the following files:")

        # Display the files that were actually found and will be loaded
        console.print("\n[green]Found:[/green]")
        for f_path in existing_config_files:
            console.print(f"- {f_path}")

        console.print(
            "\nConfiguration values are merged, with files listed earlier taking precedence over later ones."
        )
        # console.print(f"The final effective configuration is available via `cfg`.") # This was in my previous version, user removed it.
        console.print()
        return True

    # If no files were found, guide the user.
    # Show the locations that were checked.
    # To match user's logic, we base this on the file_locations_str
    checked_locations_display = [
        str(Path(loc).absolute()) for loc in file_locations_str if loc != "."
    ]

    console.print(
        "[yellow]No syftr configuration files (e.g., config.yaml, .syftr.yaml) were found.[/yellow]"
    )
    console.print("Please create a configuration file in one of these locations:")

    for loc in checked_locations_display:
        console.print(f"- {loc}")

    console.print(f"""
or specify its path using the environment variable {SYFTR_CONFIG_FILE_ENV_NAME}.
The README.md file contains an example config.yaml file.
""")
    return False


def check_database():
    db_connections = []
    # Ensure dsn is not None and has hosts method
    if cfg.database and "sqlite" in cfg.database.dsn.unicode_string():
        if not cfg.ray.local:
            console.print(
                "Found default configuration of SQLite. SQLite cannot be used in non-local mode."
            )
            console.print(
                "Set ray.local = True or configure PostgreSQL for advanced usage."
            )
            return False
        console.print("Found default configuration of SQLite. Should work locally...")
        console.print("For advanced usage, consider configuring PostgreSQL.")
        return True
    if (
        cfg.database
        and cfg.database.dsn
        and hasattr(cfg.database.dsn, "hosts")
        and callable(cfg.database.dsn.hosts)
    ):
        try:
            # hosts() might return a list of dicts or a list of pydantic models
            parsed_hosts = cfg.database.dsn.hosts()
            if parsed_hosts:
                for host_info in parsed_hosts:
                    if isinstance(host_info, dict):
                        host = host_info.get("host")
                        port = host_info.get("port")
                    elif hasattr(host_info, "host") and hasattr(
                        host_info, "port"
                    ):  # Pydantic model like
                        host = host_info.host
                        port = host_info.port
                    else:
                        db_connections.append(f"Unknown host structure: {host_info}")
                        continue

                    if host and port:
                        db_connections.append(f"{host}:{port}")
                    elif host:
                        db_connections.append(f"{host} (default port)")
                    else:
                        db_connections.append("Invalid host entry")
            else:  # If hosts() returns empty or None
                db_connections.append(f"No host information in DSN: {cfg.database.dsn}")

        except Exception as e:
            db_connections.append(
                f"Could not parse DSN: {cfg.database.dsn}. Error: {e}"
            )
    elif cfg.database and cfg.database.dsn:
        db_connections.append(f"Attempting direct DSN: {cfg.database.dsn}")
    else:
        db_connections.append("Postgres DSN not configured.")

    console.print("Checking connection to database(s) based on DSN:")
    if db_connections:
        for conn_info in db_connections:
            console.print(f"- {conn_info}")
    else:
        console.print(
            "- [yellow]No database connection details found in configuration.[/yellow]"
        )
    console.print()

    try:
        study_names = get_study_names(".*")
        console.print(
            f"[green]Database connection successful. We found {len(study_names)} studies.[/green]\n"
        )
    except OperationalError as e:
        console.print("[bold red]Postgres database connection failed.[/bold red]")
        console.print(f"[bold red]Error details: {e}[/bold red]")
        console.print(
            "[yellow]Please check your database settings and configuration.[/yellow]"
        )
        console.print(
            Text(
                """
Once you have installed PostgreSQL, you can do the following setup from a Linux bash:

  sudo -u postgres psql
  CREATE USER syftr WITH PASSWORD 'your_password';
  CREATE DATABASE syftr WITH OWNER syftr;
  \\q

In your config.yaml file, ensure the 'postgres.dsn' is correctly set, for example:

postgres:
  dsn: "postgresql://syftr:your_password@localhost:5432/syftr"

You may need to adjust the username, password, hostname, and port depending on your setup.
Make sure the PostgreSQL server is running and accessible from where you are running this script.
""",
                style="yellow",
            )
        )
        return False
    return True


def _check_single_llm_worker(llm_name: str, results_queue: queue.Queue):
    """
    Worker function for a thread to check a single LLM instance.
    Puts a tuple (llm_name, status, detail_string) into the results_queue.
    status can be "accessible", "inaccessible", "warning".
    """
    try:
        llm_instance = get_llm(llm_name)

        # Perform a simple synchronous completion test.
        response = llm_instance.complete("Just respond with the phrase `[OK]'")

        if response and hasattr(response, "text") and response.text:
            response_snippet = (
                response.text[:70].replace("\n", " ") + "..."
                if len(response.text) > 70
                else response.text.replace("\n", " ")
            )
            results_queue.put(
                (llm_name, "accessible", f'Responded: "{response_snippet}"')
            )
        else:
            results_queue.put(
                (
                    llm_name,
                    "warning",
                    "Connected but received an empty or unexpected response.",
                )
            )
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e).replace("\n", " ")
        error_snippet = (
            error_message[:200] + "..." if len(error_message) > 200 else error_message
        )
        results_queue.put((llm_name, "inaccessible", f"{error_type}: {error_snippet}"))


def check_llms():
    """
    Checks accessibility of all LLMs defined in ALL_LLMS using threads.
    Reports a summary after all checks are complete.
    This function is synchronous and integrates with the existing synchronous check workflow.
    """
    global llms  # noqa: PLW0602, PLW0603

    console.print("Checking configured Large Language Models (LLMs)...")

    console.print(
        f"Preparing to check {len(ALL_LLMS)} LLM(s) concurrently using threads: {', '.join(ALL_LLMS)}"
    )

    results_queue = queue.Queue()
    threads = []

    for llm_name in ALL_LLMS:
        thread = threading.Thread(
            target=_check_single_llm_worker, args=(llm_name, results_queue)
        )
        threads.append(thread)
        thread.start()

    console.print(
        f"Launched {len(threads)} threads for LLM checks. Waiting for completion..."
    )

    for thread in threads:
        thread.join()

    console.print("All LLM check threads completed.")

    # Collect results
    results_data = []
    while not results_queue.empty():
        results_data.append(results_queue.get())

    accessible_llms = []
    inaccessible_llms = []
    warning_llms = []

    for result_item in results_data:
        if isinstance(result_item, tuple) and len(result_item) == 3:
            name, status, detail = result_item
            if status == "accessible":
                accessible_llms.append({"name": name, "detail": detail})
            elif status == "inaccessible":
                inaccessible_llms.append({"name": name, "detail": detail})
            elif status == "warning":
                warning_llms.append({"name": name, "detail": detail})
        else:
            console.print(
                f"[bold red]Unexpected result format from LLM check queue:[/bold red] {result_item}"
            )
            inaccessible_llms.append(
                {
                    "name": "Unknown LLM (processing error)",
                    "detail": f"Unexpected result: {result_item}",
                }
            )

    # Print Summary Report
    console.print("\n[bold underline]LLM Accessibility Report[/bold underline]")
    console.print(f"Total LLMs configured and checked: {len(ALL_LLMS)}")

    if accessible_llms:
        console.print(f"\n[green]Accessible LLMs ({len(accessible_llms)}):[/green]")
        for llm_info in accessible_llms:
            console.print(
                f"  [+] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}"
            )

    if warning_llms:
        console.print(f"\n[yellow]LLMs with Warnings ({len(warning_llms)}):[/yellow]")
        for llm_info in warning_llms:
            console.print(
                f"  [!] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}"
            )

    if inaccessible_llms:
        console.print(
            f"\n[bold red]Inaccessible LLMs ({len(inaccessible_llms)}):[/bold red]"
        )
        for llm_info in inaccessible_llms:
            console.print(
                f"  [-] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}"
            )

    console.print("-" * 60)

    llms = [a["name"] for a in accessible_llms]

    if not accessible_llms:
        return False
    return True


@contextmanager
def suppress_logging(level=logging.WARNING):
    logger = logging.getLogger("syftr")
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def _check_single_embedding_worker(model_name: str, results_queue: queue.Queue):
    """
    Worker function for a thread to check a single embedding model instance.
    Puts a tuple (model_name, status, detail_string) into the results_queue.
    status can be "accessible" or "inaccessible".
    """
    try:
        with suppress_logging():  # suppress INFO and DEBUG logs
            AutoConfig.from_pretrained(model_name)
        results_queue.put(
            (
                model_name,
                "accessible",
                "available on HuggingFace.",
            )
        )
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e).replace("\n", " ")
        error_snippet = (
            error_message[:200] + "..." if len(error_message) > 200 else error_message
        )
        results_queue.put(
            (model_name, "inaccessible", f"{error_type}: {error_snippet}")
        )


def check_embedding_models():
    global embedding_models  # noqa: PLW0602, PLW0603

    console.print("Checking configured Embedding Models...")
    console.print(
        f"Preparing to check {len(DEFAULT_EMBEDDING_MODELS)} model(s) concurrently using threads: {', '.join(DEFAULT_EMBEDDING_MODELS)}"
    )

    results_queue = queue.Queue()
    threads = []

    for model_name in DEFAULT_EMBEDDING_MODELS:
        thread = threading.Thread(
            target=_check_single_embedding_worker, args=(model_name, results_queue)
        )
        threads.append(thread)
        thread.start()

    console.print(
        f"Launched {len(threads)} threads for embedding model checks. Waiting for completion..."
    )

    for thread in threads:
        thread.join()

    console.print("All embedding model check threads completed.")

    results_data = []
    while not results_queue.empty():
        results_data.append(results_queue.get())

    accessible_models = []
    inaccessible_models = []
    warning_models = []

    for result_item in results_data:
        if isinstance(result_item, tuple) and len(result_item) == 3:
            name, status, detail = result_item
            if status == "accessible":
                accessible_models.append({"name": name, "detail": detail})
            elif status == "inaccessible":
                inaccessible_models.append({"name": name, "detail": detail})
            elif status == "warning":
                warning_models.append({"name": name, "detail": detail})
        else:
            console.print(
                f"[bold red]Unexpected result format from embedding model check queue:[/bold red] {result_item}"
            )
            inaccessible_models.append(
                {
                    "name": "Unknown model (processing error)",
                    "detail": f"Unexpected result: {result_item}",
                }
            )

    console.print(
        "\n[bold underline]Embedding Model Accessibility Report[/bold underline]"
    )
    console.print(
        f"Total embedding models configured and checked: {len(DEFAULT_EMBEDDING_MODELS)}"
    )

    if accessible_models:
        console.print(
            f"\n[green]Accessible Embedding Models ({len(accessible_models)}):[/green]"
        )
        for info in accessible_models:
            console.print(f"  [+] [cyan]{info['name']:<25}[/cyan] - {info['detail']}")

    if warning_models:
        console.print(
            f"\n[yellow]Models with Warnings ({len(warning_models)}):[/yellow]"
        )
        for info in warning_models:
            console.print(f"  [!] [cyan]{info['name']:<25}[/cyan] - {info['detail']}")

    if inaccessible_models:
        console.print(
            f"\n[bold red]Inaccessible Embedding Models ({len(inaccessible_models)}):[/bold red]"
        )
        for info in inaccessible_models:
            console.print(f"  [-] [cyan]{info['name']:<25}[/cyan] - {info['detail']}")

    console.print("-" * 60)

    embedding_models = [a["name"] for a in accessible_models]

    if not accessible_models:
        return False
    return True


CHECKS = [
    print_into,
    check_config,
    check_database,
    check_llms,
    check_embedding_models,
]


def check():
    all_passed = True
    for check_func in CHECKS:
        console.rule(f"[bold blue]Running: {check_func.__name__}[/bold blue]")
        if not check_func():
            all_passed = False
            # Specific guidance is printed by the check function itself
            console.print(
                f"[yellow]Check '{check_func.__name__}' failed. Please review the messages above.[/yellow]\n"
            )
        else:
            console.print(f"[green]Check '{check_func.__name__}' passed.[/green]\n")

    console.rule("[bold blue]Summary[/bold blue]")
    if not all_passed:
        console.print("[bold red]One or more checks failed.[/bold red]")
        console.print("""
You can run this script again to check your progress after addressing the issues.
""")
        return False

    console.print("[bold green]All checks passed.[/bold green]")
    console.print("You are good to go!")
    console.print()

    if llms and embedding_models:
        _, paths = build_example_config(llms, embedding_models)
        console.print(
            "We generated an example configuration based on the available models."
        )
        console.print(
            "By default, the example is running 10 random and 10 optimization trials."
        )
        console.print("You can edit it and then run it from your project root with:")
        console.print()
        console.print(f"[yellow]syftr run {paths[0]}[/yellow]")
        console.print()
    return True


if __name__ == "__main__":
    check()
