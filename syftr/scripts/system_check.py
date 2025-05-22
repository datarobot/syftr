import threading
import queue
from pathlib import Path
from sqlalchemy.exc import OperationalError
from rich.console import Console # Import Console
from rich.text import Text # Import Text

from syftr import __version__ as version
from syftr.configuration import SYFTR_CONFIG_FILE_ENV_NAME, cfg
from syftr.optuna_helper import get_study_names
from syftr.studies import ALL_LLMS
from syftr.llm import get_llm

console = Console() # Initialize Console


def print_into():
    # Define the ASCII art string
    ascii_art = rf"""
Welcome to
 ___  _  _  ____  ____  ____
/ __)( \/ )( ___)(_  _)(  _ \
\__ \ \  /  )__)   )(   )   /
(___/ (__) (__)   (__) (_)\_)

version {version}.
Running system check..."""
    # Print the ASCII art as a Text object, ensuring it's treated literally
    # and doesn't get misinterpreted by console markup.
    console.print(Text(ascii_art))
    return True


def check_config():
    # Convert all locations to string and filter out "."
    # The user's provided code reverses here. Dynaconf loads files in the order provided,
    # with later files overriding earlier ones. So, if the user wants to show
    # "earlier taking precedence", the list should be processed as Dynaconf does.
    # However, the user's code has `reversed()`, so I will keep it to minimize changes.
    file_locations_str = [str(location) for location in reversed(cfg.model_config["yaml_file"])]
    
    # Get actual file paths for checking, excluding "."
    potential_paths = [Path(loc) for loc in file_locations_str if loc != "."]
    
    # Let's find files that actually exist from the potential paths
    existing_config_files = [str(p) for p in potential_paths if p.is_file()] # Use absolute for clarity

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
    # To match user's logic, we base this on the (potentially reversed) file_locations_str
    checked_locations_display = [str(Path(loc).absolute()) for loc in file_locations_str if loc != "."]
    if not checked_locations_display and "." in [str(loc) for loc in cfg.model_config["yaml_file"]]: # if only "." was in the original list and it was filtered out
        # This case needs to handle what to show if the original list was just "." or primarily "."
        # For simplicity, show the original list from cfg if checked_locations_display is empty.
         checked_locations_display = [str(Path(loc).absolute()) for loc in cfg.model_config["yaml_file"] if str(loc) != "."]
         if not checked_locations_display: # If still empty (e.g. original was just ["."])
             checked_locations_display = [str(Path(".").resolve()) + " (current directory)"]


    console.print("[yellow]No syftr configuration files (e.g., config.yaml, .syftr.yaml) were found.[/yellow]")
    console.print("Please create a configuration file in one of these locations:")
    
    # Use a list format for better readability
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
    if cfg.postgres and cfg.postgres.dsn and hasattr(cfg.postgres.dsn, 'hosts') and callable(cfg.postgres.dsn.hosts):
        try:
            # hosts() might return a list of dicts or a list of pydantic models
            parsed_hosts = cfg.postgres.dsn.hosts()
            if parsed_hosts: # Ensure it's not None or empty
                for host_info in parsed_hosts:
                    if isinstance(host_info, dict):
                        host = host_info.get('host')
                        port = host_info.get('port')
                    elif hasattr(host_info, 'host') and hasattr(host_info, 'port'): # Pydantic model like
                        host = host_info.host
                        port = host_info.port
                    else: # Fallback if structure is unexpected
                        db_connections.append(f"Unknown host structure: {host_info}")
                        continue
                    
                    if host and port:
                        db_connections.append(f"{host}:{port}")
                    elif host:
                        db_connections.append(f"{host} (default port)")
                    else:
                        db_connections.append("Invalid host entry")
            else: # If hosts() returns empty or None
                 db_connections.append(f"No host information in DSN: {cfg.postgres.dsn}")

        except Exception as e:
            db_connections.append(f"Could not parse DSN: {cfg.postgres.dsn}. Error: {e}")
    elif cfg.postgres and cfg.postgres.dsn: # If DSN is a simple string
        db_connections.append(f"Attempting direct DSN: {cfg.postgres.dsn}")
    else:
        db_connections.append("Postgres DSN not configured.")

    console.print("Checking connection to database(s) based on DSN:")
    if db_connections:
        for conn_info in db_connections:
            console.print(f"- {conn_info}")
    else:
        # This case should ideally be caught by the DSN parsing logic above,
        # but as a fallback:
        console.print("- [yellow]No database connection details found in configuration.[/yellow]")
    console.print() # Add a newline for spacing

    try:
        study_names = get_study_names(".*")
        console.print(f"[green]Database connection successful. We found {len(study_names)} studies.[/green]\n")
    except OperationalError as e: # Capture the exception as 'e'
        console.print("[bold red]Postgres database connection failed.[/bold red]")
        console.print(f"[bold red]Error details: {e}[/bold red]") # Print the exception details
        console.print("[yellow]Please check your database settings and configuration.[/yellow]")
        console.print(Text("""
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
""", style="yellow"))
        return False
    return True


def _check_single_llm_worker(llm_name: str, results_queue: queue.Queue):
    """
    Worker function for a thread to check a single LLM instance.
    Puts a tuple (llm_name, status, detail_string) into the results_queue.
    status can be "accessible", "inaccessible", "warning".
    """
    try:
        # Assuming get_llm is a synchronous function that prepares the LLM instance
        llm_instance = get_llm(llm_name)
        
        # Perform a simple synchronous completion test.
        # LlamaIndex LLMs usually have a `complete` method for synchronous calls.
        response = llm_instance.complete("Return the text `[OK]' (brackets included) and nothing else \\nothink")
        
        if response and hasattr(response, 'text') and response.text:
            response_snippet = response.text[:70].replace('\n', ' ') + "..." if len(response.text) > 70 else response.text.replace('\n', ' ')
            results_queue.put((llm_name, "accessible", f"Responded: \"{response_snippet}\""))
        else:
            results_queue.put((llm_name, "warning", "Connected but received an empty or unexpected response."))
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e).replace('\n', ' ')
        # Limit the length of the error message in the report
        error_snippet = error_message[:200] + "..." if len(error_message) > 200 else error_message
        results_queue.put((llm_name, "inaccessible", f"{error_type}: {error_snippet}"))


def check_llms():
    """
    Checks accessibility of all LLMs defined in ALL_LLMS using threads.
    Reports a summary after all checks are complete.
    This function is synchronous and integrates with the existing synchronous check workflow.
    """
    console.print("Checking configured Large Language Models (LLMs)...")
    
    console.print(f"Preparing to check {len(ALL_LLMS)} LLM(s) concurrently using threads: {', '.join(ALL_LLMS)}")

    results_queue = queue.Queue()
    threads = []

    for llm_name in ALL_LLMS:
        thread = threading.Thread(target=_check_single_llm_worker, args=(llm_name, results_queue))
        threads.append(thread)
        thread.start()

    console.print(f"Launched {len(threads)} threads for LLM checks. Waiting for completion...")

    for thread in threads:
        thread.join() # Wait for all threads to complete

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
            console.print(f"[bold red]Unexpected result format from LLM check queue:[/bold red] {result_item}")
            inaccessible_llms.append({"name": "Unknown LLM (processing error)", "detail": f"Unexpected result: {result_item}"})

    # Print Summary Report
    console.print("\n[bold underline]LLM Accessibility Report[/bold underline]")
    console.print(f"Total LLMs configured and checked: {len(ALL_LLMS)}")

    if accessible_llms:
        console.print(f"\n[green]Accessible LLMs ({len(accessible_llms)}):[/green]")
        for llm_info in accessible_llms:
            console.print(f"  [+] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}")
    
    if warning_llms:
        console.print(f"\n[yellow]LLMs with Warnings ({len(warning_llms)}):[/yellow]")
        for llm_info in warning_llms:
            console.print(f"  [!] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}")

    if inaccessible_llms:
        console.print(f"\n[bold red]Inaccessible LLMs ({len(inaccessible_llms)}):[/bold red]")
        for llm_info in inaccessible_llms:
            console.print(f"  [-] [cyan]{llm_info['name']:<25}[/cyan] - {llm_info['detail']}")
    
    console.print("-" * 60) 

    if inaccessible_llms or warning_llms:
        return False
    return True


CHECKS = [
    print_into,
    check_config,
    check_database,
    check_llms
]


def check():
    all_passed = True
    for check_func in CHECKS:
        console.rule(f"[bold blue]Running: {check_func.__name__}[/bold blue]")
        if not check_func():
            all_passed = False
            # Specific guidance is printed by the check function itself
            console.print(f"[yellow]Check '{check_func.__name__}' failed. Please review the messages above.[/yellow]\n")
            # No need to print "run again" here, do it once at the end if anything failed.
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
    return True


if __name__ == "__main__":
    check()
