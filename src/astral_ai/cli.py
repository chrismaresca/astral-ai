# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# Python
import subprocess
import sys
import shutil
import typer


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
app = typer.Typer(help="Astral AI CLI for managing providers.")


def detect_package_manager() -> tuple[str, list[str]]:
    """
    Detects the available package manager (Poetry, uv, or pip) and returns
    the base command list used to add/install a package.
    """
    if shutil.which("poetry"):
        typer.echo("Detected Poetry")
        return ("poetry", ["poetry", "add"])
    elif shutil.which("uv"):
        typer.echo("Detected uv")
        return ("uv", ["uv", "add"])
    elif shutil.which("pip"):
        typer.echo("Detected pip")
        return ("pip", [sys.executable, "-m", "pip", "install"])
    else:
        typer.echo("No supported package manager detected.")
        raise typer.Exit(1)


@app.command("check-package-manager")
def check_package_manager():
    """
    Checks the available package manager (Poetry, uv, or pip) and returns
    the base command list used to add/install a package.
    """
    detect_package_manager()


@app.command("add-provider")
def add_provider(provider: str = typer.Argument(..., help="The provider to add, e.g., 'anthropic'")):
    """
    Installs the specified provider extra.
    For example, 'astral-ai add-provider anthropic' installs the 'anthropic' extra.
    """
    package_manager, base_command = detect_package_manager()
    package_spec = f"astral-ai[{provider}]"
    typer.echo(f"Detected package manager: {package_manager}")
    typer.echo(f"Installing {package_spec}...")

    command = base_command + [package_spec]
    typer.echo(f"Running command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        typer.echo(f"Provider '{provider}' installed successfully!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Failed to install provider '{provider}': {e}")
        raise typer.Exit(1)


def main():
    app()
