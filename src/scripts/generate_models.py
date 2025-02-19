#!/usr/bin/env python3
"""
Generate the models.py file for Astral AI.
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import yaml
from pathlib import Path
import re

from astral_ai.config.config import config

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def extract_date(model: str, alias: str) -> tuple:
    """
    Given a model name and its alias, extract a (year, month, day) tuple.
    Supports either a "MM-DD-YY" format (assumed two-digit year, 2000+) or
    an 8-digit "YYYYMMDD" format after the alias.
    Returns (0, 0, 0) if extraction fails.
    """
    suffix = model[len(alias):].lstrip("-")
    parts = suffix.split("-")
    if len(parts) == 3:
        try:
            month = int(parts[0])
            day = int(parts[1])
            year = int(parts[2])
            if year < 100:
                year += 2000
            return (year, month, day)
        except ValueError:
            return (0, 0, 0)
    if re.fullmatch(r"\d{8}", suffix):
        try:
            year = int(suffix[:4])
            month = int(suffix[4:6])
            day = int(suffix[6:8])
            return (year, month, day)
        except ValueError:
            return (0, 0, 0)
    return (0, 0, 0)

def to_camel_case(s: str) -> str:
    """
    Convert a string to CamelCase.
    Special-case: if the string represents "openai" (ignoring hyphens),
    return "OpenAI" instead of "Openai".
    For example: "openai" -> "OpenAI", "anthropic-ai" -> "AnthropicAi"
    """
    normalized = s.lower().replace("-", "")
    if normalized == "openai":
        return "OpenAI"
    return "".join(word.capitalize() for word in s.replace("-", " ").split())

# ------------------------------------------------------------
# Main Generation Routine
# ------------------------------------------------------------
def main():
    # Use our config to get the supported model types (as a set for O(1) lookups)
    supported_model_types = config.SUPPORTED_MODEL_TYPES

    # Load the YAML file from the config directory.
    yaml_path = Path("src/astral_ai/config/models.yaml")
    with yaml_path.open("r") as f:
        data = yaml.safe_load(f)

    # New structure: providers is the top-level key.
    providers_data = data["providers"]

    # Use lists to preserve order.
    providers_list = list(providers_data.keys())
    aliases_list = []
    model_ids_list = []
    definitions = {}            # Map alias -> definition dictionary
    model_id_to_provider = {}   # Map model_id -> provider

    # Iterate over providers in order.
    for provider in providers_list:
        provider_entry = providers_data[provider]
        models_list = provider_entry.get("models", [])
        for m in models_list:
            model_type = m["model_type"]
            # Validate that model_type is supported.
            if model_type not in supported_model_types:
                raise ValueError(
                    f"Model type '{model_type}' for alias '{m['alias']}' is not supported. Supported types: {supported_model_types}"
                )
            alias = m["alias"]
            if alias not in aliases_list:
                aliases_list.append(alias)
            alias_mapped_model = m.get("alias_mapped_model")
            model_ids = m["model_names"]
            pricing = m["pricing"]

            for model_id in model_ids:
                if model_id not in model_ids_list:
                    model_ids_list.append(model_id)
                model_id_to_provider[model_id] = provider

            # Determine the most recent model by sorting using the extracted date.
            try:
                sorted_ids = sorted(model_ids, key=lambda x: extract_date(x, alias))
                most_recent_model = sorted_ids[-1]
            except Exception:
                most_recent_model = model_ids[-1]

            definitions[alias] = {
                "provider": provider,
                "model_ids": model_ids,
                "pricing": {**pricing, "per_million": 1000000},
                "most_recent_model": alias_mapped_model or most_recent_model,
                "model_type": model_type,
            }

    # Begin generating the output file.
    output_lines = []
    output_lines.append("from typing import Literal, Dict, TypeAlias\n\n")
    output_lines.append("# Auto-generated types and constants\n\n")

    # ModelProvider literal (in YAML order).
    output_lines.append("ModelProvider = Literal[\n")
    for p in providers_list:
        output_lines.append(f'    "{p}",\n')
    output_lines.append("]\n\n")

    # ModelAlias literal (in order of appearance).
    output_lines.append("ModelAlias = Literal[\n")
    for a in aliases_list:
        output_lines.append(f'    "{a}",\n')
    output_lines.append("]\n\n")

    # ModelId literal (in order of appearance).
    output_lines.append("ModelId = Literal[\n")
    for mid in model_ids_list:
        output_lines.append(f'    "{mid}",\n')
    output_lines.append("]\n\n")

    # ModelName union type.
    output_lines.append("ModelName: TypeAlias = Literal[ModelId, ModelAlias]\n\n")

    # MODEL_DEFINITIONS mapping.
    output_lines.append("MODEL_DEFINITIONS = {\n")
    for a in aliases_list:
        defn = definitions[a]
        provider = defn["provider"]
        model_ids = defn["model_ids"]
        pricing = defn["pricing"]
        most_recent = defn["most_recent_model"]
        model_type = defn["model_type"]
        output_lines.append(f'    "{a}": {{\n')
        output_lines.append(f'        "provider": "{provider}",\n')
        output_lines.append(f'        "model_type": "{model_type}",\n')
        output_lines.append(f'        "model_ids": {model_ids},\n')
        output_lines.append(f'        "pricing": {pricing},\n')
        output_lines.append(f'        "most_recent_model": "{most_recent}"\n')
        output_lines.append("    },\n")
    output_lines.append("}\n\n")

    # PROVIDER_MODEL_NAMES mapping.
    output_lines.append("PROVIDER_MODEL_NAMES: Dict[ModelName, ModelProvider] = {\n")
    for mid in model_ids_list:
        provider = model_id_to_provider[mid]
        output_lines.append(f'    "{mid}": "{provider}",\n')
    for a in aliases_list:
        provider = definitions[a]["provider"]
        output_lines.append(f'    "{a}": "{provider}",\n')
    output_lines.append("}\n\n")

    # Build provider-to-model mapping (preserving order).
    provider_to_models = {p: [] for p in providers_list}
    for mid in model_ids_list:
        provider = model_id_to_provider[mid]
        provider_to_models[provider].append(mid)
    for a in aliases_list:
        provider = definitions[a]["provider"]
        provider_to_models[provider].append(a)

    # Dynamically generate Literals for each provider.
    for provider in providers_list:
        var_name = f"{to_camel_case(provider)}Models"
        models = provider_to_models[provider]
        output_lines.append(f"{var_name} = Literal[\n")
        for m in models:
            output_lines.append(f'    "{m}",\n')
        output_lines.append("]\n\n")

    # Append helper functions.
    output_lines.append(
        "# ------------------------------------------------------------\n"
        "# Get Provider from Model Name\n"
        "# ------------------------------------------------------------\n\n"
    )
    output_lines.append(
        "def get_provider_from_model_name(model_name: ModelName) -> ModelProvider:\n"
        '    """\n'
        "    Get the provider from a model name.\n"
        '    """\n'
        "    return PROVIDER_MODEL_NAMES[model_name]\n\n"
    )
    output_lines.append(
        "# ------------------------------------------------------------\n"
        "# Is Model Alias\n"
        "# ------------------------------------------------------------\n\n"
    )
    output_lines.append(
        "def is_model_alias(model_name: ModelName) -> bool:\n"
        '    """\n'
        "    Check if a model name is a model alias.\n"
        '    """\n'
        "    return model_name in MODEL_DEFINITIONS\n\n"
    )
    output_lines.append(
        "# ------------------------------------------------------------\n"
        "# Get Model from Model Alias\n"
        "# ------------------------------------------------------------\n\n"
    )
    output_lines.append(
        "def get_model_from_model_alias(model_alias: ModelAlias) -> ModelId:\n"
        '    """\n'
        "    Get the model from a model alias.\n"
        '    """\n'
        "    return MODEL_DEFINITIONS[model_alias][\"most_recent_model\"]\n"
    )

    # Write the output file.
    target_path = Path("src/astral_ai/_models.py")
    target_path.write_text("".join(output_lines))
    print(f"Generated {target_path}")

if __name__ == "__main__":
    main()
