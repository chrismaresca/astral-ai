# -------------------------------------------------------------------------------- #
# Generate Model Constants
# -------------------------------------------------------------------------------- #
# This script reads model configuration YAML files and generates type-safe constants
# for use in the codebase. It creates or updates the _models.py file in the constants
# directory with comprehensive type definitions and mappings.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import sys
import yaml
from typing import Dict, List, Set, Any, Tuple
from pathlib import Path

# -------------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------------- #
# Output file path
OUTPUT_FILE = "src/astral_ai/constants/_models.py"

# Input files
ALIAS_TO_MODELS_FILE = "src/astral_ai/config/model_aliases.yaml"
MODELS_FILE = "src/astral_ai/config/models.yaml"

# -------------------------------------------------------------------------------- #
# Utility Functions
# -------------------------------------------------------------------------------- #
def print_step(step: str) -> None:
    """Print a step in the generation process."""
    print(f"\n{'=' * 80}")
    print(f"STEP: {step}")
    print(f"{'=' * 80}")

def print_info(info: str) -> None:
    """Print information during the generation process."""
    print(f"INFO: {info}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"\n✅ {message}")

def load_yaml_file(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    print_info(f"Loading YAML file: {file_path}")
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            print_info(f"Successfully loaded {file_path}")
            return data
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        sys.exit(1)

def create_literal_string(values: List[str]) -> str:
    """Create a Literal type definition string from a list of values."""
    if not values:
        return "Literal[]  # Empty literal"
    
    # Format each value with quotes
    formatted_values = [f'"{value}"' for value in sorted(values)]
    
    # Join values with commas and newlines for readability
    joined_values = ",\n    ".join(formatted_values)
    
    return f"Literal[\n    {joined_values},\n]"

def write_to_file(content: str) -> None:
    """Write the generated content to the output file."""
    print_info(f"Writing generated constants to {OUTPUT_FILE}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    try:
        with open(OUTPUT_FILE, 'w') as file:
            file.write(content)
        print_success(f"Successfully wrote constants to {OUTPUT_FILE}")
    except Exception as e:
        print(f"ERROR: Failed to write to {OUTPUT_FILE}: {e}")
        sys.exit(1)

# -------------------------------------------------------------------------------- #
# Data Extraction Functions
# -------------------------------------------------------------------------------- #
def extract_providers(models_data: dict) -> List[str]:
    """Extract all provider names from the models data."""
    return list(models_data.get('providers', {}).keys())

def extract_model_aliases(alias_data: dict) -> List[str]:
    """Extract all model aliases from the alias data."""
    return list(alias_data.get('aliases', {}).keys())

def extract_model_ids(models_data: dict, alias_data: dict) -> List[str]:
    """Extract all model IDs from both models and alias data."""
    # Get model IDs from models.yaml
    model_ids = []
    for provider_data in models_data.get('providers', {}).values():
        model_ids.extend(provider_data.get('models', {}).keys())
    
    # Also get model IDs from alias_to_models.yaml for completeness
    for model_list in alias_data.get('aliases', {}).values():
        model_ids.extend(model_list)
    
    # Remove duplicates
    return list(set(model_ids))

def extract_provider_specific_models(provider: str, models_data: dict, alias_data: dict) -> List[str]:
    """Extract all models (IDs and aliases) specific to a provider."""
    provider_models = []
    
    # Get model IDs from the provider
    provider_data = models_data.get('providers', {}).get(provider, {})
    provider_model_ids = list(provider_data.get('models', {}).keys())
    provider_models.extend(provider_model_ids)
    
    # Get aliases that map to this provider's models
    for alias, model_list in alias_data.get('aliases', {}).items():
        # If any model in the list belongs to this provider, add the alias
        if any(model_id in provider_model_ids for model_id in model_list):
            provider_models.append(alias)
    
    return list(set(provider_models))

def extract_model_types(models_data: dict) -> List[str]:
    """Extract all unique model types from the models data."""
    model_types = set()
    
    for provider_data in models_data.get('providers', {}).values():
        for model_info in provider_data.get('models', {}).values():
            if 'model_type' in model_info:
                model_types.add(model_info['model_type'])
    
    return sorted(list(model_types))

def extract_completion_models(models_data: dict) -> List[str]:
    """Extract all models with model_type 'completion'."""
    completion_models = []
    
    for provider_data in models_data.get('providers', {}).values():
        for model_id, model_info in provider_data.get('models', {}).items():
            if model_info.get('model_type') == 'completion':
                completion_models.append(model_id)
    
    return completion_models

def build_provider_model_mapping(models_data: dict, alias_data: dict) -> Dict[str, str]:
    """Build a mapping of all model names (IDs and aliases) to their providers."""
    mapping = {}
    
    # Process model IDs from models.yaml
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id in provider_data.get('models', {}).keys():
            mapping[model_id] = provider
    
    # Process aliases
    for alias, model_list in alias_data.get('aliases', {}).items():
        if model_list:  # Ensure the list is not empty
            # Get the provider of the first model in the list
            # Assuming all models in an alias list belong to the same provider
            model_id = model_list[0]
            for provider, provider_data in models_data.get('providers', {}).items():
                if model_id in provider_data.get('models', {}):
                    mapping[alias] = provider
                    break
    
    return mapping

def build_model_resource_type_mapping(models_data: dict, alias_data: dict) -> Dict[str, str]:
    """Build a mapping of all model names (IDs and aliases) to their resource types."""
    mapping = {}
    
    # First map model IDs to their resource types
    for provider_data in models_data.get('providers', {}).values():
        for model_id, model_info in provider_data.get('models', {}).items():
            if 'model_type' in model_info:
                mapping[model_id] = model_info['model_type']
    
    # Then map aliases to resource types based on the first model in their list
    for alias, model_list in alias_data.get('aliases', {}).items():
        if model_list:  # Ensure the list is not empty
            model_id = model_list[0]
            # Find the resource type for this model ID
            for provider_data in models_data.get('providers', {}).values():
                if model_id in provider_data.get('models', {}):
                    model_info = provider_data['models'][model_id]
                    if 'model_type' in model_info:
                        mapping[alias] = model_info['model_type']
                    break
    
    return mapping

def get_provider_variable_name(provider: str) -> str:
    """Convert provider name to a proper variable name with correct capitalization."""
    if provider.lower() == "openai":
        return "OpenAIModels"
    elif provider.lower() == "deepseek":
        return "DeepSeekModels"
    else:
        # For other providers, capitalize first letter
        return provider.capitalize() + "Models"

# -------------------------------------------------------------------------------- #
# Main Script
# -------------------------------------------------------------------------------- #
def main():
    """Main function to generate model constants."""
    print_step("Starting model constants generation")
    
    # Load configuration files
    print_step("Loading configuration files")
    alias_data = load_yaml_file(ALIAS_TO_MODELS_FILE)
    models_data = load_yaml_file(MODELS_FILE)
    
    # Extract data
    print_step("Extracting data from configuration files")
    providers = extract_providers(models_data)
    print_info(f"Found providers: {', '.join(providers)}")
    
    model_aliases = extract_model_aliases(alias_data)
    print_info(f"Found {len(model_aliases)} model aliases")
    
    model_ids = extract_model_ids(models_data, alias_data)
    print_info(f"Found {len(model_ids)} model IDs")
    
    # Extract model types
    print_step("Extracting model types")
    model_types = extract_model_types(models_data)
    print_info(f"Found model types: {', '.join(model_types)}")
    
    # Extract provider-specific models
    print_step("Extracting provider-specific models")
    provider_models = {}
    for provider in providers:
        provider_models[provider] = extract_provider_specific_models(provider, models_data, alias_data)
        print_info(f"Found {len(provider_models[provider])} models for {provider}")
    
    # Extract completion models
    print_step("Extracting completion models")
    completion_models = extract_completion_models(models_data)
    print_info(f"Found {len(completion_models)} completion models")
    
    # Build provider model mapping
    print_step("Building provider model mapping")
    provider_model_mapping = build_provider_model_mapping(models_data, alias_data)
    print_info(f"Built mapping for {len(provider_model_mapping)} models")
    
    # Build model resource type mapping
    print_step("Building model to resource type mapping")
    model_resource_type_mapping = build_model_resource_type_mapping(models_data, alias_data)
    print_info(f"Built resource type mapping for {len(model_resource_type_mapping)} models")
    
    # Generate file content
    print_step("Generating file content")
    
    content = [
        "# -------------------------------------------------------------------------------- #",
        "# Model Constants",
        "# -------------------------------------------------------------------------------- #",
        "# This file is auto-generated by scripts/generate_model_constants.py",
        "# DO NOT EDIT MANUALLY",
        "# -------------------------------------------------------------------------------- #",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Imports",
        "# -------------------------------------------------------------------------------- #",
        "# Built-in imports",
        "from typing import Literal, Dict, TypeAlias",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Types",
        "# -------------------------------------------------------------------------------- #",
        "",
        f"ResourceType = {create_literal_string(model_types)}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Provider",
        "# -------------------------------------------------------------------------------- #",
        "",
        f"ModelProvider = {create_literal_string(providers)}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Alias and IDs",
        "# -------------------------------------------------------------------------------- #",
        "",
        f"ModelAlias = {create_literal_string(model_aliases)}",
        "",
        f"ModelId = {create_literal_string(model_ids)}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Breakdown by Provider",
        "# -------------------------------------------------------------------------------- #",
        "",
    ]
    
    # Add provider-specific model literals
    for provider in providers:
        provider_name = get_provider_variable_name(provider)
        content.extend([
            f"{provider_name} = {create_literal_string(provider_models[provider])}",
            "",
        ])
    
    content.extend([
        "# -------------------------------------------------------------------------------- #",
        "# Model Breakdown by Function",
        "# -------------------------------------------------------------------------------- #",
        "",
        "",
        f"CompletionModels = {create_literal_string(completion_models)}",
        "",
        "# EmbeddingModels = Literal[",
        "#     \"text-embedding-3-small\",",
        "#     \"text-embedding-3-large\",",
        "# ]",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# ALL MODEL NAMES",
        "# Type alias combining all provider models",
        "# -------------------------------------------------------------------------------- #",
        "",
        f"ModelName: TypeAlias = Literal[{', '.join([get_provider_variable_name(provider) for provider in providers])}]",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Name to Provider Mapping",
        "# -------------------------------------------------------------------------------- #",
        "",
        "PROVIDER_MODEL_NAMES: Dict[ModelName, ModelProvider] = {",
    ])
    
    # Add provider model mapping entries
    for model, provider in sorted(provider_model_mapping.items()):
        content.append(f"    \"{model}\": \"{provider}\",")
    
    content.append("}")
    
    # Add model to resource type mapping
    content.extend([
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Name to Resource Type Mapping",
        "# -------------------------------------------------------------------------------- #",
        "",
        "MODEL_NAMES_TO_RESOURCE_TYPE: Dict[ModelName, ResourceType] = {",
    ])
    
    # Add model resource type mapping entries
    for model, resource_type in sorted(model_resource_type_mapping.items()):
        content.append(f"    \"{model}\": \"{resource_type}\",")
    
    content.append("}")
    
    # Write the content to the output file
    write_to_file("\n".join(content))
    
    print_step("Model constants generation complete")
    print_success(f"Model constants have been generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

