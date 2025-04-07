# -------------------------------------------------------------------------------- #
# Generate Model Capabilities
# -------------------------------------------------------------------------------- #
# This script reads model configuration YAML files and generates type-safe capability
# definitions for use in the codebase. It creates or updates a file in the constants
# directory with comprehensive capability mappings and utility functions.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import sys
import yaml
from typing import Dict, List, Set, Any, Tuple, Union
from pathlib import Path

# -------------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------------- #
# Output file path
OUTPUT_FILE = "src/astral_ai/constants/_model_capabilities.py"

# Input files
MODELS_FILE = "src/astral_ai/config/models.yaml"
FEATURES_FILE = "src/astral_ai/config/features.yaml"
MODELS_CONST_FILE = "src/astral_ai/constants/_models.py"  # To reference CompletionModels

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
    print_info(f"Writing generated capabilities to {OUTPUT_FILE}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    try:
        with open(OUTPUT_FILE, 'w') as file:
            file.write(content)
        print_success(f"Successfully wrote capabilities to {OUTPUT_FILE}")
    except Exception as e:
        print(f"ERROR: Failed to write to {OUTPUT_FILE}: {e}")
        sys.exit(1)

# -------------------------------------------------------------------------------- #
# Data Extraction Functions
# -------------------------------------------------------------------------------- #
def extract_features(features_data: dict) -> List[str]:
    """Extract all feature names from the features data."""
    return list(features_data.get('features', {}).keys())

def extract_chat_models_from_constants(models_file: str) -> List[str]:
    """Extract chat models from the generated constants file."""
    print_info(f"Attempting to extract CompletionModels from {models_file}")
    
    try:
        chat_models = []
        parsing_chat_models = False
        
        with open(models_file, 'r') as file:
            for line in file:
                if "CompletionModels = Literal[" in line:
                    parsing_chat_models = True
                    continue
                
                if parsing_chat_models:
                    if "]" in line and not '"' in line:  # End of Literal
                        break
                    
                    if '"' in line:  # Extract model name
                        # Extract the text between quotes
                        model_name = line.split('"')[1]
                        chat_models.append(model_name)
        
        print_info(f"Extracted {len(chat_models)} chat models")
        return chat_models
    except Exception as e:
        print(f"ERROR: Failed to extract CompletionModels: {e}")
        sys.exit(1)

def build_alias_to_model_mapping(models_data: dict) -> Dict[str, str]:
    """Build a mapping from model aliases to their default model IDs."""
    mapping = {}
    
    # Process models in models.yaml
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model_info in provider_data.get('models', {}).items():
            alias = model_info.get('alias')
            is_default = model_info.get('default', False)
            
            if alias and is_default:
                mapping[alias] = model_id
    
    print_info(f"Built mapping for {len(mapping)} model aliases to their default IDs")
    return mapping

def build_model_capabilities(models_data: dict, feature_names: List[str], chat_models: List[str]) -> Dict[str, Dict[str, bool]]:
    """Build a mapping of model capabilities for all chat models."""
    capabilities = {}
    
    # Process models in models.yaml
    for provider, provider_data in models_data.get('providers', {}).items():
        for model_id, model_info in provider_data.get('models', {}).items():
            # Only process chat models
            if model_id not in chat_models:
                continue
                
            # Initialize capability dict for this model
            model_capabilities = {}
            
            # Extract supported features
            supported_features = model_info.get('supported_features', [])
            
            # Convert supported_features from list of dicts to a single dict
            features_dict = {}
            for feature_entry in supported_features:
                for feature_name, value in feature_entry.items():
                    features_dict[feature_name] = value
            
            # Create capability entries
            for feature in feature_names:
                # Check if feature is supported
                is_supported = features_dict.get(feature, False)
                # Create capability key (prepend "supports_")
                capability_key = f"supports_{feature}"
                model_capabilities[capability_key] = is_supported
            
            # Add to capabilities dict
            capabilities[model_id] = model_capabilities
    
    print_info(f"Built capabilities mapping for {len(capabilities)} models")
    return capabilities

# -------------------------------------------------------------------------------- #
# Main Script
# -------------------------------------------------------------------------------- #
def main():
    """Main function to generate model capabilities."""
    print_step("Starting model capabilities generation")
    
    # Load configuration files
    print_step("Loading configuration files")
    models_data = load_yaml_file(MODELS_FILE)
    features_data = load_yaml_file(FEATURES_FILE)
    
    # Extract data
    print_step("Extracting data from configuration files")
    feature_names = extract_features(features_data)
    print_info(f"Found {len(feature_names)} features: {', '.join(feature_names)}")
    
    # Extract chat models from constants file
    print_step("Extracting chat models from constants file")
    chat_models = extract_chat_models_from_constants(MODELS_CONST_FILE)
    
    # Build alias to model ID mapping
    print_step("Building alias to model ID mapping")
    alias_to_model_id = build_alias_to_model_mapping(models_data)
    
    # Build model capabilities mapping
    print_step("Building model capabilities mapping")
    model_capabilities = build_model_capabilities(models_data, feature_names, chat_models)
    
    # Generate file content
    print_step("Generating file content")
    
    content = [
        "# -------------------------------------------------------------------------------- #",
        "# Model Capabilities",
        "# -------------------------------------------------------------------------------- #",
        "# This file is auto-generated by scripts/generate_model_capabilities.py",
        "# DO NOT EDIT MANUALLY",
        "# -------------------------------------------------------------------------------- #",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Imports",
        "# -------------------------------------------------------------------------------- #",
        "# Built-in imports",
        "from typing import Literal, Dict, TypedDict, Union, Optional",
        "",
        "# Module imports",
        "from astral_ai.constants._models import ModelAlias, ModelId, CompletionModels",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Feature Names",
        "# -------------------------------------------------------------------------------- #",
        "",
        f"FEATURE_NAME = {create_literal_string(feature_names)}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Capabilities TypedDict",
        "# -------------------------------------------------------------------------------- #",
        "",
        "class ModelCapabilities(TypedDict, total=False):",
        "    \"\"\"TypedDict for capturing model capabilities.\"\"\"",
    ]
    
    # Add capability fields to ModelCapabilities
    for feature in sorted(feature_names):
        capability_key = f"supports_{feature}"
        content.append(f"    {capability_key}: bool")
    
    content.extend([
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Alias to Model ID Mapping",
        "# -------------------------------------------------------------------------------- #",
        "",
        "ALIAS_TO_MODEL_ID: Dict[ModelAlias, ModelId] = {",
    ])
    
    # Add alias to model ID mappings
    for alias, model_id in sorted(alias_to_model_id.items()):
        content.append(f"    \"{alias}\": \"{model_id}\",")
    
    content.extend([
        "}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Model Capabilities Mapping",
        "# -------------------------------------------------------------------------------- #",
        "",
        "MODEL_CAPABILITIES: Dict[ModelId, ModelCapabilities] = {",
    ])
    
    # Add model capabilities mappings
    for model_id, capabilities in sorted(model_capabilities.items()):
        content.append(f"    \"{model_id}\": " + "{")
        for capability, value in sorted(capabilities.items()):
            content.append(f"        \"{capability}\": {str(value)},")
        content.append("    },")
    
    content.extend([
        "}",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Helper Functions",
        "# -------------------------------------------------------------------------------- #",
        "",
        "",
        "def get_specific_model_id(model: Union[ModelAlias, ModelId]) -> ModelId:",
        "    \"\"\"",
        "    Convert a model alias to its specific model ID.",
        "    If a specific model ID is provided, return it directly.",
        "",
        "    Args:",
        "        model: The model alias or ID to convert",
        "",
        "    Returns:",
        "        ModelId: The specific model ID",
        "    \"\"\"",
        "    if model in ALIAS_TO_MODEL_ID:",
        "        return ALIAS_TO_MODEL_ID[model]",
        "    return model  # It's already a specific model ID",
        "",
        "",
        "def supports_feature(model: Union[ModelAlias, ModelId], feature: FEATURE_NAME) -> bool:",
        "    \"\"\"",
        "    Check if a model supports a specific feature with O(1) lookup.",
        "    Works with both aliases and specific model IDs.",
        "",
        "    Args:",
        "        model: The model alias or ID to check",
        "        feature: The feature to check for (e.g., 'reasoning_effort')",
        "",
        "    Returns:",
        "        bool: True if the model supports the feature, False otherwise",
        "    \"\"\"",
        "    model_id = get_specific_model_id(model)",
        "    if model_id not in MODEL_CAPABILITIES:",
        "        return False",
        "    ",
        "    # Prepend 'supports_' to the feature name for lookup",
        "    capability_key = f\"supports_{feature}\"",
        "    return MODEL_CAPABILITIES[model_id].get(capability_key, False)",
    ])
    
    # Write the content to the output file
    write_to_file("\n".join(content))
    
    print_step("Model capabilities generation complete")
    print_success(f"Model capabilities have been generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
