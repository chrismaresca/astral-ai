# -------------------------------------------------------------------------------- #
# Generate Cost Constants
# -------------------------------------------------------------------------------- #
# This script reads model configuration YAML files and generates type-safe cost
# constants for use in the codebase. It creates or updates a file in the constants
# directory with comprehensive cost mappings and type definitions.
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import sys
import yaml
from typing import Dict, List, Set, Any, Tuple, Union, Optional
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------------------- #
# Output file path
OUTPUT_FILE = "src/astral_ai/constants/_costs.py"

# Input files
MODELS_FILE = "src/astral_ai/config/models.yaml"
MODELS_CONST_FILE = "src/astral_ai/constants/_models.py"  # To reference models

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

def write_to_file(content: str) -> None:
    """Write the generated content to the output file."""
    print_info(f"Writing generated costs to {OUTPUT_FILE}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    try:
        with open(OUTPUT_FILE, 'w') as file:
            file.write(content)
        print_success(f"Successfully wrote costs to {OUTPUT_FILE}")
    except Exception as e:
        print(f"ERROR: Failed to write to {OUTPUT_FILE}: {e}")
        sys.exit(1)

def format_time_period(time_period: str) -> Optional[str]:
    """
    Format a time period string to be parseable by datetime, or return None if not applicable.
    
    For time windows like "UTC 00:30–16:30", we'll transform them to ISO format or similar
    that can be parsed by datetime. For "All day" or empty strings, we return None.
    """
    if not time_period or time_period == "All day":
        print(f"INFO: Formatting time period: {time_period} -> None")
        return None
    
    # For now, just return the string as is
    # In a real implementation, you might want to transform this into a datetime format
    # based on your specific needs
    return time_period

# -------------------------------------------------------------------------------- #
# Data Extraction Functions
# -------------------------------------------------------------------------------- #
def extract_providers_and_models(models_data: dict) -> Dict[str, List[str]]:
    """Extract all providers and their models from the models data."""
    providers_models = {}
    
    for provider, provider_data in models_data.get('providers', {}).items():
        models = list(provider_data.get('models', {}).keys())
        providers_models[provider] = models
    
    return providers_models

def extract_model_costs(models_data: dict) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Extract cost information for all models."""
    model_costs = {}
    
    # Process models in models.yaml
    for provider, provider_data in models_data.get('providers', {}).items():
        provider_costs = {}
        
        for model_id, model_info in provider_data.get('models', {}).items():
            # Extract pricing information
            pricing = model_info.get('pricing', {})
            
            # Check if time-based
            is_time_based = pricing.get('time_based', False)
            
            if is_time_based:
                # Handle time-based pricing (e.g., DeepSeek)
                standard_window = pricing.get('text', {}).get('standard_window', "")
                discount_window = pricing.get('text', {}).get('discount_window', "")
                
                standard_pricing = pricing.get('text', {}).get('standard', {})
                discount_pricing = pricing.get('text', {}).get('discount', {})
                
                provider_costs[model_id] = {
                    'time_based': True,
                    'standard': {
                        'time_period': format_time_period(standard_window),
                        'input_base_cost': standard_pricing.get('input', 0.0),
                        'input_cache_hit_cost': standard_pricing.get('input_cache_hit', 0.0),
                        'input_cache_write_cost': standard_pricing.get('input_cache_write', None),
                        'output_base_cost': standard_pricing.get('output', 0.0),
                    },
                    'discount': {
                        'time_period': format_time_period(discount_window),
                        'input_base_cost': discount_pricing.get('input', 0.0),
                        'input_cache_hit_cost': discount_pricing.get('input_cache_hit', 0.0),
                        'input_cache_write_cost': discount_pricing.get('input_cache_write', None),
                        'output_base_cost': discount_pricing.get('output', 0.0),
                    }
                }
            else:
                # Handle regular pricing
                text_pricing = pricing.get('text', {})
                provider_costs[model_id] = {
                    'time_based': False,
                    'standard': {
                        'time_period': None,  # Not time-based, so no time period
                        'input_base_cost': text_pricing.get('input', 0.0),
                        'input_cache_hit_cost': text_pricing.get('input_cache_hit', 0.0),
                        'input_cache_write_cost': text_pricing.get('input_cache_write', None),
                        'output_base_cost': text_pricing.get('output', 0.0),
                    }
                }
        
        model_costs[provider] = provider_costs
    
    return model_costs

# -------------------------------------------------------------------------------- #
# Main Script
# -------------------------------------------------------------------------------- #
def main():
    """Main function to generate cost constants."""
    print_step("Starting cost constants generation")
    
    # Load configuration files
    print_step("Loading configuration files")
    models_data = load_yaml_file(MODELS_FILE)
    
    # Extract data
    print_step("Extracting cost data from configuration files")
    providers_models = extract_providers_and_models(models_data)
    print_info(f"Found {len(providers_models)} providers with models")
    
    # Extract model costs
    print_step("Extracting model costs")
    model_costs = extract_model_costs(models_data)
    
    # Generate file content
    print_step("Generating file content")
    
    content = [
        "# -------------------------------------------------------------------------------- #",
        "# Model Costs",
        "# -------------------------------------------------------------------------------- #",
        "# This file is auto-generated by scripts/generate_cost_constants.py",
        "# DO NOT EDIT MANUALLY",
        "# -------------------------------------------------------------------------------- #",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Imports",
        "# -------------------------------------------------------------------------------- #",
        "# Built-in imports",
        "from typing import Dict, List, TypedDict, Optional, Union, Any",
        "from datetime import datetime, time",
        "",
        "# Module imports",
        "from astral_ai.constants._models import ModelProvider, ModelId, ModelName",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Cost Type Definitions",
        "# -------------------------------------------------------------------------------- #",
        "",
        "class ModelSpecificCosts(TypedDict, total=False):",
        "    \"\"\"TypedDict for capturing model-specific cost details.\"\"\"",
        "    time_period: Optional[str]  # A datetime-parseable string or None if not applicable",
        "    input_base_cost: float",
        "    input_cache_hit_cost: float",
        "    input_cache_write_cost: Optional[float]",
        "    output_base_cost: float",
        "",
        "",
        "class TimePeriodCosts(TypedDict):",
        "    \"\"\"TypedDict for capturing different cost periods for time-based pricing.\"\"\"",
        "    time_based: bool",
        "    standard: ModelSpecificCosts",
        "    discount: Optional[ModelSpecificCosts]",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Helper Functions for Time-Based Costs",
        "# -------------------------------------------------------------------------------- #",
        "",
        "def parse_time_window(time_window: Optional[str]) -> Optional[tuple[time, time]]:",
        "    \"\"\"",
        "    Parse a time window string (e.g., 'UTC 00:30–16:30') into a tuple of start and end times.",
        "    Returns None if the input is None or not parseable.",
        "    \"\"\"",
        "    if time_window is None:",
        "        return None",
        "    ",
        "    try:",
        "        # Example format: 'UTC 00:30–16:30'",
        "        # Strip UTC prefix if present",
        "        if time_window.startswith('UTC '):",
        "            time_window = time_window[4:]",
        "        ",
        "        # Split by the dash/hyphen",
        "        parts = time_window.split('–')",
        "        if len(parts) != 2:",
        "            return None",
        "        ",
        "        # Parse start and end times",
        "        start_str, end_str = parts[0].strip(), parts[1].strip()",
        "        start_time = datetime.strptime(start_str, '%H:%M').time()",
        "        end_time = datetime.strptime(end_str, '%H:%M').time()",
        "        ",
        "        return (start_time, end_time)",
        "    except (ValueError, IndexError):",
        "        return None",
        "",
        "",
        "def is_current_time_in_window(time_window: Optional[str]) -> bool:",
        "    \"\"\"",
        "    Check if the current time is within the specified time window.",
        "    If time_window is None, always returns True.",
        "    \"\"\"",
        "    if time_window is None:",
        "        return True",
        "    ",
        "    window = parse_time_window(time_window)",
        "    if window is None:",
        "        return True",
        "    ",
        "    start_time, end_time = window",
        "    current_time = datetime.now().time()",
        "    ",
        "    # Handle cases where the window spans midnight",
        "    if start_time <= end_time:",
        "        return start_time <= current_time <= end_time",
        "    else:",
        "        return current_time >= start_time or current_time <= end_time",
        "",
        "",
        "# -------------------------------------------------------------------------------- #",
        "# Provider Cost Mappings",
        "# -------------------------------------------------------------------------------- #",
        "",
    ]
    
    # Create individual provider mappings
    for provider, models in providers_models.items():
        var_name = f"{provider.upper()}_COSTS"
        content.extend([
            f"{var_name}: Dict[ModelId, TimePeriodCosts] = {{",
        ])
        
        # Add models and their costs
        provider_model_costs = model_costs.get(provider, {})
        for model_id in sorted(models):
            cost_data = provider_model_costs.get(model_id, {})
            if cost_data:
                content.append(f"    \"{model_id}\": {{")
                content.append(f"        \"time_based\": {str(cost_data.get('time_based', False))},")
                
                # Add standard pricing
                standard = cost_data.get('standard', {})
                content.append(f"        \"standard\": {{")
                
                # Handle time_period which might be None
                time_period = standard.get('time_period')
                if time_period is None:
                    content.append(f"            \"time_period\": None,")
                else:
                    content.append(f"            \"time_period\": \"{time_period}\",")
                    
                content.append(f"            \"input_base_cost\": {standard.get('input_base_cost', 0.0)},")
                content.append(f"            \"input_cache_hit_cost\": {standard.get('input_cache_hit_cost', 0.0)},")
                
                # Handle possible None value for input_cache_write_cost
                cache_write = standard.get('input_cache_write_cost')
                if cache_write is None:
                    content.append(f"            \"input_cache_write_cost\": None,")
                else:
                    content.append(f"            \"input_cache_write_cost\": {cache_write},")
                
                content.append(f"            \"output_base_cost\": {standard.get('output_base_cost', 0.0)},")
                content.append(f"        }},")
                
                # Add discount pricing if time-based
                if cost_data.get('time_based', False) and 'discount' in cost_data:
                    discount = cost_data.get('discount', {})
                    content.append(f"        \"discount\": {{")
                    
                    # Handle time_period which might be None
                    discount_time_period = discount.get('time_period')
                    if discount_time_period is None:
                        content.append(f"            \"time_period\": None,")
                    else:
                        content.append(f"            \"time_period\": \"{discount_time_period}\",")
                        
                    content.append(f"            \"input_base_cost\": {discount.get('input_base_cost', 0.0)},")
                    content.append(f"            \"input_cache_hit_cost\": {discount.get('input_cache_hit_cost', 0.0)},")
                    
                    # Handle possible None value for input_cache_write_cost
                    discount_cache_write = discount.get('input_cache_write_cost')
                    if discount_cache_write is None:
                        content.append(f"            \"input_cache_write_cost\": None,")
                    else:
                        content.append(f"            \"input_cache_write_cost\": {discount_cache_write},")
                    
                    content.append(f"            \"output_base_cost\": {discount.get('output_base_cost', 0.0)},")
                    content.append(f"        }},")
                else:
                    content.append(f"        \"discount\": None,")
                
                content.append(f"    }},")
        
        content.extend([
            "}",
            "",
        ])
    
    # Create the unified model_specific_cost_mapping
    content.extend([
        "# -------------------------------------------------------------------------------- #",
        "# Unified Cost Mapping",
        "# -------------------------------------------------------------------------------- #",
        "",
        "MODEL_COST_MAPPING: Dict[ModelProvider, Dict[ModelId, TimePeriodCosts]] = {",
    ])
    
    for provider in providers_models.keys():
        var_name = f"{provider.upper()}_COSTS"
        content.append(f"    \"{provider}\": {var_name},")
    
    content.append("}")
    
    # Write the content to the output file
    write_to_file("\n".join(content))
    
    print_step("Cost constants generation complete")
    print_success(f"Cost constants have been generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 