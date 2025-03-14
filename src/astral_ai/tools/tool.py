# -------------------------------------------------------------------------------- #
# Tool Decorators for Function Calling
# -------------------------------------------------------------------------------- #

"""
Tools for function calling with language models.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar, cast, get_type_hints
from typing import get_args, get_origin, Literal, Tuple, List, Union
import json

# Third-party imports
try:
    from griffe import Docstring, DocstringSectionKind
    GRIFFE_AVAILABLE = True
except ImportError:
    GRIFFE_AVAILABLE = False

from pydantic import BaseModel, Field, create_model

# module imports
from astral_ai._types._request._request_params import Tool, ToolDefinition

# -------------------------------------------------------------------------------- #
# Type Variables and Literals
# -------------------------------------------------------------------------------- #
F = TypeVar('F', bound=Callable[..., Any])
DocstringStyle = Literal["google", "numpy", "sphinx"]

# -------------------------------------------------------------------------------- #
# Schema Utilities
# -------------------------------------------------------------------------------- #
def ensure_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures that a JSON schema adheres to the 'strict' standard expected by OpenAI's API.
    
    Args:
        schema: The schema to ensure is strict
        
    Returns:
        The schema with strict mode enforced
    """
    # Only keep allowed top-level schema keys for strict mode
    allowed_keys = {
        "type", "properties", "required", "items", "enum", 
        "anyOf", "allOf", "oneOf", "not", "definitions", "$ref",
        "title", "description", "default", "format",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minLength", "maxLength", "pattern",
        "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties"
    }
    
    if not schema:
        return schema
    
    # Create a new schema with only allowed keys
    strict_schema = {k: v for k, v in schema.items() if k in allowed_keys}
    
    # Process properties recursively if present
    if "properties" in strict_schema and isinstance(strict_schema["properties"], dict):
        for prop_name, prop_schema in strict_schema["properties"].items():
            if isinstance(prop_schema, dict):
                strict_schema["properties"][prop_name] = ensure_strict_json_schema(prop_schema)
    
    # Process array items if present
    if "items" in strict_schema and isinstance(strict_schema["items"], dict):
        strict_schema["items"] = ensure_strict_json_schema(strict_schema["items"])
    
    return strict_schema

# -------------------------------------------------------------------------------- #
# Docstring Parsing Utilities
# -------------------------------------------------------------------------------- #
@dataclass
class FuncDocumentation:
    """Contains metadata about a python function, extracted from its docstring."""
    name: str
    """The name of the function, via `__name__`."""
    description: Optional[str] = None
    """The description of the function, derived from the docstring."""
    param_descriptions: Optional[Dict[str, str]] = None
    """The parameter descriptions of the function, derived from the docstring."""


def _detect_docstring_style(doc: str) -> DocstringStyle:
    """
    Detect the style of a docstring.
    
    Args:
        doc: The docstring to analyze
        
    Returns:
        The detected style (google, numpy, or sphinx)
    """
    scores: Dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters', 'Returns', or 'Yields' followed by
    # a dashed underline
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"  # Default to Google style

    # Priority order: sphinx > numpy > google in case of tie
    styles: List[DocstringStyle] = ["sphinx", "numpy", "google"]
    for style in styles:
        if scores[style] == max_score:
            return style

    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    """Suppress warnings from griffe about missing annotations for params."""
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def generate_func_documentation(
    func: Callable[..., Any], style: Optional[DocstringStyle] = None
) -> FuncDocumentation:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, we will attempt to
            auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=name)

    if GRIFFE_AVAILABLE:
        with _suppress_griffe_logging():
            docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
            parsed = docstring.parse()

        description: Optional[str] = next(
            (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
        )

        param_descriptions: Dict[str, str] = {
            param.name: param.description
            for section in parsed
            if section.kind == DocstringSectionKind.parameters
            for param in section.value
        }

        return FuncDocumentation(
            name=name,
            description=description,
            param_descriptions=param_descriptions or None,
        )
    else:
        # Fallback to regex parsing if griffe is not available
        style = style or _detect_docstring_style(doc)
        
        # Simple extraction for description (everything before Args/Parameters section)
        description = None
        param_descriptions = {}
        
        if style == "google":
            # Google-style docstring parsing
            sections = re.split(r"^(Args|Returns|Raises):\s*$", doc, flags=re.MULTILINE)
            if sections:
                description = sections[0].strip()
                
            # Find Args section
            for i in range(1, len(sections) - 1, 2):
                if sections[i] == "Args":
                    args_section = sections[i + 1].strip()
                    # Parse parameter definitions
                    param_matches = re.finditer(
                        r"^\s*([a-zA-Z0-9_]+)(\s*\([^)]+\))?\s*:\s*(.+?)(?=^\s*[a-zA-Z0-9_]+\s*:|$)", 
                        args_section, re.MULTILINE | re.DOTALL
                    )
                    for match in param_matches:
                        name = match.group(1).strip()
                        desc = match.group(3).strip()
                        param_descriptions[name] = desc
        
        elif style == "sphinx":
            # Extract description (text before any :param or other directives)
            desc_match = re.match(r"(.*?)(?=\n\s*:|\Z)", doc, re.DOTALL)
            if desc_match:
                description = desc_match.group(1).strip()
            
            # Extract parameters
            param_matches = re.finditer(r":param\s+([a-zA-Z0-9_]+):\s*(.+?)(?=\n\s*:|$)", doc, re.DOTALL)
            for match in param_matches:
                name = match.group(1).strip()
                desc = match.group(2).strip()
                param_descriptions[name] = desc
                
        elif style == "numpy":
            sections = re.split(r"^(\w+)\s*\n\s*-+\s*$", doc, flags=re.MULTILINE)
            if sections:
                description = sections[0].strip()
            
            # Process sections
            for i in range(1, len(sections) - 1, 2):
                section_name = sections[i].strip()
                section_content = sections[i + 1].strip()
                
                if section_name == "Parameters":
                    # Simplified parsing for parameter blocks
                    param_blocks = re.split(r"^([a-zA-Z0-9_]+)(?:\s*:\s*[^\n]+)?\s*$", 
                                          section_content, flags=re.MULTILINE)
                    for j in range(1, len(param_blocks) - 1, 2):
                        name = param_blocks[j].strip()
                        desc = param_blocks[j + 1].strip()
                        param_descriptions[name] = desc
        
        return FuncDocumentation(
            name=name,
            description=description,
            param_descriptions=param_descriptions or None,
        )


# -------------------------------------------------------------------------------- #
# Schema Generation
# -------------------------------------------------------------------------------- #
@dataclass
class FunctionSchema:
    """
    Schema representation of a function for tool generation.
    """
    name: str
    """The name of the function."""
    description: Optional[str]
    """The description of the function."""
    pydantic_model: type[BaseModel]
    """The Pydantic model for the function parameters."""
    params_json_schema: Dict[str, Any]
    """The JSON schema for the function parameters."""
    signature: inspect.Signature
    """The signature of the function."""
    
    def to_call_args(self, data: BaseModel) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs),
        suitable for calling the original function.
        
        Args:
            data: The validated data from the Pydantic model
            
        Returns:
            A tuple of (positional_args, keyword_args)
        """
        positional_args: List[Any] = []
        keyword_args: Dict[str, Any] = {}
        seen_var_positional = False
        
        for name, param in self.signature.parameters.items():
            value = getattr(data, name, None)
            
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
                
        return positional_args, keyword_args


def generate_function_schema(
    func: Callable[..., Any],
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    docstring_style: Optional[DocstringStyle] = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FunctionSchema:
    """
    Generate a schema representation for a function.
    
    Args:
        func: The function to analyze
        name_override: Optional override for the function name
        description_override: Optional override for the function description
        docstring_style: The style of the docstring to use for parsing
        use_docstring_info: Whether to use docstring information
        strict_json_schema: Whether to ensure the schema adheres to strict standards
        
    Returns:
        A FunctionSchema object containing the schema representation
    """
    # Parse docstring for description and param descriptions
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = doc_info.param_descriptions or {}
    else:
        doc_info = FuncDocumentation(name=func.__name__)
        param_descs = {}
    
    # Get function name (use override if provided)
    func_name = name_override or doc_info.name
    
    # Get function signature and type hints
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    # We will collect field definitions for create_model as a dict:
    # field_name -> (type_annotation, default_value_or_Field(...))
    fields: Dict[str, Any] = {}
    
    for name, param in sig.parameters.items():
        ann = type_hints.get(name, param.annotation)
        default = param.default
        
        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any
            
        # Get parameter description from docstring if available
        field_description = param_descs.get(name)
        
        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = List[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = List[ann]  # type: ignore
                
            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),
            )
            
        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = Dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = Dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = Dict[str, ann]  # type: ignore
                
            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),
            )
            
        else:
            # Normal parameter
            if default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )
    
    # Create Pydantic model
    model = create_model(f"{func_name}_params", **fields)
    
    # Get JSON schema from model
    json_schema = model.model_json_schema()
    
    # Ensure strict schema if requested
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)
    
    return FunctionSchema(
        name=func_name,
        description=description_override or doc_info.description,
        pydantic_model=model,
        params_json_schema=json_schema,
        signature=sig
    )


# -------------------------------------------------------------------------------- #
# Tool Decorator
# -------------------------------------------------------------------------------- #
def function_tool(
    func: Optional[F] = None, 
    *, 
    name: Optional[str] = None, 
    description: Optional[str] = None,
    docstring_style: Optional[DocstringStyle] = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to convert a Python function into a tool for use with language models.
    
    Args:
        func: The function to decorate
        name: Optional override for the function name
        description: Optional override for the function description
        docstring_style: The docstring style to use for parsing
        use_docstring_info: Whether to use docstring information for schema generation
        strict_json_schema: Whether to ensure the schema adheres to strict standards
        
    Returns:
        The decorated function
        
    Example:
        @function_tool
        def get_weather(location: str, unit: str = "C") -> str:
            '''Get weather for a location.'''
            return f"The weather in {location} is sunny."
            
        @function_tool(name="fetch_weather", description="Get weather information")
        def get_weather(location: str, unit: str = "C") -> str:
            return f"The weather in {location} is sunny."
    """
    def decorator(fn: F) -> F:
        # Generate function schema
        schema = generate_function_schema(
            fn, 
            name_override=name,
            description_override=description,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_json_schema
        )
        
        # Create tool definition
        tool_definition = ToolDefinition(
            name=schema.name,
            description=schema.description or "",
            parameters=schema.params_json_schema
        )
        
        # Create tool
        tool_obj = Tool(
            type="function",
            function=tool_definition
        )
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        
        # Attach tool and schema to the function
        wrapper.tool = tool_obj
        wrapper.schema = schema
        
        # Add convenience method for validating and converting input
        def validate_and_call(input_data: Dict[str, Any]) -> Any:
            """Validate input data against the schema and call the function."""
            model_instance = schema.pydantic_model(**input_data)
            args, kwargs = schema.to_call_args(model_instance)
            return fn(*args, **kwargs)
        
        wrapper.validate_and_call = validate_and_call
        
        return cast(F, wrapper)
    
    # Handle both @function_tool and @function_tool() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


# -------------------------------------------------------------------------------- #
# Example Usage
# -------------------------------------------------------------------------------- #
if __name__ == "__main__":
    @function_tool
    def get_weather(location: str, unit: str = "C") -> str:
        """
        Fetch the weather for a given location, returning a short description.
        
        Args:
            location: The city or location to get weather for
            unit: Temperature unit, either C or F
            
        Returns:
            A description of the weather
        """
        # Example logic
        return f"The weather in {location} is 22 degrees {unit}."
    
    @function_tool(name="calculate_geometric_area")
    def calculate_area(shape: str, dimensions: Dict[str, float]) -> float:
        """
        Calculate the area of a geometric shape.
        
        Args:
            shape: The type of shape (circle, rectangle, triangle)
            dimensions: Dimensions required for the calculation
            
        Returns:
            The calculated area
        """
        if shape == "circle" and "radius" in dimensions:
            return 3.14159 * dimensions["radius"] ** 2
        elif shape == "rectangle" and "width" in dimensions and "height" in dimensions:
            return dimensions["width"] * dimensions["height"]
        else:
            return 0.0
    
    @function_tool(
        name="product_search", 
        description="Search for products in the catalog with optional filtering"
    )
    def search_products(query: str, category: Optional[str] = None, max_results: int = 10) -> list:
        """
        Search for products in a catalog.
        
        Args:
            query: Search query string
            category: Optional category to filter results
            max_results: Maximum number of results to return
            
        Returns:
            List of matching products
        """
        # Example implementation
        return [{"name": f"Product {i}", "category": category or "general"} for i in range(min(3, max_results))]
    
    # Variable argument example
    @function_tool
    def sum_values(*numbers: int, scale: float = 1.0) -> float:
        """
        Sum a variable number of values with optional scaling.
        
        Args:
            *numbers: Numbers to sum
            scale: Value to multiply the sum by
            
        Returns:
            The scaled sum
        """
        return sum(numbers) * scale
    
    # Keyword argument example
    @function_tool 
    def format_user_data(name: str, **attributes: Any) -> Dict[str, Any]:
        """
        Format user data with optional attributes.
        
        Args:
            name: The user's name
            **attributes: Additional user attributes
            
        Returns:
            Formatted user data object
        """
        return {"name": name, "attributes": attributes}
    
    print("\n" + "="*50)
    print("TOOL DEFINITIONS")
    print("="*50)
    
    # Print the tool definitions
    print("\nWeather Tool:")
    weather_tool = get_weather.tool
    print(f"Type: {weather_tool['type']}")
    print(f"Name: {weather_tool['function']['name']}")
    print(f"Description: {weather_tool['function']['description']}")
    print(f"Parameters: {weather_tool['function']['parameters']}")
    
    print("\nCalculate Area Tool:")
    area_tool = calculate_area.tool
    print(f"Type: {area_tool['type']}")
    print(f"Name: {area_tool['function']['name']}")
    print(f"Description: {area_tool['function']['description']}")
    print(f"Parameters: {area_tool['function']['parameters']}")
    
    print("\nProduct Search Tool:")
    search_tool = search_products.tool
    print(f"Type: {search_tool['type']}")
    print(f"Name: {search_tool['function']['name']}")
    print(f"Description: {search_tool['function']['description']}")
    print(f"Parameters: {search_tool['function']['parameters']}")
    
    print("\nSum Values Tool (variadic args):")
    sum_tool = sum_values.tool
    print(f"Type: {sum_tool['type']}")
    print(f"Name: {sum_tool['function']['name']}")
    print(f"Description: {sum_tool['function']['description']}")
    print(f"Parameters: {sum_tool['function']['parameters']}")
    
    print("\nFormat User Data Tool (keyword args):")
    format_tool = format_user_data.tool
    print(f"Type: {format_tool['type']}")
    print(f"Name: {format_tool['function']['name']}")
    print(f"Description: {format_tool['function']['description']}")
    print(f"Parameters: {format_tool['function']['parameters']}")
    
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    
    # Example usage with validation
    print("\nCalling get_weather:")
    result = get_weather.validate_and_call({"location": "Seattle", "unit": "F"})
    print(f"Result: {result}")
    
    print("\nCalling calculate_area:")
    result = calculate_area.validate_and_call({"shape": "circle", "dimensions": {"radius": 5.0}})
    print(f"Result: {result}")
    
    print("\nCalling sum_values with variadic args:")
    result = sum_values.validate_and_call({"numbers": [1, 2, 3, 4], "scale": 2.0})
    print(f"Result: {result}")
    
    print("\nCalling format_user_data with keyword args:")
    result = format_user_data.validate_and_call({
        "name": "Alice", 
        "attributes": {"age": 30, "email": "alice@example.com"}
    })
    print(f"Result: {result}")
