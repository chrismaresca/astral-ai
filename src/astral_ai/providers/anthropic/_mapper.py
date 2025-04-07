# -------------------------------------------------------------------------------- #
# Anthropic Mappers
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import List, Optional, Union, Dict, Any
import logging

# module imports
from astral_ai._types import NOT_GIVEN

from astral_ai.messages._models import ValidatedMessageList
from astral_ai.constants._models import AnthropicModels
from astral_ai.providers.anthropic._types import AnthropicMessageType

# Model Capabilities
from astral_ai.constants._model_capabilities import get_model_max_tokens

# Types
from astral_ai._types._request._request_params import ReasoningEffort
from ._types._request import AnthropicReasoning

# Get logger
from astral_ai.logger import logger

# ------------------------------------------------------------------------------
# Anthropic Message Mapping
# ------------------------------------------------------------------------------


def to_anthropic_messages(messages: ValidatedMessageList, system_message: Optional[str] = None) -> List[AnthropicMessageType]:
    """
    Convert Astral AI messages to Anthropic chat messages format.

    This function transforms the internal Astral AI message format into the format
    expected by Anthropic's API. It handles different message types (text, image_url, 
    image_base64) and transforms them to the correct Anthropic format.

    Note: Anthropic only supports 'user' and 'assistant' roles. System messages are
    prepended to the first user message with a clear separator.

    Args:
        messages: A ValidatedMessageList containing the conversation messages
        system_message: Optional system message to prepend to the conversation

    Returns:
        A list of messages in Anthropic's expected format

    Example:
        ```python
        # Text-only example
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "Hello, how are you?", "type": "text"},
            {"role": "assistant", "content": "I'm doing well, thank you!", "type": "text"}
        ])

        anthropic_msgs = to_anthropic_messages(
            messages=messages,
            system_message="You are a helpful assistant."
        )
        # Result: [
        #   {"role": "user", "content": "You are a helpful assistant.\n\nHello, how are you?"},
        #   {"role": "assistant", "content": "I'm doing well, thank you!"}
        # ]

        # Image example
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "What's in this image?", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/image.jpg"}
        ])

        anthropic_msgs = to_anthropic_messages(messages=messages)
        # Result: [
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "What's in this image?"},
        #       {"type": "image", "source": {"type": "url", "url": "https://example.com/image.jpg"}}
        #     ]
        #   }
        # ]

        # Multi-turn conversation example
        messages = ValidatedMessageList(messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes images.", "type": "text"},
            {"role": "user", "content": "Can you help me identify what's in this picture?", "type": "text"},
            {"role": "assistant", "content": "Of course! I'd be happy to help. Please share the image.", "type": "text"},
            {"role": "user", "content": "Here it is:", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/flower.jpg"}
        ])

        anthropic_msgs = to_anthropic_messages(messages=messages)
        # Result: [
        #   {
        #     "role": "user", 
        #     "content": "You are a helpful assistant that analyzes images.\n\nCan you help me identify what's in this picture?"
        #   },
        #   {
        #     "role": "assistant", 
        #     "content": "Of course! I'd be happy to help. Please share the image."
        #   },
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "Here it is:"},
        #       {"type": "image", "source": {"type": "url", "url": "https://example.com/flower.jpg"}}
        #     ]
        #   }
        # ]

        # Multiple images example (mixed types)
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "Compare these two images:", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/image1.jpg"},
            {"role": "user", "type": "image_base64", "image_data": "base64data", "media_type": "image/jpeg"}
        ])

        anthropic_msgs = to_anthropic_messages(messages=messages)
        # Result: [
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "Compare these two images:"},
        #       {"type": "image", "source": {"type": "url", "url": "https://example.com/image1.jpg"}},
        #       {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "base64data"}}
        #     ]
        #   }
        # ]
        ```
    """
    # Get validated message list
    validated_messages = messages.to_validated_message_list()
    anthropic_messages: List[AnthropicMessageType] = []

    current_user_content: List[Dict[str, Any]] = []
    current_role = None

    # Track if we've included the system message
    system_message_added = False

    # Log warning if system message is provided
    if system_message:
        logger.warning("Anthropic does not support system messages natively. Transforming into a user message and putting at the top.")

    # Process each message
    for idx, msg in enumerate(validated_messages):
        msg_type = msg.get("type", "text")  # Default to text for backward compatibility
        msg_role = msg["role"]

        # Skip system messages - they'll be handled separately
        if msg_role == "system":
            # Log warning about system message
            logger.warning("Anthropic does not support system messages natively. "
                           "Transforming into a user message and putting at the top. "
                           "This is not the same as a system prompt. "
                           "Please use the system_message parameter to pass a system prompt.")

            # If it's a system message and we have a dedicated system_message parameter,
            # we'll prefer the parameter over the message in the list
            if system_message is None:
                system_message = msg["content"]
            continue

        # Ensure role is either "user" or "assistant"
        if msg_role not in ["user", "assistant"]:
            # Default to user for any other roles
            logger.warning(f"Role '{msg_role}' is not supported by Anthropic. Converting to 'user' role.")
            msg_role = "user"

        # If role changes or this is the last message, add the previous batch
        if current_role is not None and (current_role != msg_role or idx == len(validated_messages) - 1):
            # If this is the last message with the same role, process current message first
            if idx == len(validated_messages) - 1 and current_role == msg_role:
                if msg_type == "text":
                    current_user_content.append({"type": "text", "text": msg["content"]})
                elif msg_type == "image_url":
                    current_user_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": msg["image_url"]
                        }
                    })
                elif msg_type == "image_base64":
                    current_user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": msg["media_type"],
                            "data": msg["image_data"]
                        }
                    })

            # Add system message to first user message if necessary
            if current_role == "user" and not system_message_added and system_message:
                # If we only have text content, handle directly
                if len(current_user_content) == 1 and current_user_content[0].get("type") == "text":
                    current_user_content[0]["text"] = f"{system_message}\n\n{current_user_content[0]['text']}"
                # If we have multiple content blocks or non-text content
                elif any(item.get("type") == "text" for item in current_user_content):
                    # Find the first text block and prepend to it
                    for item in current_user_content:
                        if item.get("type") == "text":
                            item["text"] = f"{system_message}\n\n{item['text']}"
                            break
                else:
                    # If no text blocks, add one at the beginning
                    current_user_content.insert(0, {"type": "text", "text": f"{system_message}\n\n"})

                system_message_added = True

            # Add the content batch to messages
            if len(current_user_content) == 1 and current_user_content[0].get("type") == "text":
                # If only one text content, use string shorthand
                anthropic_messages.append({
                    "role": current_role,
                    "content": current_user_content[0]["text"]
                })
            else:
                # Otherwise use the content array
                anthropic_messages.append({
                    "role": current_role,
                    "content": current_user_content
                })

            # Reset for new content batch
            current_user_content = []

            # Process current message if it wasn't the last of its role
            if not (idx == len(validated_messages) - 1 and current_role == msg_role):
                if msg_type == "text":
                    current_user_content.append({"type": "text", "text": msg["content"]})
                elif msg_type == "image_url":
                    current_user_content.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": msg["image_url"]
                        }
                    })
                elif msg_type == "image_base64":
                    current_user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": msg["media_type"],
                            "data": msg["image_data"]
                        }
                    })
        else:
            # Add to current content batch
            if msg_type == "text":
                current_user_content.append({"type": "text", "text": msg["content"]})
            elif msg_type == "image_url":
                current_user_content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": msg["image_url"]
                    }
                })
            elif msg_type == "image_base64":
                current_user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": msg["media_type"],
                        "data": msg["image_data"]
                    }
                })

        current_role = msg_role

    # Add any remaining content
    if current_user_content:
        # Add system message to first user message if necessary
        if current_role == "user" and not system_message_added and system_message:
            # If we only have text content, handle directly
            if len(current_user_content) == 1 and current_user_content[0].get("type") == "text":
                current_user_content[0]["text"] = f"{system_message}\n\n{current_user_content[0]['text']}"
            # If we have multiple content blocks or non-text content
            elif any(item.get("type") == "text" for item in current_user_content):
                # Find the first text block and prepend to it
                for item in current_user_content:
                    if item.get("type") == "text":
                        item["text"] = f"{system_message}\n\n{item['text']}"
                        break
            else:
                # If no text blocks, add one at the beginning
                current_user_content.insert(0, {"type": "text", "text": f"{system_message}\n\n"})

            system_message_added = True

        # Always use the content array format, as Anthropic requires the explicit "type" field
        anthropic_messages.append({
            "role": current_role,
            "content": current_user_content
        })

    # If we have a system message but no user messages to add it to,
    if system_message and not system_message_added:
        logger.warning("No user messages found. Creating a standalone user message with system instructions.")
        anthropic_messages.insert(0, {
            "role": "user",
            "content": [{"type": "text", "text": system_message}]
        })

    return anthropic_messages


# -------------------------------------------------------------------------------- #
# Anthropic Thinking Mapper
# -------------------------------------------------------------------------------- #


def to_anthropic_thinking(reasoning_effort: ReasoningEffort, max_tokens: int, model: AnthropicModels, movement: int = 12000) -> AnthropicReasoning:
    """
    Convert Astral AI reasoning effort to Anthropic thinking.

    Args:
        reasoning_effort: The reasoning effort to convert
        max_tokens: The maximum number of tokens in the response
        model: The Anthropic model being used
        movement: The number of tokens to adjust the budget by for LOW and HIGH reasoning efforts (default: 12000)

    Returns:
        AnthropicReasoning: A dictionary with either {"type": "disabled"} or 
                           {"type": "enabled", "budget_tokens": <token_budget>}
                           
    Examples:
        ```python
        # With reasoning effort disabled
        thinking = to_anthropic_thinking(
            reasoning_effort=None,
            max_tokens=4096,
            model="claude-3-opus"
        )
        # Result: {"type": "disabled"}
        
        # With low reasoning effort
        thinking = to_anthropic_thinking(
            reasoning_effort=ReasoningEffort.LOW,
            max_tokens=40000,
            model="claude-3-5-sonnet"
        )
        # Result: {"type": "enabled", "budget_tokens": 8000}
        
        # With medium reasoning effort for Claude 3.7 Sonnet
        thinking = to_anthropic_thinking(
            reasoning_effort=ReasoningEffort.MEDIUM,
            max_tokens=128000,
            model="claude-3-7-sonnet-20250219"
        )
        # Result: {"type": "enabled", "budget_tokens": 64000}
        
        # With high reasoning effort
        thinking = to_anthropic_thinking(
            reasoning_effort=ReasoningEffort.HIGH,
            max_tokens=8192,
            model="claude-3-opus"
        )
        # Result: {"type": "enabled", "budget_tokens": 200192}
        ```
    """
    if not reasoning_effort:
        return {"type": "disabled"}

    if model == "claude-3-7-sonnet-20250219":
        logger.warning(f"Claude 3.7 Sonnet has a max output tokens of 128000. "
                       f"We will use 4,000 for the base thinking tokens. "
                       f"This is the maximum amount of tokens that can be used for thinking. "
                       f"We're adjusting the budget tokens by {movement} to account for the movement for reasoning effort.")
        base_thinking_tokens = 4000
    else:
        base_thinking_tokens = max_tokens - 20000

    if reasoning_effort == "low":
        return {"type": "enabled", "budget_tokens": base_thinking_tokens - movement}
    elif reasoning_effort == "medium":
        return {"type": "enabled", "budget_tokens": base_thinking_tokens}
    elif reasoning_effort == "high":
        return {"type": "enabled", "budget_tokens": base_thinking_tokens + movement}
    return {"type": "disabled"}


# -------------------------------------------------------------------------------- #
# Anthropic System Message Mapper
# -------------------------------------------------------------------------------- #


def to_anthropic_system_message(system_message: Optional[str] = None) -> Dict[str, Any]:
    
    if system_message is None:
        return NOT_GIVEN
    
    return system_message


# -------------------------------------------------------------------------------- #
# Anthropic Max Tokens Mapper
# -------------------------------------------------------------------------------- #


def to_anthropic_max_tokens(model: AnthropicModels, max_tokens: Optional[int] = None, with_reasoning_effort: Optional[bool] = False) -> int:
    """
    Convert Astral AI max tokens to Anthropic max tokens.

    This function determines the appropriate max_tokens value to use with Anthropic models.
    If a max_tokens value is provided, it uses that value directly. Otherwise, it attempts
    to retrieve the default maximum tokens for the specified model from model capabilities.
    If retrieval fails, it defaults to 8192 tokens.

    When with_reasoning_effort is True, the function will attempt to get the adjusted
    max_tokens value that accounts for additional tokens needed for the reasoning process.
    Note that most Anthropic models have max_output_tokens_reasoning_effort set to None,
    except for claude-3-7-sonnet-20250219 which has 64000.

    Args:
        model: The Anthropic model identifier as a Literal string (e.g., "claude-3-opus" or 
               "claude-3-opus-20240229"). Can be either an alias or specific model ID.
        max_tokens: Optional user-specified maximum number of tokens for the response.
                   If provided, this value will be used directly.
        with_reasoning_effort: Whether to include additional tokens for reasoning effort.
                              This affects models that support reasoning capabilities.

    Returns:
        int: The maximum number of tokens to use with the Anthropic API

    Example:
        ```python
        # Using default tokens for Claude-3-Opus (returns 4096)
        max_tokens = to_anthropic_max_tokens(model="claude-3-opus")

        # Using default tokens for Claude-3-Haiku (returns 4096)
        max_tokens = to_anthropic_max_tokens(model="claude-3-haiku")

        # Using default tokens for Claude-3-5-Sonnet (returns 8192)
        max_tokens = to_anthropic_max_tokens(model="claude-3-5-sonnet")

        # Using custom max tokens
        max_tokens = to_anthropic_max_tokens(
            model="claude-3-sonnet",
            max_tokens=2000
        )

        # With reasoning effort consideration for Claude-3-7-Sonnet (returns 64000)
        max_tokens = to_anthropic_max_tokens(
            model="claude-3-7-sonnet",
            with_reasoning_effort=True
        )

        # With reasoning effort for Claude-3-Opus (fallback to regular max_tokens: 4096)
        # since max_output_tokens_reasoning_effort is None
        max_tokens = to_anthropic_max_tokens(
            model="claude-3-opus",
            with_reasoning_effort=True
        )
        ```
    """
    logger.debug(f"Anthropic needs a specific max tokens value. ")

    if max_tokens is None or max_tokens == NOT_GIVEN:
        model_spec_max_tokens = get_model_max_tokens(model, with_reasoning_effort=with_reasoning_effort)
        if model_spec_max_tokens is None:
            logger.warning(f"You did not provide a max_tokens value and something went wrong when attempting to get the max tokens for model {model}.",
                           "Max tokens is required for Anthropic.",
                           "Setting to 8192 for most models 3.5 series models, or 4096 for 3-series models.")

            if model == "claude-3-7-sonnet-20250219":
                return 128000

            if model in ["claude-3-haiku", "claude-3-opus", "claude-3-haiku-20240307", "claude-3-opus-20240229"]:
                return 4096
            else:
                return 8192
        else:
            max_tokens = model_spec_max_tokens

    return max_tokens
