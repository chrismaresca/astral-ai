# -------------------------------------------------------------------------------- #
# OpenAI Message Mapper
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
from typing import List, Optional, Union, Dict, Any

# module imports
from astral_ai.constants._models import OpenAIModels
from astral_ai.constants._model_capabilities import supports_feature
from astral_ai.messages._models import ValidatedMessageList, ValidatedMessageDict
from astral_ai.providers.openai._types._message import OpenAIMessageType

# -------------------------------------------------------------------------------- #
# Message Mapping Functions
# -------------------------------------------------------------------------------- #


def to_openai_messages(model_name: OpenAIModels, messages: ValidatedMessageList, system_message: Optional[str]) -> Union[List[OpenAIMessageType], OpenAIMessageType]:
    """
    Convert Astral AI messages to OpenAI chat messages format.

    This function transforms the internal Astral AI message format into the format
    expected by OpenAI's API. It handles system messages differently based on model
    capabilities and transforms different message types (text, image_url, image_base64)
    to the correct OpenAI format.

    Args:
        model_name: The OpenAI model being used (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: A ValidatedMessageList containing the conversation messages
        system_message: Optional system message to prepend to the conversation

    Returns:
        A list of messages in OpenAI's expected format

    Example:
        ```python
        # Text-only example
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "Hello, how are you?", "type": "text"},
            {"role": "assistant", "content": "I'm doing well, thank you!", "type": "text"}
        ])

        openai_msgs = to_openai_messages(
            model_name="gpt-4o",
            messages=messages,
            system_message="You are a helpful assistant."
        )
        # Result: [
        #   {"role": "system", "content": "You are a helpful assistant."},
        #   {"role": "user", "content": "Hello, how are you?"},
        #   {"role": "assistant", "content": "I'm doing well, thank you!"}
        # ]
        
        # Image example
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "What's in this image?", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/image.jpg", "image_detail": "high"}
        ])

        openai_msgs = to_openai_messages(
            model_name="gpt-4o",
            messages=messages,
            system_message=None
        )
        # Result: [
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "What's in this image?"},
        #       {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg", "detail": "high"}}
        #     ]
        #   }
        # ]
        
        # Multi-turn conversation example
        messages = ValidatedMessageList(messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes images.", "type": "text"},
            {"role": "user", "content": "Can you help me identify what's in this picture?", "type": "text"},
            {"role": "assistant", "content": "Of course! I'd be happy to help. Please share the image.", "type": "text"},
            {"role": "user", "content": "Here it is:", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/flower.jpg", "image_detail": "high"}
        ])

        openai_msgs = to_openai_messages(
            model_name="gpt-4o",
            messages=messages,
            system_message=None
        )
        # Result: [
        #   {"role": "system", "content": "You are a helpful assistant that analyzes images."},
        #   {"role": "user", "content": "Can you help me identify what's in this picture?"},
        #   {"role": "assistant", "content": "Of course! I'd be happy to help. Please share the image."},
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "Here it is:"},
        #       {"type": "image_url", "image_url": {"url": "https://example.com/flower.jpg", "detail": "high"}}
        #     ]
        #   }
        # ]
        
        # Multiple images example
        messages = ValidatedMessageList(messages=[
            {"role": "user", "content": "Compare these two images:", "type": "text"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/image1.jpg", "image_detail": "high"},
            {"role": "user", "type": "image_url", "image_url": "https://example.com/image2.jpg", "image_detail": "high"}
        ])

        openai_msgs = to_openai_messages(
            model_name="gpt-4o",
            messages=messages,
            system_message=None
        )
        # Result: [
        #   {
        #     "role": "user", 
        #     "content": [
        #       {"type": "text", "text": "Compare these two images:"},
        #       {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg", "detail": "high"}},
        #       {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg", "detail": "high"}}
        #     ]
        #   }
        # ]
        ```
    """
    # Get validated message list
    validated_messages = messages.to_validated_message_list()
    openai_messages: List[Dict[str, Any]] = []
    
    # Keep track of the last message for potential merging
    last_msg_role = None
    last_msg_content = None
    i = 0
    
    # Process each message based on its type
    while i < len(validated_messages):
        msg = validated_messages[i]
        msg_type = msg.get("type", "text")  # Default to text for backward compatibility
        msg_role = msg["role"]
        
        # Handle text message
        if msg_type == "text":
            text_content = msg["content"]
            
            # Check for consecutive user messages that include images
            if msg_role == "user" and i + 1 < len(validated_messages) and validated_messages[i+1]["role"] == "user":
                # Look ahead for image messages from the same user
                image_content_items = []
                next_idx = i + 1
                
                while (next_idx < len(validated_messages) and 
                       validated_messages[next_idx]["role"] == msg_role and
                       validated_messages[next_idx].get("type", "text") in ["image_url", "image_base64"]):
                    
                    next_msg = validated_messages[next_idx]
                    next_type = next_msg.get("type", "text")
                    
                    if next_type == "image_url":
                        image_content_items.append({
                            "type": "image_url",
                            "image_url": {
                                "url": next_msg["image_url"],
                                "detail": next_msg["image_detail"]
                            }
                        })
                    elif next_type == "image_base64":
                        image_content_items.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{next_msg['media_type']};base64,{next_msg['image_data']}",
                                "detail": next_msg["image_detail"]
                            }
                        })
                    
                    next_idx += 1
                
                # If we found images to combine with this text message
                if image_content_items:
                    content_items = [{"type": "text", "text": text_content}] + image_content_items
                    openai_messages.append({
                        "role": msg_role,
                        "content": content_items
                    })
                    i = next_idx  # Skip the processed image messages
                    continue
            
            # Simple text message (no images to combine)
            openai_messages.append({
                "role": msg_role,
                "content": text_content
            })
        
        # Handle image messages that weren't combined with text
        elif msg_type == "image_url":
            openai_messages.append({
                "role": msg_role,
                "content": [
                    {"type": "text", "text": ""},  # Empty text
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": msg["image_url"], 
                            "detail": msg["image_detail"]
                        }
                    }
                ]
            })
        elif msg_type == "image_base64":
            openai_messages.append({
                "role": msg_role,
                "content": [
                    {"type": "text", "text": ""},  # Empty text
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:{msg['media_type']};base64,{msg['image_data']}",
                            "detail": msg["image_detail"]
                        }
                    }
                ]
            })
        
        i += 1
    
    # Add system message if provided
    if system_message:
        if supports_feature(model_name, "developer_message"):
            openai_messages.insert(0, {"role": "developer", "content": system_message})
        elif supports_feature(model_name, "system_message"):
            openai_messages.insert(0, {"role": "system", "content": system_message})
            
    return openai_messages

# # -------------------------------------------------------------------------------- #
# # OpenAI Message Mapper Tests
# # -------------------------------------------------------------------------------- #

# def test_to_openai_messages_text_only():
#     """Test conversion of text-only messages."""
#     from astral_ai.constants._models import OpenAIModels
    
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "Hello", "type": "text"},
#         {"role": "assistant", "content": "Hi there!", "type": "text"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message="You are a helpful assistant."
#     )
    
#     assert len(result) == 3
#     # System message
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "You are a helpful assistant."
#     # User message
#     assert result[1]["role"] == "user"
#     assert result[1]["content"] == "Hello"
#     # Assistant message
#     assert result[2]["role"] == "assistant"
#     assert result[2]["content"] == "Hi there!"

# def test_to_openai_messages_with_remote_image():
#     """Test conversion of messages with a remote image."""
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "What's in this image?", "type": "text"},
#         {"role": "user", "type": "image_url", "image_url": "https://example.com/image.jpg", "image_detail": "high"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message=None
#     )
    
#     assert len(result) == 1
#     # Combined text and image message
#     assert result[0]["role"] == "user"
#     assert isinstance(result[0]["content"], list)
#     assert len(result[0]["content"]) == 2
#     # Text part
#     assert result[0]["content"][0]["type"] == "text"
#     assert result[0]["content"][0]["text"] == "What's in this image?"
#     # Image part
#     assert result[0]["content"][1]["type"] == "image_url"
#     assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
#     assert result[0]["content"][1]["image_url"]["detail"] == "high"

# def test_to_openai_messages_with_base64_image():
#     """Test conversion of messages with a base64 encoded image."""
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "What's in this image?", "type": "text"},
#         {
#             "role": "user", 
#             "type": "image_base64", 
#             "image_data": "base64data", 
#             "media_type": "image/jpeg", 
#             "image_detail": "low"
#         }
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message=None
#     )
    
#     assert len(result) == 1
#     # Combined text and image message
#     assert result[0]["role"] == "user"
#     assert isinstance(result[0]["content"], list)
#     assert len(result[0]["content"]) == 2
#     # Text part
#     assert result[0]["content"][0]["type"] == "text"
#     assert result[0]["content"][0]["text"] == "What's in this image?"
#     # Image part
#     assert result[0]["content"][1]["type"] == "image_url"
#     assert result[0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,base64data"
#     assert result[0]["content"][1]["image_url"]["detail"] == "low"

# def test_to_openai_messages_mixed_types():
#     """Test conversion of messages with mixed types."""
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "Hello", "type": "text"},
#         {"role": "assistant", "content": "Hi there!", "type": "text"},
#         {"role": "user", "content": "What's in this image?", "type": "text"},
#         {"role": "user", "type": "image_url", "image_url": "https://example.com/image.jpg", "image_detail": "high"},
#         {
#             "role": "user", 
#             "type": "image_base64", 
#             "image_data": "base64data", 
#             "media_type": "image/jpeg", 
#             "image_detail": "low"
#         },
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message="You are a helpful assistant."
#     )
    
#     assert len(result) == 4
#     # System message
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "You are a helpful assistant."
#     # First user message
#     assert result[1]["role"] == "user"
#     assert result[1]["content"] == "Hello"
#     # Assistant message
#     assert result[2]["role"] == "assistant"
#     assert result[2]["content"] == "Hi there!"
#     # Second user message with images
#     assert result[3]["role"] == "user"
#     assert isinstance(result[3]["content"], list)
#     assert len(result[3]["content"]) == 3
#     # Text part
#     assert result[3]["content"][0]["type"] == "text"
#     assert result[3]["content"][0]["text"] == "What's in this image?"
#     # First image
#     assert result[3]["content"][1]["type"] == "image_url"
#     assert result[3]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
#     # Second image
#     assert result[3]["content"][2]["type"] == "image_url"
#     assert "data:image/jpeg;base64," in result[3]["content"][2]["image_url"]["url"]

# def test_to_openai_messages_system_only():
#     """Test conversion with only a system message."""
#     from astral_ai.constants._models import OpenAIModels
    
#     messages = ValidatedMessageList(messages=[])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message="You are a helpful assistant."
#     )
    
#     assert len(result) == 1
#     # System message
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "You are a helpful assistant."

# def test_to_openai_messages_system_in_list():
#     """Test conversion with a system message in the message list."""
#     from astral_ai.constants._models import OpenAIModels
    
#     messages = ValidatedMessageList(messages=[
#         {"role": "system", "content": "You are a helpful assistant.", "type": "text"},
#         {"role": "user", "content": "Hello", "type": "text"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message=None
#     )
    
#     assert len(result) == 2
#     # System message
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "You are a helpful assistant."
#     # User message
#     assert result[1]["role"] == "user"
#     assert result[1]["content"] == "Hello"

# def test_to_openai_messages_system_priority():
#     """Test that system message parameter takes priority when there's a converted system message in the list."""
#     from astral_ai.constants._models import OpenAIModels
    
#     # The system message in the list will be converted to assistant by standardize_messages
#     messages = ValidatedMessageList(messages=[
#         {"role": "assistant", "content": "I was originally a system message in the list.", "type": "text"},
#         {"role": "user", "content": "Hello", "type": "text"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message="I am the system message parameter."
#     )
    
#     assert len(result) == 3
#     # System message parameter should be used
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "I am the system message parameter."
#     # Former system message (now assistant)
#     assert result[1]["role"] == "assistant"
#     assert result[1]["content"] == "I was originally a system message in the list."
#     # User message
#     assert result[2]["role"] == "user"
#     assert result[2]["content"] == "Hello"

# def test_to_openai_messages_multi_turn_with_images():
#     """Test conversion with a multi-turn conversation including images."""
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "Hello", "type": "text"},
#         {"role": "assistant", "content": "Hi there! How can I help you?", "type": "text"},
#         {"role": "user", "content": "What's in this image?", "type": "text"},
#         {"role": "user", "type": "image_url", "image_url": "https://example.com/image.jpg", "image_detail": "high"},
#         {"role": "assistant", "content": "This is an image of a cat.", "type": "text"},
#         {"role": "user", "content": "What about this one?", "type": "text"},
#         {"role": "user", "type": "image_base64", "image_data": "base64data", "media_type": "image/jpeg", "image_detail": "low"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message="You are a helpful assistant."
#     )
    
#     assert len(result) == 6
#     # System message
#     assert result[0]["role"] == "system"
#     assert result[0]["content"] == "You are a helpful assistant."
#     # First user message
#     assert result[1]["role"] == "user"
#     assert result[1]["content"] == "Hello"
#     # First assistant message
#     assert result[2]["role"] == "assistant"
#     assert result[2]["content"] == "Hi there! How can I help you?"
#     # Second user message with image
#     assert result[3]["role"] == "user"
#     assert isinstance(result[3]["content"], list)
#     assert len(result[3]["content"]) == 2
#     assert result[3]["content"][0]["type"] == "text"
#     assert result[3]["content"][0]["text"] == "What's in this image?"
#     assert result[3]["content"][1]["type"] == "image_url"
#     assert result[3]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
#     # Second assistant message
#     assert result[4]["role"] == "assistant"
#     assert result[4]["content"] == "This is an image of a cat."
#     # Third user message with second image
#     assert result[5]["role"] == "user"
#     assert isinstance(result[5]["content"], list)
#     assert len(result[5]["content"]) == 2
#     assert result[5]["content"][0]["type"] == "text"
#     assert result[5]["content"][0]["text"] == "What about this one?"
#     assert result[5]["content"][1]["type"] == "image_url"
#     assert "data:image/jpeg;base64," in result[5]["content"][1]["image_url"]["url"]

# def test_to_openai_messages_multiple_images():
#     """Test conversion with multiple images in sequence."""
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "Compare these two images:", "type": "text"},
#         {"role": "user", "type": "image_url", "image_url": "https://example.com/image1.jpg", "image_detail": "high"},
#         {"role": "user", "type": "image_url", "image_url": "https://example.com/image2.jpg", "image_detail": "high"}
#     ])
    
#     result = to_openai_messages(
#         model_name="gpt-4o",
#         messages=messages,
#         system_message=None
#     )
    
#     assert len(result) == 1
#     # Combined message with text and two images
#     assert result[0]["role"] == "user"
#     assert isinstance(result[0]["content"], list)
#     assert len(result[0]["content"]) == 3
#     # Text part
#     assert result[0]["content"][0]["type"] == "text"
#     assert result[0]["content"][0]["text"] == "Compare these two images:"
#     # First image
#     assert result[0]["content"][1]["type"] == "image_url"
#     assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/image1.jpg"
#     # Second image
#     assert result[0]["content"][2]["type"] == "image_url"
#     assert result[0]["content"][2]["image_url"]["url"] == "https://example.com/image2.jpg"

# def test_to_openai_messages_developer_message():
#     """Test conversion with a model that supports developer messages."""
#     from astral_ai.constants._models import OpenAIModels
#     from astral_ai.constants._model_capabilities import supports_feature
#     import unittest.mock as mock
    
#     messages = ValidatedMessageList(messages=[
#         {"role": "user", "content": "Hello", "type": "text"}
#     ])
    
#     # Mock the supports_feature function to simulate a model with developer message support
#     with mock.patch('astral_ai.providers.openai._mapper.supports_feature') as mock_supports:
#         # Configure mock to return True for developer_message and False for system_message
#         def side_effect(model, feature):
#             if feature == "developer_message":
#                 return True
#             return False
            
#         mock_supports.side_effect = side_effect
        
#         result = to_openai_messages(
#             model_name="o1-mini",  # Use a model name, actual capabilities will be mocked
#             messages=messages,
#             system_message="You are a helpful assistant."
#         )
    
#     assert len(result) == 2
#     # Developer message (since we mocked the support)
#     assert result[0]["role"] == "developer"
#     assert result[0]["content"] == "You are a helpful assistant."
#     # User message
#     assert result[1]["role"] == "user"
#     assert result[1]["content"] == "Hello"

# if __name__ == "__main__":
#     # Run tests
#     test_to_openai_messages_text_only()
#     test_to_openai_messages_with_remote_image()
#     test_to_openai_messages_with_base64_image()
#     test_to_openai_messages_mixed_types()
#     test_to_openai_messages_system_only()
#     test_to_openai_messages_system_in_list()
#     test_to_openai_messages_system_priority()
#     test_to_openai_messages_multi_turn_with_images()
#     test_to_openai_messages_multiple_images()
#     test_to_openai_messages_developer_message()
#     print("All OpenAI mapper tests passed!")
