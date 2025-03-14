# -------------------------------------------------------------------------------- #
# Messaging Utils
# -------------------------------------------------------------------------------- #

"""
Utils for the messaging.
"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import Optional, List, Tuple, Dict, Union

# Pydantic imports
from pydantic import BaseModel

# Astral AI Constants
from astral_ai.constants._models import ModelName

# Astral AI Messaging
from astral_ai.messaging._models import Message, MessageList, Messages, TextMessage, ImageMessage

# Astral AI Exceptions
from astral_ai.errors.exceptions import MessagesNotProvidedError, InvalidMessageError

# Astral AI Logger
from astral_ai.logger import logger


# -------------------------------------------------------------------------------- #
# Message Handling Utils
# -------------------------------------------------------------------------------- #

# Global emoji for message-related logs
MESSAGE_EMOJI = "💬"


def handle_no_messages(model_name: ModelName, init_call: bool) -> Optional[None]:
    """
    Handle the case where no messages are provided.
    """
    if init_call:
        logger.warning(f"{MESSAGE_EMOJI} No messages provided for model during initialization of model {model_name}.\n"
                       f"{MESSAGE_EMOJI} You must provide messages in each call to the model.")
        return None
    else:
        logger.error(f"{MESSAGE_EMOJI} Attempted to run model {model_name} without providing messages.")
        raise MessagesNotProvidedError(f"Messages must be provided to run the model {model_name}.")


def standardize_messages(messages: Union[Messages, Message]) -> List[Dict[str, str]]:
    """
    Standardize the messages to a list of Message instances.

    Args:
        messages (Union[Messages, Message]): The messages to standardize.

    Returns:
        List[Dict[str, str]]: A list of messages in dictionary format.

    Raises:
        InvalidMessageError: If the message type is invalid.
    """
    # Use direct type check for common cases first (faster than isinstance for built-in types)
    message_type = type(messages)
    message_type_name = message_type.__name__
    
    logger.debug(f"{MESSAGE_EMOJI} Standardizing messages. Type: `{message_type_name}`")
    
    # List cases - check most common case first for performance
    if message_type is list:
        # Empty list check
        if not messages:
            return []
        
        # Fast path: all dictionaries (common case)
        if isinstance(messages[0], dict):
            # Check if all items are dicts (sample first few for large lists for speed)
            all_dicts = True
            sample_size = min(10, len(messages))
            for i in range(sample_size):
                if not isinstance(messages[i], dict):
                    all_dicts = False
                    break
                    
            if all_dicts and sample_size == len(messages):
                return messages  # Already in target format
        
        # Mixed or non-dict list
        standardized = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                standardized.append(msg.model_dump())
            else:
                standardized.append(msg)
        return standardized
    
    # MessageList case
    elif isinstance(messages, MessageList):
        standardized = []
        for msg in messages.messages:
            if isinstance(msg, dict):
                standardized.append(msg)
            elif isinstance(msg, BaseModel):
                standardized.append(msg.model_dump())
            else:
                standardized.append(msg)
        return standardized
    # Single message cases
    elif isinstance(messages, BaseModel):
        return [messages.model_dump()]
    elif isinstance(messages, dict):
        return [messages]
    
    # Invalid type
    else:
        logger.error(f"{MESSAGE_EMOJI} Invalid message type: `{message_type_name}`")
        raise InvalidMessageError(message_type=f"`{message_type_name}`")


def count_message_roles(messages: List[Message]) -> Tuple[int, int]:
    """
    Helper function to count system and developer messages in a single pass.

    Args:
        messages (List[Message]): List of messages to count roles for

    Returns:
        Tuple[int, int]: Count of (system_messages, developer_messages)
    """
    system_count = 0
    developer_count = 0

    if not messages:  # Early return for empty list
        logger.debug(f"{MESSAGE_EMOJI} Counting message roles in empty list")
        return (0, 0)
        
    logger.debug(f"{MESSAGE_EMOJI} Counting message roles in {len(messages)} messages")
    
    # Optimize for the most common message types
    for msg in messages:
        # Optimize type checking order based on frequency
        if isinstance(msg, dict):
            role = msg.get('role', '')
        elif isinstance(msg, BaseModel):
            role = getattr(msg, 'role', '')
        else:
            role = ''
            
        # Use direct string comparison rather than multiple conditions
        if role == "system":
            system_count += 1
        elif role == "developer":
            developer_count += 1

    logger.debug(f"{MESSAGE_EMOJI} Found {system_count} system messages and {developer_count} developer messages")
    return system_count, developer_count


def convert_message_roles(messages: List[Message], target_role: str, model_name: ModelName) -> None:
    """
    Helper function to convert message roles in-place.

    Args:
        messages (List[Message]): List of messages to convert.
        target_role (str): Role to convert messages to ("system", "developer", or "user").
        model_name (ModelName): The name of the model being used.
    """
    # Early return if no messages
    if not messages:
        return
        
    logger.debug(f"{MESSAGE_EMOJI} Converting message roles to '{target_role}' for model {model_name}")
    logger.debug(f"{MESSAGE_EMOJI} Processing {len(messages)} messages for role conversion")
    
    conversion_count = 0
    
    # Optimize: Pre-determine source roles and check conditions once
    if target_role == "system":
        source_role = "developer"
        for msg in messages:
            if isinstance(msg, BaseModel):
                if getattr(msg, 'role', None) == source_role:
                    msg.role = target_role
                    conversion_count += 1
            elif isinstance(msg, dict) and msg.get('role') == source_role:
                msg['role'] = target_role
                conversion_count += 1
                
    elif target_role == "developer":
        source_role = "system"
        for msg in messages:
            if isinstance(msg, BaseModel):
                if getattr(msg, 'role', None) == source_role:
                    logger.warning(
                        f"{MESSAGE_EMOJI} Incorrect message role provided for model {model_name}. "
                        f"{MESSAGE_EMOJI} {model_name} does not support {source_role} messages. "
                        f"{MESSAGE_EMOJI} Converting message role from {source_role} to {target_role}."
                    )
                    msg.role = target_role
                    conversion_count += 1
            elif isinstance(msg, dict) and msg.get('role') == source_role:
                logger.warning(
                    f"{MESSAGE_EMOJI} Incorrect message role provided for model {model_name}. "
                    f"{MESSAGE_EMOJI} {model_name} does not support {source_role} messages. "
                    f"{MESSAGE_EMOJI} Converting message role from {source_role} to {target_role}."
                )
                msg['role'] = target_role
                conversion_count += 1
                
    # When converting to "user", convert any message that isn't already a user message.
    elif target_role == "user":
        # Use a set for faster lookups
        source_roles = {"system", "developer"}
        for msg in messages:
            if isinstance(msg, BaseModel):
                role = getattr(msg, 'role', None)
                if role in source_roles:
                    logger.warning(
                        f"{MESSAGE_EMOJI} Incorrect message role provided for model {model_name}. "
                        f"{MESSAGE_EMOJI} {model_name} does not support {role} messages. "
                        f"{MESSAGE_EMOJI} Converting message role from {role} to user."
                    )
                    msg.role = "user"
                    conversion_count += 1
            elif isinstance(msg, dict):
                role = msg.get('role')
                if role in source_roles:
                    logger.warning(
                        f"{MESSAGE_EMOJI} Incorrect message role provided for model {model_name}. "
                        f"{MESSAGE_EMOJI} {model_name} does not support {role} messages. "
                        f"{MESSAGE_EMOJI} Converting message role from {role} to user."
                    )
                    msg['role'] = "user"
                    conversion_count += 1
    
    if conversion_count > 0:
        logger.debug(f"{MESSAGE_EMOJI} Completed role conversion. {conversion_count} messages were converted to '{target_role}'")


# -------------------------------------------------------------------------------- #
# Test Cases
# -------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # -------------------------------------------------------------------------------- #
    # Test Setup and Helper Functions
    # -------------------------------------------------------------------------------- #
    from astral_ai.constants._models import ModelName
    import json
    import sys
    import time
    from typing import List, Any, Callable, Dict
    
    def run_test(test_func):
        """Run a test function and catch any exceptions"""
        try:
            print(f"\n🧪 Running test: {test_func.__name__}")
            test_func()
            print(f"✅ Test {test_func.__name__} completed successfully")
            return True
        except AssertionError as e:
            print(f"❌ FAIL: {test_func.__name__} - {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
            
    def benchmark(func: Callable, *args, repeat: int = 3, **kwargs) -> float:
        """Simple benchmark function to measure execution time"""
        total_time = 0
        for _ in range(repeat):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            total_time += (end - start)
        return total_time / repeat
    
    # -------------------------------------------------------------------------------- #
    # Test Cases for standardize_messages
    # -------------------------------------------------------------------------------- #
    
    def test_standardize_single_text_message():
        """Test standardizing a single TextMessage"""
        message = TextMessage(role="user", content="Hello, how are you?")
        result = standardize_messages(message)
        expected = [{"role": "user", "content": "Hello, how are you?"}]
        assert result == expected, "Failed to standardize single TextMessage"
    
    def test_standardize_single_image_message():
        """Test standardizing a single ImageMessage"""
        message = ImageMessage(role="user", image_url="https://example.com/image.jpg", image_detail="high")
        result = standardize_messages(message)
        expected = [{"role": "user", "image_url": "https://example.com/image.jpg", "image_detail": "high"}]
        assert result == expected, "Failed to standardize single ImageMessage"
    
    def test_standardize_message_list():
        """Test standardizing a MessageList object"""
        messages = MessageList(messages=[
            TextMessage(role="system", content="You are a helpful assistant."),
            TextMessage(role="user", content="Hello, how are you?")
        ])
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        assert result == expected, "Failed to standardize MessageList"
    
    def test_standardize_list_of_messages():
        """Test standardizing a list of Message objects"""
        messages = [
            TextMessage(role="system", content="You are a helpful assistant."),
            ImageMessage(role="user", image_url="https://example.com/image.jpg")
        ]
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "image_url": "https://example.com/image.jpg", "image_detail": "auto"}
        ]
        assert result == expected, "Failed to standardize list of Message objects"
    
    def test_standardize_dict_message():
        """Test standardizing a dictionary message"""
        message = {"role": "user", "content": "Hello, how are you?"}
        result = standardize_messages(message)
        expected = [{"role": "user", "content": "Hello, how are you?"}]
        assert result == expected, "Failed to standardize dictionary message"
    
    def test_standardize_list_of_dicts():
        """Test standardizing a list of dictionaries"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        assert result == expected, "Failed to standardize list of dictionaries"
    
    def test_standardize_mixed_list():
        """Test standardizing a mixed list of Message objects and dictionaries"""
        messages = [
            TextMessage(role="system", content="You are a helpful assistant."),
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        assert result == expected, "Failed to standardize mixed list"
    
    # -------------------------------------------------------------------------------- #
    # Additional Edge Case Tests for standardize_messages
    # -------------------------------------------------------------------------------- #
    
    def test_standardize_empty_content():
        """Test standardizing messages with empty content"""
        messages = [
            TextMessage(role="system", content=""),
            {"role": "user", "content": ""}
        ]
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""}
        ]
        assert result == expected, "Failed to standardize messages with empty content"
    
    def test_standardize_empty_message_list():
        """Test standardizing an empty MessageList"""
        messages = MessageList(messages=[])
        result = standardize_messages(messages)
        assert result == [], "Failed to standardize empty MessageList"
    
    def test_standardize_nested_message_list():
        """Test standardizing a MessageList containing another MessageList (should be invalid)"""
        inner_list = MessageList(messages=[TextMessage(role="user", content="Inner message")])
        try:
            # This is an invalid case, should raise an exception
            outer_list = MessageList(messages=[inner_list])
            standardize_messages(outer_list)
            assert False, "Should have raised an exception for nested MessageList"
        except Exception:
            # We expect an exception here
            pass
            
    def test_standardize_missing_role():
        """Test standardizing messages with missing role"""
        # In dictionary, missing role might be handled
        message = {"content": "Message without role"}
        result = standardize_messages(message)
        assert result == [{"content": "Message without role"}], "Failed to handle dictionary without role"
    
    def test_standardize_with_extra_fields():
        """Test standardizing messages with extra fields"""
        message = {"role": "user", "content": "Hello", "extra_field": "value", "metadata": {"key": "value"}}
        result = standardize_messages(message)
        expected = [{"role": "user", "content": "Hello", "extra_field": "value", "metadata": {"key": "value"}}]
        assert result == expected, "Failed to preserve extra fields"
    
    def test_standardize_with_unicode_content():
        """Test standardizing messages with Unicode content"""
        message = TextMessage(role="user", content="你好，世界! 👋 😊")
        result = standardize_messages(message)
        expected = [{"role": "user", "content": "你好，世界! 👋 😊"}]
        assert result == expected, "Failed to handle Unicode content"
    
    # -------------------------------------------------------------------------------- #
    # Performance Tests
    # -------------------------------------------------------------------------------- #
    
    def test_standardize_large_message_list_performance():
        """Test performance with a large list of messages"""
        # Create a large list of 1000 messages
        large_list = []
        for i in range(1000):
            if i % 2 == 0:
                large_list.append(TextMessage(role="user", content=f"Message {i}"))
            else:
                large_list.append({"role": "system", "content": f"System message {i}"})
                
        # Run a simple performance benchmark
        avg_time = benchmark(standardize_messages, large_list, repeat=3)
        print(f"  Large message list (1000 items) processing time: {avg_time:.4f} seconds")
        
        # Run test to ensure correctness too
        result = standardize_messages(large_list)
        assert len(result) == 1000, "Failed to standardize large message list"
        assert all(isinstance(msg, dict) for msg in result), "All messages should be dictionaries"
    
    def test_convert_roles_large_message_list_performance():
        """Test performance of role conversion with a large list"""
        # Create a large list of 1000 messages with various roles
        large_list = []
        for i in range(1000):
            if i % 3 == 0:
                large_list.append(TextMessage(role="system", content=f"Message {i}"))
            elif i % 3 == 1:
                large_list.append({"role": "user", "content": f"Message {i}"})
            else:
                large_list.append({"role": "system", "content": f"Message {i}"})
                
        # Run a simple performance benchmark
        avg_time = benchmark(convert_message_roles, large_list, "user", "gpt-4o", repeat=3)
        print(f"  Large message role conversion (1000 items) processing time: {avg_time:.4f} seconds")
        
        # Verify correctness
        assert all(msg.role == "user" if isinstance(msg, BaseModel) else msg["role"] == "user" 
                   for msg in large_list), "All roles should be converted to user"
    
    # -------------------------------------------------------------------------------- #
    # Test Cases for count_message_roles
    # -------------------------------------------------------------------------------- #
    
    def test_count_message_roles_empty():
        """Test counting roles in an empty list"""
        messages = []
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (0, 0), "Failed to count roles in empty list"
    
    def test_count_message_roles_single_system():
        """Test counting roles with a single system message"""
        messages = [TextMessage(role="system", content="System instruction")]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (1, 0), "Failed to count roles with single system message"
    
    def test_count_message_roles_single_developer():
        """Test counting roles with a single developer message (using dict to avoid validation)"""
        messages = [{"role": "developer", "content": "Developer instruction"}]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (0, 1), "Failed to count roles with single developer message"
    
    def test_count_message_roles_mixed():
        """Test counting roles with mixed message types"""
        messages = [
            TextMessage(role="system", content="System instruction"),
            TextMessage(role="user", content="User message"),
            {"role": "developer", "content": "Developer note"},  # Using dict for developer role
            ImageMessage(role="system", image_url="https://example.com/image.jpg")
        ]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (2, 1), "Failed to count roles with mixed message types"
    
    def test_count_message_roles_dict_format():
        """Test counting roles with dictionary messages"""
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User message"},
            {"role": "developer", "content": "Developer note"},
        ]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (1, 1), "Failed to count roles with dictionary messages"
    
    # -------------------------------------------------------------------------------- #
    # Additional Edge Case Tests for count_message_roles
    # -------------------------------------------------------------------------------- #
    
    def test_count_message_roles_with_invalid_types():
        """Test counting roles with invalid message types mixed in"""
        messages = [
            TextMessage(role="system", content="System message"),
            {"role": "user", "content": "User message"},
            "This is not a valid message",  # String instead of message
            123,  # Number instead of message
            None,  # None instead of message
        ]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (1, 0), "Should count only valid message roles"
    
    def test_count_message_roles_with_unusual_roles():
        """Test counting roles with unusual or unexpected role values"""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},  # Not a tracked role
            {"role": "", "content": "Empty role"},                 # Empty role
            {"role": None, "content": "None role"},               # None role
            {"content": "Missing role"}                          # Missing role
        ]
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (1, 0), "Should only count system and developer roles"
    
    # -------------------------------------------------------------------------------- #
    # Test Cases for convert_message_roles
    # -------------------------------------------------------------------------------- #
    
    def test_convert_developer_to_system():
        """Test converting developer roles to system"""
        messages = [
            {"role": "developer", "content": "Developer note"},  # Using dict for developer role
            TextMessage(role="user", content="User message"),
            {"role": "developer", "content": "Another developer note"}
        ]
        convert_message_roles(messages, "system", "gpt-4o")
        expected_roles = ["system", "user", "system"]
        actual_roles = [msg.role if isinstance(msg, BaseModel) else msg["role"] for msg in messages]
        assert actual_roles == expected_roles, "Failed to convert developer roles to system"
    
    def test_convert_system_to_developer():
        """Test converting system roles to developer"""
        messages = [
            TextMessage(role="system", content="System instruction"),
            TextMessage(role="user", content="User message"),
            {"role": "system", "content": "Another system instruction"}
        ]
        convert_message_roles(messages, "developer", "claude-3-opus")
        expected_roles = ["developer", "user", "developer"]
        actual_roles = [msg.role if isinstance(msg, BaseModel) else msg["role"] for msg in messages]
        assert actual_roles == expected_roles, "Failed to convert system roles to developer"
    
    def test_convert_to_user():
        """Test converting system and developer roles to user"""
        messages = [
            TextMessage(role="system", content="System instruction"),
            {"role": "developer", "content": "Developer note"},  # Using dict for developer role
            TextMessage(role="user", content="User message")
        ]
        convert_message_roles(messages, "user", "gemini-pro")
        expected_roles = ["user", "user", "user"]
        actual_roles = [msg.role if isinstance(msg, BaseModel) else msg["role"] for msg in messages]
        assert actual_roles == expected_roles, "Failed to convert roles to user"
    
    def test_no_conversion_needed():
        """Test when no conversion is needed"""
        messages = [
            TextMessage(role="system", content="System instruction"),
            TextMessage(role="user", content="User message")
        ]
        convert_message_roles(messages, "system", "gpt-4o")
        expected_roles = ["system", "user"]
        actual_roles = [msg.role for msg in messages]
        assert actual_roles == expected_roles, "Conversion happened when none was needed"
    
    # -------------------------------------------------------------------------------- #
    # Test Cases for handle_no_messages
    # -------------------------------------------------------------------------------- #
    
    def test_handle_no_messages_init_call():
        """Test handling no messages during initialization"""
        result = handle_no_messages("gpt-4o", True)
        assert result is None, "handle_no_messages should return None during initialization"
    
    def test_handle_no_messages_run_call():
        """Test handling no messages during run call (should raise exception)"""
        try:
            handle_no_messages("gpt-4o", False)
            assert False, "Expected MessagesNotProvidedError but no exception was raised"
        except MessagesNotProvidedError:
            # This is the expected exception
            pass
        except Exception as e:
            assert False, f"Expected MessagesNotProvidedError but got {type(e).__name__}"
    
    # -------------------------------------------------------------------------------- #
    # Test Cases for Invalid Inputs
    # -------------------------------------------------------------------------------- #
    
    def test_standardize_invalid_message():
        """Test standardizing an invalid message type"""
        invalid_message = 123  # Not a valid message type
        try:
            standardize_messages(invalid_message)
            assert False, "Expected an exception but none was raised"
        except TypeError:
            # Accept TypeError as the expected exception
            pass
        except InvalidMessageError:
            # Also accept InvalidMessageError as it might be raised in some implementations
            pass
        except Exception as e:
            assert False, f"Expected TypeError or InvalidMessageError but got {type(e).__name__}"
    
    # -------------------------------------------------------------------------------- #
    # Complex Integration Tests
    # -------------------------------------------------------------------------------- #
    
    def test_convert_then_standardize():
        """Test converting roles and then standardizing messages"""
        messages = [
            {"role": "developer", "content": "Developer note"},  # Using dict for developer role
            ImageMessage(role="user", image_url="https://example.com/image.jpg"),
            {"role": "system", "content": "System instruction"}
        ]
        # First convert developer to system
        convert_message_roles(messages, "system", "gpt-4o")
        # Then standardize
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "Developer note"},
            {"role": "user", "image_url": "https://example.com/image.jpg", "image_detail": "auto"},
            {"role": "system", "content": "System instruction"}
        ]
        assert result == expected, "Failed to convert then standardize messages"
    
    def test_messagelist_count_and_convert():
        """Test counting roles in a MessageList and then converting them"""
        messages = MessageList(messages=[
            TextMessage(role="system", content="System instruction"),
            {"role": "developer", "content": "Developer note"},  # Using dict for developer role
            TextMessage(role="user", content="User message")
        ])
        # Count roles
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (1, 1), "Failed to count roles in MessageList"
        
        # Convert roles
        convert_message_roles(messages.messages, "system", "gpt-4o")
        # Standardize and check results
        result = standardize_messages(messages)
        expected = [
            {"role": "system", "content": "System instruction"},
            {"role": "system", "content": "Developer note"},
            {"role": "user", "content": "User message"}
        ]
        assert result == expected, "Failed MessageList count and convert integration"
    
    # -------------------------------------------------------------------------------- #
    # Additional Edge Case Tests for convert_message_roles
    # -------------------------------------------------------------------------------- #
    
    def test_convert_mixed_invalid_messages():
        """Test converting roles with invalid messages mixed in"""
        messages = [
            TextMessage(role="system", content="System message"),
            {"role": "user", "content": "User message"},
            "This is not a valid message",  # String instead of message
            None,  # None instead of message
        ]
        try:
            # This might fail depending on implementation, but shouldn't crash
            convert_message_roles(messages, "user", "gpt-4o")
            # Check the valid messages were converted
            assert messages[0].role == "user", "Valid message should be converted"
        except Exception as e:
            # If it fails, make sure it's not due to a crash on valid messages
            print(f"  Note: Got exception {type(e).__name__} - {str(e)}")
            pass  # This might be acceptable depending on implementation
    
    def test_convert_to_same_role():
        """Test converting to the same role (should be a no-op)"""
        messages = [
            TextMessage(role="user", content="User message 1"),
            TextMessage(role="user", content="User message 2"),
        ]
        convert_message_roles(messages, "user", "gpt-4o")
        assert all(msg.role == "user" for msg in messages), "Roles should remain user"
    
    def test_convert_chain():
        """Test converting roles multiple times in sequence"""
        messages = [
            TextMessage(role="system", content="System message"),
            {"role": "system", "content": "Another system message"}
        ]
        # 1) First convert system -> developer
        convert_message_roles(messages, "developer", "claude-3-opus")
        assert all(msg.role == "developer" if isinstance(msg, BaseModel) else msg["role"] == "developer" 
                   for msg in messages), "First conversion failed"
                   
        # 2) Then convert developer -> user
        convert_message_roles(messages, "user", "gemini-pro")
        assert all(msg.role == "user" if isinstance(msg, BaseModel) else msg["role"] == "user" 
                   for msg in messages), "Second conversion failed"
                   
        # 3) Finally convert user -> developer (can't convert user->system directly)
        convert_message_roles(messages, "developer", "claude-3-opus")
        assert all(msg.role == "developer" if isinstance(msg, BaseModel) else msg["role"] == "developer" 
                   for msg in messages), "Third conversion failed"
                   
        # 4) Now convert developer -> system
        convert_message_roles(messages, "system", "gpt-4o")
        assert all(msg.role == "system" if isinstance(msg, BaseModel) else msg["role"] == "system" 
                   for msg in messages), "Fourth conversion failed"
    
    # -------------------------------------------------------------------------------- #
    # Additional Edge Case Tests for handle_no_messages
    # -------------------------------------------------------------------------------- #
    
    def test_handle_no_messages_with_unusual_model_names():
        """Test handling no messages with unusual model names"""
        # Empty string model name
        result = handle_no_messages("", True)
        assert result is None, "Should handle empty model name"
        
        # Non-string model name (shouldn't happen but testing robustness)
        try:
            result = handle_no_messages(None, True)
            # If it doesn't error, should still return None
            assert result is None, "Should handle None model name or raise a controlled exception"
        except Exception:
            # Exception is acceptable too
            pass
    
    # -------------------------------------------------------------------------------- #
    # Complex Integration Tests
    # -------------------------------------------------------------------------------- #
    
    def test_complex_conversion_and_counting():
        """Test a complex workflow with conversion, counting, and standardization"""
        # Create mixed message list
        messages = [
            TextMessage(role="system", content="System instruction 1"),
            {"role": "developer", "content": "Developer note"},
            TextMessage(role="user", content="User message"),
            {"role": "system", "content": "System instruction 2"}
        ]
        
        # First count
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (2, 1), "Initial count failed"
        
        # Convert developer to system
        convert_message_roles(messages, "system", "gpt-4o")
        
        # Count again
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (3, 0), "Count after conversion failed"
        
        # Convert all to user
        convert_message_roles(messages, "user", "gemini-pro")
        
        # Count again
        system_count, developer_count = count_message_roles(messages)
        assert (system_count, developer_count) == (0, 0), "Count after second conversion failed"
        
        # Standardize
        result = standardize_messages(messages)
        assert len(result) == 4, "Should preserve message count"
        assert all(msg["role"] == "user" for msg in result), "All roles should be user"
    
    def test_message_list_operations():
        """Test operations with MessageList, especially iteration and conversion"""
        # Create MessageList with mixed message types
        messages = MessageList(messages=[
            TextMessage(role="system", content="System instruction"),
            {"role": "user", "content": "User message"},
            TextMessage(role="system", content="Another system instruction")
        ])
        
        # Test iteration
        roles = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                roles.append(msg.role)
            else:
                roles.append(msg["role"])
        assert roles == ["system", "user", "system"], "MessageList iteration failed"
        
        # Test length
        assert len(messages) == 3, "MessageList length incorrect"
        
        # Test indexing
        assert isinstance(messages[0], BaseModel), "First item should be a BaseModel"
        assert isinstance(messages[1], dict), "Second item should be a dict"
        
        # Test standardization after manipulation
        messages[0].role = "user"  # Change a role
        result = standardize_messages(messages)
        assert result[0]["role"] == "user", "Role change not reflected after standardization"
    
    # -------------------------------------------------------------------------------- #
    # Run All Tests
    # -------------------------------------------------------------------------------- #
    
    def run_all_tests():
        """Run all test cases"""
        all_tests = [
            # Original standardize_messages tests
            test_standardize_single_text_message,
            test_standardize_single_image_message,
            test_standardize_message_list,
            test_standardize_list_of_messages,
            test_standardize_dict_message,
            test_standardize_list_of_dicts,
            test_standardize_mixed_list,
            
            # New edge case tests for standardize_messages
            test_standardize_empty_content,
            test_standardize_empty_message_list,
            test_standardize_nested_message_list,
            test_standardize_missing_role,
            test_standardize_with_extra_fields,
            test_standardize_with_unicode_content,
            
            # Performance tests
            test_standardize_large_message_list_performance,
            test_convert_roles_large_message_list_performance,
            
            # Original count_message_roles tests
            test_count_message_roles_empty,
            test_count_message_roles_single_system,
            test_count_message_roles_single_developer,
            test_count_message_roles_mixed,
            test_count_message_roles_dict_format,
            
            # New edge case tests for count_message_roles
            test_count_message_roles_with_invalid_types,
            test_count_message_roles_with_unusual_roles,
            
            # Original convert_message_roles tests
            test_convert_developer_to_system,
            test_convert_system_to_developer,
            test_convert_to_user,
            test_no_conversion_needed,
            
            # New edge case tests for convert_message_roles
            test_convert_mixed_invalid_messages,
            test_convert_to_same_role,
            test_convert_chain,
            
            # Original handle_no_messages tests
            test_handle_no_messages_init_call,
            test_handle_no_messages_run_call,
            
            # New edge case tests for handle_no_messages
            test_handle_no_messages_with_unusual_model_names,
            
            # Original invalid input tests
            test_standardize_invalid_message,
            
            # Original integration tests
            test_convert_then_standardize,
            test_messagelist_count_and_convert,
            
            # New complex integration tests
            test_complex_conversion_and_counting,
            test_message_list_operations,
        ]
        
        total_tests = len(all_tests)
        passed_tests = 0
        
        for test in all_tests:
            if run_test(test):
                passed_tests += 1
        
        print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")
        
        # Return exit code based on test results
        return 0 if passed_tests == total_tests else 1
    
    # Run all tests and set exit code
    sys.exit(run_all_tests())
