# -------------------------------------------------------------------------------- #
# Test Module for Astral AI Messaging Utilities
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
from typing import List, Tuple, Optional
from unittest.mock import patch, MagicMock

# Module imports
from astral_ai.messaging._utils import (
    handle_no_messages,
    standardize_messages,
    count_message_roles,
    convert_message_roles
)
from astral_ai.messaging._models import (
    Message,
    MessageList,
    TextMessage,
    ImageMessage
)
from astral_ai.exceptions import MessagesNotProvidedError, InvalidMessageError
from astral_ai.constants._models import ModelName

# -------------------------------------------------------------------------------- #
# Test Constants
# -------------------------------------------------------------------------------- #
TEST_MODEL_NAME = ModelName.CLAUDE_HAIKU

# -------------------------------------------------------------------------------- #
# handle_no_messages Tests
# -------------------------------------------------------------------------------- #

class TestHandleNoMessages:
    """Test suite for handle_no_messages function"""
    
    def test_handle_no_messages_init_call(self) -> None:
        """Test handling no messages during initialization"""
        # When
        result = handle_no_messages(TEST_MODEL_NAME, init_call=True)
        
        # Then
        assert result is None
    
    def test_handle_no_messages_not_init_call(self) -> None:
        """Test handling no messages not during initialization (should raise error)"""
        # When/Then
        with pytest.raises(MessagesNotProvidedError) as exc_info:
            handle_no_messages(TEST_MODEL_NAME, init_call=False)
            
        # Then
        assert str(TEST_MODEL_NAME) in str(exc_info.value)
        assert "Messages must be provided" in str(exc_info.value)

# -------------------------------------------------------------------------------- #
# standardize_messages Tests
# -------------------------------------------------------------------------------- #

class TestStandardizeMessages:
    """Test suite for standardize_messages function"""
    
    def test_standardize_single_message(self) -> None:
        """Test standardizing a single Message object"""
        # Given
        message = TextMessage(text="Hello")
        
        # When
        result = standardize_messages(message)
        
        # Then
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == message
    
    def test_standardize_message_list(self) -> None:
        """Test standardizing a MessageList object"""
        # Given
        messages = [
            TextMessage(text="Hello"),
            ImageMessage(image_url="https://example.com/image.jpg")
        ]
        message_list = MessageList(messages=messages)
        
        # When
        result = standardize_messages(message_list)
        
        # Then
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == messages
    
    def test_standardize_message_list_direct(self) -> None:
        """Test standardizing a direct list of Message objects"""
        # Given
        messages = [
            TextMessage(text="Hello"),
            ImageMessage(image_url="https://example.com/image.jpg")
        ]
        
        # When
        result = standardize_messages(messages)
        
        # Then
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == messages
    
    def test_standardize_invalid_message_type(self) -> None:
        """Test standardizing an invalid message type raises error"""
        # Given
        invalid_message = "not a message"
        
        # When/Then
        with pytest.raises(InvalidMessageError) as exc_info:
            standardize_messages(invalid_message)
            
        # Then
        assert "str" in str(exc_info.value)

# -------------------------------------------------------------------------------- #
# count_message_roles Tests
# -------------------------------------------------------------------------------- #

class TestCountMessageRoles:
    """Test suite for count_message_roles function"""
    
    def test_count_no_special_roles(self) -> None:
        """Test counting roles when there are no system or developer messages"""
        # Given
        messages = [
            TextMessage(role="user", text="Hello"),
            TextMessage(role="user", text="How are you?")
        ]
        
        # When
        system_count, developer_count = count_message_roles(messages)
        
        # Then
        assert system_count == 0
        assert developer_count == 0
    
    def test_count_system_roles(self) -> None:
        """Test counting roles when there are system messages"""
        # Given
        messages = [
            TextMessage(role="system", text="You are a helpful assistant"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="system", text="Be concise")
        ]
        
        # When
        system_count, developer_count = count_message_roles(messages)
        
        # Then
        assert system_count == 2
        assert developer_count == 0
    
    def test_count_developer_roles(self) -> None:
        """Test counting roles when there are developer messages"""
        # Given
        messages = [
            TextMessage(role="user", text="Hello"),
            TextMessage(role="developer", text="Debug info"),
            TextMessage(role="developer", text="More debug info")
        ]
        
        # When
        system_count, developer_count = count_message_roles(messages)
        
        # Then
        assert system_count == 0
        assert developer_count == 2
    
    def test_count_mixed_roles(self) -> None:
        """Test counting roles when there are mixed roles"""
        # Given
        messages = [
            TextMessage(role="system", text="You are a helpful assistant"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="developer", text="Debug info"),
            TextMessage(role="user", text="How are you?"),
            TextMessage(role="system", text="Be concise"),
            TextMessage(role="developer", text="More debug info")
        ]
        
        # When
        system_count, developer_count = count_message_roles(messages)
        
        # Then
        assert system_count == 2
        assert developer_count == 2
    
    def test_count_with_empty_list(self) -> None:
        """Test counting roles with an empty list"""
        # Given
        messages: List[Message] = []
        
        # When
        system_count, developer_count = count_message_roles(messages)
        
        # Then
        assert system_count == 0
        assert developer_count == 0

# -------------------------------------------------------------------------------- #
# convert_message_roles Tests
# -------------------------------------------------------------------------------- #

class TestConvertMessageRoles:
    """Test suite for convert_message_roles function"""
    
    @patch('astral_ai.messaging._utils.logger')
    def test_convert_developer_to_system(self, mock_logger: MagicMock) -> None:
        """Test converting developer messages to system messages"""
        # Given
        messages = [
            TextMessage(role="user", text="Hello"),
            TextMessage(role="developer", text="Debug info"),
            TextMessage(role="user", text="How are you?")
        ]
        
        # When
        convert_message_roles(messages, target_role="system", model_name=TEST_MODEL_NAME)
        
        # Then
        assert messages[0].role == "user"
        assert messages[1].role == "system"
        assert messages[2].role == "user"
        mock_logger.warning.assert_called_once()
    
    @patch('astral_ai.messaging._utils.logger')
    def test_convert_system_to_developer(self, mock_logger: MagicMock) -> None:
        """Test converting system messages to developer messages"""
        # Given
        messages = [
            TextMessage(role="system", text="You are a helpful assistant"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="system", text="Be concise")
        ]
        
        # When
        convert_message_roles(messages, target_role="developer", model_name=TEST_MODEL_NAME)
        
        # Then
        assert messages[0].role == "developer"
        assert messages[1].role == "user"
        assert messages[2].role == "developer"
        assert mock_logger.warning.call_count == 2
    
    @patch('astral_ai.messaging._utils.logger')
    def test_convert_special_roles_to_user(self, mock_logger: MagicMock) -> None:
        """Test converting system and developer messages to user messages"""
        # Given
        messages = [
            TextMessage(role="system", text="You are a helpful assistant"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="developer", text="Debug info")
        ]
        
        # When
        convert_message_roles(messages, target_role="user", model_name=TEST_MODEL_NAME)
        
        # Then
        assert all(msg.role == "user" for msg in messages)
        assert mock_logger.warning.call_count == 2
    
    @patch('astral_ai.messaging._utils.logger')
    def test_no_conversion_needed(self, mock_logger: MagicMock) -> None:
        """Test when no conversion is needed"""
        # Given
        messages = [
            TextMessage(role="user", text="Hello"),
            TextMessage(role="user", text="How are you?")
        ]
        original_messages = messages.copy()
        
        # When
        convert_message_roles(messages, target_role="user", model_name=TEST_MODEL_NAME)
        
        # Then
        assert messages == original_messages
        mock_logger.warning.assert_not_called()

# -------------------------------------------------------------------------------- #
# Integration Tests
# -------------------------------------------------------------------------------- #

class TestMessagingUtilsIntegration:
    """Integration tests for messaging utils"""
    
    def test_standardize_and_count(self) -> None:
        """Test standardizing messages and then counting roles"""
        # Given
        message = TextMessage(role="system", text="You are a helpful assistant")
        
        # When
        standardized = standardize_messages(message)
        system_count, developer_count = count_message_roles(standardized)
        
        # Then
        assert system_count == 1
        assert developer_count == 0
    
    @patch('astral_ai.messaging._utils.logger')
    def test_standardize_and_convert(self, mock_logger: MagicMock) -> None:
        """Test standardizing messages and then converting roles"""
        # Given
        message_list = MessageList(messages=[
            TextMessage(role="system", text="You are a helpful assistant"),
            TextMessage(role="user", text="Hello")
        ])
        
        # When
        standardized = standardize_messages(message_list)
        convert_message_roles(standardized, target_role="user", model_name=TEST_MODEL_NAME)
        
        # Then
        assert all(msg.role == "user" for msg in standardized)
        mock_logger.warning.assert_called_once()
