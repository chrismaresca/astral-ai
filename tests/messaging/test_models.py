# -------------------------------------------------------------------------------- #
# Test Module for Astral AI Messaging Models
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import pytest
import sys
import time
from typing import List, Dict, Any

# Pydantic imports
from pydantic import ValidationError

# Module imports
from astral_ai.messaging._models import (
    Message,
    MessageList,
    TextMessage, 
    ImageMessage,
    AudioMessage,
    MessageRole,
    ImageDetail
)

# -------------------------------------------------------------------------------- #
# Utility Functions for Verbose Testing
# -------------------------------------------------------------------------------- #

def print_separator() -> None:
    """Print a horizontal separator line."""
    print("\n" + "=" * 80)

def print_test_header(test_name: str) -> None:
    """Print a formatted test header."""
    print_separator()
    print(f"STARTING TEST: {test_name}")
    print_separator()

def print_test_footer(test_name: str) -> None:
    """Print a formatted test footer."""
    print_separator()
    print(f"TEST COMPLETED SUCCESSFULLY: {test_name}")
    print_separator()

def print_section(section_name: str) -> None:
    """Print a section header within a test."""
    print("\n" + "-" * 40)
    print(f"SECTION: {section_name}")
    print("-" * 40)

def print_object_details(obj: Any, name: str) -> None:
    """Print detailed information about an object."""
    print(f"\nDETAILS OF {name}:")
    print(f"  Type: {type(obj)}")
    print(f"  Dir: {dir(obj)}")
    if hasattr(obj, 'model_dump'):
        print(f"  Model Dump: {obj.model_dump()}")
    else:
        print(f"  Repr: {repr(obj)}")

# -------------------------------------------------------------------------------- #
# TextMessage Tests
# -------------------------------------------------------------------------------- #

class TestTextMessage:
    """Test suite for TextMessage class"""
    
    def test_create_text_message_with_defaults(self) -> None:
        """Test creating a TextMessage with default values"""
        print_test_header("TextMessage - Create with Default Values")
        
        print("\nSTEP: Creating TextMessage with text='Hello, world!'")
        start_time = time.time()
        message = TextMessage(text="Hello, world!")
        creation_time = time.time() - start_time
        print(f"Message created in {creation_time:.6f} seconds")
        
        print_object_details(message, "TextMessage")
        
        print("\nSTEP: Verifying default attributes")
        print(f"Expected role: 'user', Actual: '{message.role}'")
        assert message.role == "user"
        
        print(f"Expected text: 'Hello, world!', Actual: '{message.text}'")
        assert message.text == "Hello, world!"
        
        print_test_footer("TextMessage - Create with Default Values")
        
    def test_create_text_message_with_custom_role(self) -> None:
        """Test creating a TextMessage with custom role"""
        print_test_header("TextMessage - Create with Custom Role")
        
        print("\nSTEP: Defining test roles")
        roles: List[MessageRole] = ["system", "user", "developer"]
        print(f"Test roles: {roles}")
        
        for role in roles:
            print(f"\nSTEP: Testing with role '{role}'")
            start_time = time.time()
            message = TextMessage(role=role, text="Hello, world!")
            creation_time = time.time() - start_time
            print(f"Message created in {creation_time:.6f} seconds")
            
            print_object_details(message, f"TextMessage with role={role}")
            
            print(f"Verifying role: Expected '{role}', Actual: '{message.role}'")
            assert message.role == role
            
            print(f"Verifying text: Expected 'Hello, world!', Actual: '{message.text}'")
            assert message.text == "Hello, world!"
        
        print_test_footer("TextMessage - Create with Custom Role")
    
    def test_text_message_validation_error(self) -> None:
        """Test validation errors in TextMessage"""
        print_test_header("TextMessage - Validation Errors")
        
        print("\nSTEP: Testing missing required field")
        print("Attempting to create TextMessage without 'text' field")
        with pytest.raises(ValidationError) as excinfo:
            TextMessage()
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print("\nSTEP: Testing invalid role")
        print("Attempting to create TextMessage with invalid role 'invalid_role'")
        with pytest.raises(ValidationError) as excinfo:
            TextMessage(role="invalid_role", text="Hello")
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print_test_footer("TextMessage - Validation Errors")

    def test_text_message_dict_conversion(self) -> None:
        """Test conversion to and from dict"""
        print_test_header("TextMessage - Dict Conversion")
        
        print("\nSTEP: Creating source dictionary")
        message_dict = {"role": "system", "text": "System message"}
        print(f"Source dict: {message_dict}")
        
        print("\nSTEP: Converting dict to TextMessage")
        start_time = time.time()
        message = TextMessage.model_validate(message_dict)
        validate_time = time.time() - start_time
        print(f"Validation completed in {validate_time:.6f} seconds")
        
        print_object_details(message, "Converted TextMessage")
        
        print("\nSTEP: Converting TextMessage back to dict")
        start_time = time.time()
        result_dict = message.model_dump()
        dump_time = time.time() - start_time
        print(f"Dump completed in {dump_time:.6f} seconds")
        
        print(f"Result dict: {result_dict}")
        
        print("\nSTEP: Verifying dictionary values")
        print(f"Expected role: 'system', Actual: '{result_dict['role']}'")
        assert result_dict["role"] == "system"
        
        print(f"Expected text: 'System message', Actual: '{result_dict['text']}'")
        assert result_dict["text"] == "System message"
        
        print_test_footer("TextMessage - Dict Conversion")

# -------------------------------------------------------------------------------- #
# ImageMessage Tests
# -------------------------------------------------------------------------------- #

class TestImageMessage:
    """Test suite for ImageMessage class"""
    
    def test_create_image_message_with_defaults(self) -> None:
        """Test creating an ImageMessage with default values"""
        print_test_header("ImageMessage - Create with Default Values")
        
        print("\nSTEP: Creating ImageMessage with default values")
        start_time = time.time()
        message = ImageMessage(image_url="https://example.com/image.jpg")
        creation_time = time.time() - start_time
        print(f"Message created in {creation_time:.6f} seconds")
        
        print_object_details(message, "ImageMessage")
        
        print("\nSTEP: Verifying default attributes")
        print(f"Expected role: 'user', Actual: '{message.role}'")
        assert message.role == "user"
        
        print(f"Expected image_url: 'https://example.com/image.jpg', Actual: '{message.image_url}'")
        assert message.image_url == "https://example.com/image.jpg"
        
        print(f"Expected image_detail: 'auto', Actual: '{message.image_detail}'")
        assert message.image_detail == "auto"
        
        print_test_footer("ImageMessage - Create with Default Values")
        
    def test_create_image_message_with_custom_values(self) -> None:
        """Test creating an ImageMessage with custom values"""
        print_test_header("ImageMessage - Create with Custom Values")
        
        print("\nSTEP: Defining test roles and details")
        roles: List[MessageRole] = ["system", "user", "developer"]
        details: List[ImageDetail] = ["high", "low", "auto"]
        
        print(f"Test roles: {roles}")
        print(f"Test details: {details}")
        
        for role in roles:
            for detail in details:
                print(f"\nSTEP: Testing with role='{role}', detail='{detail}'")
                
                start_time = time.time()
                message = ImageMessage(
                    role=role,
                    image_url="https://example.com/image.jpg",
                    image_detail=detail
                )
                creation_time = time.time() - start_time
                print(f"Message created in {creation_time:.6f} seconds")
                
                print_object_details(message, f"ImageMessage with role={role}, detail={detail}")
                
                print(f"Verifying role: Expected '{role}', Actual: '{message.role}'")
                assert message.role == role
                
                print(f"Verifying image_url: Expected 'https://example.com/image.jpg', Actual: '{message.image_url}'")
                assert message.image_url == "https://example.com/image.jpg"
                
                print(f"Verifying image_detail: Expected '{detail}', Actual: '{message.image_detail}'")
                assert message.image_detail == detail
        
        print_test_footer("ImageMessage - Create with Custom Values")
    
    def test_image_message_validation_error(self) -> None:
        """Test validation errors in ImageMessage"""
        print_test_header("ImageMessage - Validation Errors")
        
        print("\nSTEP: Testing missing required field")
        print("Attempting to create ImageMessage without 'image_url' field")
        with pytest.raises(ValidationError) as excinfo:
            ImageMessage()
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print("\nSTEP: Testing invalid role")
        print("Attempting to create ImageMessage with invalid role 'invalid_role'")
        with pytest.raises(ValidationError) as excinfo:
            ImageMessage(role="invalid_role", image_url="https://example.com/image.jpg")
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print("\nSTEP: Testing invalid image_detail")
        print("Attempting to create ImageMessage with invalid image_detail 'invalid_detail'")
        with pytest.raises(ValidationError) as excinfo:
            ImageMessage(
                image_url="https://example.com/image.jpg",
                image_detail="invalid_detail"
            )
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print_test_footer("ImageMessage - Validation Errors")

    def test_image_message_dict_conversion(self) -> None:
        """Test conversion to and from dict"""
        print_test_header("ImageMessage - Dict Conversion")
        
        print("\nSTEP: Creating source dictionary")
        message_dict = {
            "role": "system", 
            "image_url": "https://example.com/image.jpg",
            "image_detail": "high"
        }
        print(f"Source dict: {message_dict}")
        
        print("\nSTEP: Converting dict to ImageMessage")
        start_time = time.time()
        message = ImageMessage.model_validate(message_dict)
        validate_time = time.time() - start_time
        print(f"Validation completed in {validate_time:.6f} seconds")
        
        print_object_details(message, "Converted ImageMessage")
        
        print("\nSTEP: Converting ImageMessage back to dict")
        start_time = time.time()
        result_dict = message.model_dump()
        dump_time = time.time() - start_time
        print(f"Dump completed in {dump_time:.6f} seconds")
        
        print(f"Result dict: {result_dict}")
        
        print("\nSTEP: Verifying dictionary values")
        print(f"Expected role: 'system', Actual: '{result_dict['role']}'")
        assert result_dict["role"] == "system"
        
        print(f"Expected image_url: 'https://example.com/image.jpg', Actual: '{result_dict['image_url']}'")
        assert result_dict["image_url"] == "https://example.com/image.jpg"
        
        print(f"Expected image_detail: 'high', Actual: '{result_dict['image_detail']}'")
        assert result_dict["image_detail"] == "high"
        
        print_test_footer("ImageMessage - Dict Conversion")

# -------------------------------------------------------------------------------- #
# AudioMessage Tests
# -------------------------------------------------------------------------------- #

class TestAudioMessage:
    """Test suite for AudioMessage class"""
    
    def test_audio_message_creation(self) -> None:
        """Test creating an AudioMessage"""
        print_test_header("AudioMessage - Creation")
        
        print("\nSTEP: Creating a basic AudioMessage instance")
        print("Note: This is a placeholder test since AudioMessage isn't fully implemented")
        
        start_time = time.time()
        message = AudioMessage()
        creation_time = time.time() - start_time
        print(f"Message created in {creation_time:.6f} seconds")
        
        print_object_details(message, "AudioMessage")
        
        print("\nSTEP: Verifying instance type")
        print(f"Expected type: AudioMessage, Actual type: {type(message).__name__}")
        assert isinstance(message, AudioMessage)
        
        print_test_footer("AudioMessage - Creation")

# -------------------------------------------------------------------------------- #
# MessageList Tests
# -------------------------------------------------------------------------------- #

class TestMessageList:
    """Test suite for MessageList class"""
    
    def test_create_message_list(self) -> None:
        """Test creating a MessageList"""
        print_test_header("MessageList - Creation")
        
        print("\nSTEP: Creating message instances for the list")
        text_message = TextMessage(text="Hello")
        image_message = ImageMessage(image_url="https://example.com/image.jpg")
        
        messages = [text_message, image_message]
        print(f"Created {len(messages)} messages for testing")
        
        print("\nSTEP: Creating MessageList with messages")
        start_time = time.time()
        message_list = MessageList(messages=messages)
        creation_time = time.time() - start_time
        print(f"MessageList created in {creation_time:.6f} seconds")
        
        print_object_details(message_list, "MessageList")
        
        print("\nSTEP: Verifying list contents")
        print(f"Expected length: 2, Actual length: {len(message_list.messages)}")
        assert len(message_list.messages) == 2
        
        print(f"Expected type of first item: TextMessage, Actual: {type(message_list.messages[0]).__name__}")
        assert isinstance(message_list.messages[0], TextMessage)
        
        print(f"Expected type of second item: ImageMessage, Actual: {type(message_list.messages[1]).__name__}")
        assert isinstance(message_list.messages[1], ImageMessage)
        
        print_test_footer("MessageList - Creation")
    
    def test_message_list_iteration(self) -> None:
        """Test iterating through a MessageList"""
        print_test_header("MessageList - Iteration")
        
        print("\nSTEP: Creating messages and message list")
        messages = [
            TextMessage(text="Hello"),
            ImageMessage(image_url="https://example.com/image.jpg")
        ]
        message_list = MessageList(messages=messages)
        
        print("\nSTEP: Iterating through MessageList")
        print("Converting iterator to list to verify iteration works")
        start_time = time.time()
        result_messages = list(message_list)
        iteration_time = time.time() - start_time
        print(f"Iteration completed in {iteration_time:.6f} seconds")
        
        print(f"Iteration result: {result_messages}")
        
        print("\nSTEP: Verifying iteration results")
        print(f"Expected length: {len(messages)}, Actual length: {len(result_messages)}")
        assert len(result_messages) == 2
        
        print("Verifying that iteration returns original messages")
        assert result_messages == messages
        
        print_test_footer("MessageList - Iteration")
    
    def test_message_list_getitem(self) -> None:
        """Test accessing items in a MessageList using indexing"""
        print_test_header("MessageList - Indexing")
        
        print("\nSTEP: Creating messages and message list")
        messages = [
            TextMessage(text="Hello"),
            ImageMessage(image_url="https://example.com/image.jpg")
        ]
        message_list = MessageList(messages=messages)
        
        print("\nSTEP: Accessing items by index")
        print("Getting item at index 0")
        item0 = message_list[0]
        print(f"Item at index 0: {item0}")
        
        print("Getting item at index 1")
        item1 = message_list[1]
        print(f"Item at index 1: {item1}")
        
        print("\nSTEP: Verifying indexed access")
        print(f"Expected item0: {messages[0]}, Actual: {item0}")
        assert message_list[0] == messages[0]
        
        print(f"Expected item1: {messages[1]}, Actual: {item1}")
        assert message_list[1] == messages[1]
        
        print_test_footer("MessageList - Indexing")
    
    def test_message_list_len(self) -> None:
        """Test getting the length of a MessageList"""
        print_test_header("MessageList - Length")
        
        print("\nSTEP: Creating messages and message list")
        messages = [
            TextMessage(text="Hello"),
            ImageMessage(image_url="https://example.com/image.jpg")
        ]
        message_list = MessageList(messages=messages)
        
        print("\nSTEP: Getting length of message list")
        list_length = len(message_list)
        print(f"MessageList length: {list_length}")
        
        print("\nSTEP: Verifying length")
        print(f"Expected length: 2, Actual length: {list_length}")
        assert list_length == 2
        
        print_test_footer("MessageList - Length")
        
    def test_message_list_validation_error(self) -> None:
        """Test validation errors in MessageList"""
        print_test_header("MessageList - Validation Errors")
        
        print("\nSTEP: Testing missing required field")
        print("Attempting to create MessageList without 'messages' field")
        with pytest.raises(ValidationError) as excinfo:
            MessageList()
        
        print(f"Validation error raised successfully: {excinfo.value}")
        print(f"Error details: {excinfo.value.errors()}")
        
        print_test_footer("MessageList - Validation Errors")
            
    def test_message_list_dict_conversion(self) -> None:
        """Test conversion to and from dict"""
        print_test_header("MessageList - Dict Conversion")
        
        print("\nSTEP: Creating source dictionary")
        message_list_dict = {
            "messages": [
                {"role": "user", "text": "Hello"},
                {"role": "system", "image_url": "https://example.com/image.jpg", "image_detail": "high"}
            ]
        }
        print(f"Source dict: {message_list_dict}")
        
        print("\nSTEP: Converting dict to MessageList")
        message_list = MessageList.model_validate(message_list_dict)
        print_object_details(message_list, "Converted MessageList")
        
        print("\nSTEP: Verifying conversion results")
        assert len(message_list) == 2
        assert isinstance(message_list[0], TextMessage)
        assert message_list[0].text == "Hello"
        assert isinstance(message_list[1], ImageMessage)
        assert message_list[1].image_url == "https://example.com/image.jpg"
        assert message_list[1].image_detail == "high"
        
        print("\nSTEP: Creating MessageList directly with individual message objects")
        direct_message_list = MessageList(
            messages=[
                TextMessage(role="user", text="Hello"),
                ImageMessage(role="system", image_url="https://example.com/image.jpg", image_detail="high")
            ]
        )
        
        print_object_details(direct_message_list, "Direct MessageList")
        
        print("\nSTEP: Verifying direct creation")
        print(f"Expected length: 2, Actual length: {len(direct_message_list)}")
        assert len(direct_message_list) == 2
        
        print_test_footer("MessageList - Dict Conversion")

# -------------------------------------------------------------------------------- #
# Type Alias Tests
# -------------------------------------------------------------------------------- #

def test_message_type_alias() -> None:
    """Test the Message type alias"""
    print_test_header("Message Type Alias")
    
    print("\nSTEP: Creating instances of different message types")
    text_message = TextMessage(text="Hello")
    image_message = ImageMessage(image_url="https://example.com/image.jpg")
    
    print_object_details(text_message, "TextMessage")
    print_object_details(image_message, "ImageMessage")
    
    print("\nSTEP: Verifying instances are Message type")
    print(f"Is text_message a Message? {isinstance(text_message, Message)}")
    assert isinstance(text_message, Message)
    
    print(f"Is image_message a Message? {isinstance(image_message, Message)}")
    assert isinstance(image_message, Message)
    
    print_test_footer("Message Type Alias")



