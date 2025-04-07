# -------------------------------------------------------------------------------- #
# Test Authentication Mechanism
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
# Built-in imports
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add src directory to path if needed
sys.path.insert(0, 'src')

# Import the DeepSeek client
from astral_ai.providers.deepseek._client import DeepSeekProviderClient
from astral_ai._auth import AUTH_CONFIG, DEFAULT_AUTH_CONFIG

# -------------------------------------------------------------------------------- #
# Test Authentication
# -------------------------------------------------------------------------------- #

def main():
    """Test the authentication mechanism."""
    print("Testing DeepSeek client authentication...")
    
    # Print auth config for reference
    print("\nAuth configuration for DeepSeek:")
    print(f"Provider-specific: {AUTH_CONFIG.get('deepseek', {})}")
    print(f"Default for api_key_with_base_url: {DEFAULT_AUTH_CONFIG.get('api_key_with_base_url', {})}")
    
    # Set environment variables for testing
    os.environ['DEEPSEEK_API_KEY'] = 'test_api_key'
    os.environ['DEEPSEEK_BASE_URL'] = 'https://api.deepseek.com'

    
    
    # Create client and inspect auth strategies
    client = DeepSeekProviderClient()
    
    # Print available auth strategies
    print(f"\nAvailable auth strategies: {list(client._auth_strategies.keys())}")
    
    # Print which auth method was used
    print(f"Client is authenticated: {client.client is not None}")

if __name__ == "__main__":
    main() 