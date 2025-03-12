from __future__ import annotations

# -------------------------------------------------------------------------------- #
# Completions Resource
# -------------------------------------------------------------------------------- #

"""

Astral AI Completions Resource

Handles both chat and structured completion requests by providing:
- Type-safe request handling
- Provider-specific request/response adaptation
- Cost calculation and tracking
- Response validation and parsing

"""
# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #

# Built-in
from typing import Dict, List, Optional, Union, Iterable, Literal, overload, TypeVar, Type
from abc import ABC

# Pydantic
from pydantic import BaseModel

# HTTPX Timeout
from httpx import Timeout

# Astral AI Types
from astral_ai._types import (
    # Base
    NotGiven,
    NOT_GIVEN,

    # Request
    AstralCompletionRequest,
    AstralStructuredCompletionRequest,

    # Response
    Metadata,

    # Request Params
    Modality,
    ResponsePrediction,
    ReasoningEffort,
    ResponseFormat,
    StreamOptions,
    ToolChoice,
    Tool,
    AstralParams,
    AstralChatResponse,
    AstralStructuredResponse,
)

# Astral AI Exceptions
from astral_ai.exceptions import ResponseModelMissingError

# Astral AI Decorators
from astral_ai._decorators import required_parameters

# Astral AI Messaging Models
from astral_ai.messaging._models import Messages

# Astral AI Resources
from astral_ai.resources._base_resource import AstralResource

# -------------------------------------------------------------------------------- #
# Generic Types
# -------------------------------------------------------------------------------- #

_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# -------------------------------------------------------------------------------- #
# Completions Resource Class
# -------------------------------------------------------------------------------- #


class Completions(AstralResource):
    """
    Astral AI Completions Resource.

    Handles both chat and structured completion requests by providing:
    - Type-safe request handling
    - Provider-specific request/response adaptation
    - Cost calculation and tracking
    - Response validation and parsing
    """

    def __init__(
        self,
        request: AstralCompletionRequest,
    ) -> None:
        super().__init__(request)

    # -------------------------------------------------------------------------------- #
    # Run Method Overloads & Implementation
    # -------------------------------------------------------------------------------- #

    @overload
    def run(self) -> AstralChatResponse:
        ...

    @overload
    def run(self, response_format: Type[_StructuredOutputT]) -> AstralStructuredResponse[_StructuredOutputT]:
        ...

    def run(
        self, response_format: Optional[Type[_StructuredOutputT]] = None,
    ) -> Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]:
        """
        Execute the completion request.

        This method handles both chat and structured completion requests by:
        1. Converting the Astral request to provider-specific format
        2. Executing the request with the appropriate provider client
        3. Converting and validating the provider response
        4. Calculating and attaching cost metrics if enabled

        Args:
            response_format: Optional structured output model for parsing responses.
                           If provided, the response will be parsed into this model type.

        Returns:
            AstralChatResponse: For standard chat completions when response_format is None
            AstralStructuredResponse: For structured outputs when response_format is provided

        Note:
            Cost calculation is only performed if a cost_strategy is configured.
            The cost metrics will be attached to the response object.
        """
        # Convert the request to provider-specific format
        provider_request = self.adapter.to_provider_request(self.request)

        # Execute request and convert response based on type
        if response_format is None:
            provider_response = self.client.create_completion_chat(provider_request)
            astral_response = self.adapter.to_astral_completion_response(provider_response)
        else:
            provider_response = self.client.create_completion_structured(provider_request)
            astral_response = self.adapter.to_astral_completion_response(
                provider_response,
                response_model=response_format
            )

        # Apply cost strategy if configured
        astral_response = self._apply_cost(astral_response)

        return astral_response
    
    def _apply_cost(self, response: Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]) -> Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]:
        """
        Apply cost calculation to any response type if a cost strategy is configured.
        
        The method relies on the overloaded signatures in BaseCostStrategy to handle
        the appropriate response types correctly.
        
        Args:
            response: The response object (either chat or structured)
            
        Returns:
            The same response object with cost information attached if a cost strategy is configured
        """
        if self.cost_strategy is not None:
            return self.cost_strategy.run_cost_strategy(
                response=response,
                model_name=self.model,
                model_provider=self.model_provider,
            )
        return response

    @overload
    async def run_async(self) -> AstralChatResponse:
        ...

    @overload
    async def run_async(self, response_format: Type[_StructuredOutputT]) -> AstralStructuredResponse[_StructuredOutputT]:
        ...

    async def run_async(
        self, response_format: Optional[Type[_StructuredOutputT]] = None,
    ) -> Union[AstralChatResponse, AstralStructuredResponse[_StructuredOutputT]]:
        """
        Execute the completion request asynchronously.
        """
        pass


# -------------------------------------------------------------------------------- #
# Top-level Functions
# -------------------------------------------------------------------------------- #


@required_parameters("model", "messages")
def completion(
    *,
    model: str,
    messages: Messages | List[Dict[str, str]],
    astral_params: Optional[AstralParams] | None = None,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
    modalities: Optional[List[Modality]] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
    prediction: Optional[ResponsePrediction] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    response_format: ResponseFormat | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    service_tier: Literal["auto", "default"] | NotGiven = NOT_GIVEN,
    stop: Optional[str] | List[str] | NotGiven = NOT_GIVEN,
    store: Optional[bool] | NotGiven = NOT_GIVEN,
    stream_options: Optional[StreamOptions] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[Tool] | NotGiven = NOT_GIVEN,
    top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    timeout: Union[float, Timeout, None] | NotGiven = NOT_GIVEN,
) -> AstralChatResponse:
    """
    Top-level function for a chat completion request.

    Args:
        model: The model to use for completion
        messages: The conversation history
        astral_params: Optional Astral-specific parameters
        **kwargs: Additional model-specific parameters

    Returns:
        AstralChatResponse: The chat completion response
    """
    request_data = {
        "model": model,
        "messages": messages,
        "astral_params": astral_params,
        "stream": False,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": store,
        "stream_options": stream_options,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "user": user,
        "timeout": timeout,
    }
    request = AstralCompletionRequest(**request_data)
    comp = Completions(request)
    return comp.run()

# -------------------------------------------------------------------------------- #
# Structured Completion
# -------------------------------------------------------------------------------- #


StructuredOutputResponseT = TypeVar('StructuredOutputResponseT', bound=BaseModel)


@required_parameters("model", "messages", "response_model")
def completion_structured(
    *,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Type[StructuredOutputResponseT],
    astral_params: Optional[AstralParams] | None = None,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    logprobs: Optional[bool] | NotGiven = NOT_GIVEN,
    max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    metadata: Optional[Metadata] | NotGiven = NOT_GIVEN,
    modalities: Optional[List[Modality]] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
    prediction: Optional[ResponsePrediction] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    service_tier: Literal["auto", "default"] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    store: Optional[bool] | NotGiven = NOT_GIVEN,
    stream_options: Optional[StreamOptions] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
    tools: Iterable[Tool] | NotGiven = NOT_GIVEN,
    top_logprobs: Optional[int] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    timeout: Union[float, Timeout, None] | NotGiven = NOT_GIVEN,
) -> AstralStructuredResponse[StructuredOutputResponseT]:
    """
    Top-level function for a structured completion request.

    Args:
        model: The model to use for completion
        messages: The conversation history
        response_format: The Pydantic model to parse the response into
        astral_params: Optional Astral-specific parameters
        **kwargs: Additional model-specific parameters

    Returns:
        AstralStructuredResponse: The structured response, with its inner `response`
        field parsed using the provided `response_model`

    Raises:
        ResponseModelMissingError: If response_format is None
    """

    if response_format is None:
        raise ResponseModelMissingError(model_name=model)

    # Mark the request as structured.
    request_data = {
        "model": model,
        "response_format": response_format,
        "messages": messages,
        "stream": False,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "metadata": metadata,
        "modalities": modalities,
        "n": n,
        "parallel_tool_calls": parallel_tool_calls,
        "prediction": prediction,
        "presence_penalty": presence_penalty,
        "reasoning_effort": reasoning_effort,
        "response_format": response_format,
        "seed": seed,
        "service_tier": service_tier,
        "stop": stop,
        "store": store,
        "stream_options": stream_options,
        "temperature": temperature,
        "tool_choice": tool_choice,
        "tools": tools,
        "top_logprobs": top_logprobs,
        "top_p": top_p,
        "user": user,
        "timeout": timeout,
        # Additional flag to signal a structured response.
        "structured": True,
    }

    request = AstralStructuredCompletionRequest(**request_data)
    comp = Completions(request, astral_params=astral_params)
    return comp.run(response_format=response_format)

# -------------------------------------------------------------------------------- #

# # -------------------------------------------------------------------------------- #
# # Testing and Benchmarking
# # -------------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     import time
#     import statistics
#     from tabulate import tabulate
#     from astral_ai.messaging._models import MessageList
#     from typing import Tuple
#     from openai import OpenAI
#     from openai.types.chat import ChatCompletion

#     def call_completion(model: str, messages: MessageList) -> Tuple[AstralChatResponse, float]:
#         start_time = time.time()
#         comp = completion(model=model, messages=messages)
#         end_time = time.time()
#         latency = end_time - start_time
#         return comp, latency

#     def call_openai_directly(model: str, messages: list) -> Tuple[ChatCompletion, float]:
#         import openai
#         client = openai.OpenAI()
        
#         start_time = time.time()
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             stream=False,
#         )
#         end_time = time.time()
#         latency = end_time - start_time
#         return response, latency

#     # Test messages
#     messages_1 = [
#         {"role": "developer", "content": "You are an expert software engineer."},
#         {"role": "user", "content": "Write a function to print 'Hello, world!'"},
#     ]

#     messages_2 = [
#         {"role": "user", "content": "Just say 'Hello, world!'"},
#     ]

#     messages_3 = [
#         {"role": "user", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of the moon?"},
#     ]

#     # Convert to MessageList and back to dict format
#     messages_1 = MessageList(messages=messages_1).model_dump()["messages"]
#     messages_2 = MessageList(messages=messages_2).model_dump()["messages"]
#     messages_3 = MessageList(messages=messages_3).model_dump()["messages"]

#     # Prepare test cases
#     test_cases = [
#         {"name": "Complex Query", "messages": messages_1},
#         {"name": "Simple Query", "messages": messages_2},
#         {"name": "Creative Query", "messages": messages_3},
#     ]

#     # Results storage
#     results = []
    
#     # Number of iterations
#     iterations = 3
    
#     # Track total cost
#     total_cost = 0.0
    
#     print(f"Running benchmark with {iterations} iterations per test case...")
    
#     # Run benchmarks
#     for test_case in test_cases:
#         name = test_case["name"]
#         messages = test_case["messages"]
        
#         # Astral API calls
#         astral_latencies = []
#         astral_costs = []
        
#         print(f"\nRunning {name} through Astral API...")
#         for i in range(iterations):
#             print(f"  Iteration {i+1}/{iterations}", end="\r")
#             comp, latency = call_completion(model="gpt-4o", messages=messages)

#             astral_latencies.append(latency)
#             if comp.cost:
#                 cost_value = comp.cost.total_cost or 0.0
#                 astral_costs.append(cost_value)
#                 total_cost += cost_value
        
#         # OpenAI direct calls
#         openai_latencies = []
        
#         print(f"\nRunning {name} through direct OpenAI API...")
#         for i in range(iterations):
#             print(f"  Iteration {i+1}/{iterations}", end="\r")
#             _, latency = call_openai_directly(model="gpt-4o", messages=messages)
#             openai_latencies.append(latency)
        
#         # Calculate statistics
#         results.append({
#             "Test Case": name,
#             "Astral Avg Latency": f"{statistics.mean(astral_latencies):.3f}s",
#             "Astral Min Latency": f"{min(astral_latencies):.3f}s",
#             "Astral Max Latency": f"{max(astral_latencies):.3f}s",
#             "Astral Avg Cost": f"${statistics.mean(astral_costs):.6f}" if astral_costs else "N/A",
#             "OpenAI Avg Latency": f"{statistics.mean(openai_latencies):.3f}s",
#             "OpenAI Min Latency": f"{min(openai_latencies):.3f}s",
#             "OpenAI Max Latency": f"{max(openai_latencies):.3f}s",
#             "Latency Diff": f"{(statistics.mean(astral_latencies) - statistics.mean(openai_latencies)):.3f}s"
#         })
    
#     # Display results
#     print("\n\n# -------------------------------------------------------------------------------- #")
#     print("# Benchmark Results")
#     print("# -------------------------------------------------------------------------------- #\n")
    
#     print(tabulate(results, headers="keys", tablefmt="grid"))
    
#     # Summary
#     avg_astral_latency = statistics.mean([float(r["Astral Avg Latency"].replace("s", "")) for r in results])
#     avg_openai_latency = statistics.mean([float(r["OpenAI Avg Latency"].replace("s", "")) for r in results])
    
#     print(f"\nOverall Astral Average Latency: {avg_astral_latency:.3f}s")
#     print(f"Overall OpenAI Average Latency: {avg_openai_latency:.3f}s")
#     print(f"Overall Latency Difference: {avg_astral_latency - avg_openai_latency:.3f}s")
#     print(f"Total Astral Cost: ${total_cost:.6f}")


# -------------------------------------------------------------------------------- #
# Simple Test Function
# -------------------------------------------------------------------------------- #

def run_simple_test():
    """
    Run a simple test with the Astral API and print the results.
    
    This function demonstrates a basic completion request and displays:
    - Response content
    - Latency
    - Token usage
    - Cost information
    """
    import time
    from typing import List, Dict, Any, Tuple
    
    # Sample messages for testing
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]

    # Print the messages
    print(f"Messages: {messages}")
    
    print("\n# -------------------------------------------------------------------------------- #")
    print("# Running Simple Astral API Test")
    print("# -------------------------------------------------------------------------------- #\n")
    
    # Measure latency
    start_time = time.time()
    
    # Make the completion request
    response = completion(
        model="gpt-4o",
        messages=messages
    )


    # Calculate latency
    latency = time.time() - start_time
    
    # Print results
    print(f"Response:\n{'-' * 40}")
    print(response.response)
    print(f"\n{'-' * 40}")
    
    print(f"\nLatency: {latency:.3f}s")
    
    # Print usage information
    if response.usage:
        print("\nToken Usage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
    
    # Print cost information
    if response.cost:
        print("\nCost Information:")
        print(f"  Input cost: ${response.cost.input_cost:.6f}")
        print(f"  Output cost: ${response.cost.output_cost:.6f}")
        print(f"  Total cost: ${response.cost.total_cost:.6f}")
    
    return response

if __name__ == "__main__":
    run_simple_test()
