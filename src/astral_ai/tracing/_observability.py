"""
# -------------------------------------------------------------------------------- #
# Astral Observability with Non-Blocking Queue and Batched Log Delivery
# -------------------------------------------------------------------------------- #

Non-blocking observability utilities for Astral AI, incorporating:
 1) An object pool for AstralBaseLogEvent to reduce GC overhead.
 2) A queue-based approach similar to logging.handlers.QueueHandler.
 3) Daemon thread processing for zero impact on user requests.
 4) Best-effort delivery with graceful degradation.
 5) Batched HTTP delivery for improved network efficiency.
 6) Comprehensive logging and verbose documentation for clarity.

Enhancements over previous implementation:
 - Support for batching multiple log events into a single HTTP request if a batch endpoint is configured.
 - Configurable batch size and timeout to optimize throughput and reduce network overhead.
 - Improved error handling and retry logic in both batch and single-event sending.
 - More robust shutdown procedure that flushes the queue in batches.
"""

# -------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------- #
from typing import Literal
import os
import json
import time
import queue
import threading
import traceback
import atexit
import signal
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Awaitable, Concatenate, Optional, List, Dict, Any, cast, Type
import uuid
from pydantic import BaseModel

# Logging
from astral_ai.logger import logger

# httpx for async requests (using synchronous client for background thread)
import httpx

# Astral AI Types
from astral_ai.constants._models import ModelName, ModelProvider, ResourceType
from astral_ai._types import AstralBaseRequest, AstralBaseResponse, BaseUsage, BaseCost
from astral_ai._types._request._request import AstralBaseRequest

# -------------------------------------------------------------------------------- #
# Config / Constants
# -------------------------------------------------------------------------------- #
ASTRAL_API_KEY = os.getenv("ASTRAL_API_KEY", "")
OBSERVABILITY_EMOJI = "🔍"

if not ASTRAL_API_KEY:
    logger.warning("⚠️ ASTRAL_API_KEY is not set, observability will not be enabled")
    logger.warning("To enable logging, go here to create a project and API key: https://useastral.dev/dashboard/projects")

LOG_ENABLED = bool(ASTRAL_API_KEY)

# Flag to determine if we should use local endpoints
LOCAL = os.getenv("ASTRAL_LOCAL", "").lower() in ("true", "1", "yes", "y")

# Set API URLs based on LOCAL flag
if LOCAL:
    # Local fallback endpoints
    ASTRAL_API_BASE_URL = os.getenv("ASTRAL_API_BASE_URL", "http://localhost:4000/dev/")
    ASTRAL_API_KEY_ENDPOINT = os.getenv("ASTRAL_API_KEY_ENDPOINT", "http://localhost:4000/dev/validate")
    ASTRAL_LOG_ENDPOINT = os.getenv("ASTRAL_LOG_ENDPOINT", "http://localhost:4000/dev")
    ASTRAL_BATCH_LOG_ENDPOINT = os.getenv("ASTRAL_BATCH_LOG_ENDPOINT", "http://localhost:4000/dev/batch")
else:
    # Production endpoints
    ASTRAL_API_BASE_URL = os.getenv("ASTRAL_API_BASE_URL", "https://8srqt59tm8.execute-api.us-east-1.amazonaws.com/dev/")
    ASTRAL_API_KEY_ENDPOINT = os.getenv("ASTRAL_API_KEY_ENDPOINT", "https://8srqt59tm8.execute-api.us-east-1.amazonaws.com/dev/validate")
    ASTRAL_LOG_ENDPOINT = os.getenv("ASTRAL_LOG_ENDPOINT", "https://8srqt59tm8.execute-api.us-east-1.amazonaws.com/dev/")
    ASTRAL_BATCH_LOG_ENDPOINT = os.getenv("ASTRAL_BATCH_LOG_ENDPOINT", "https://8srqt59tm8.execute-api.us-east-1.amazonaws.com/dev/batch")

# Batch size: number of events to accumulate before sending in one HTTP request.
BATCH_SIZE = int(os.getenv("ASTRAL_BATCH_SIZE", "3"))
# Batch timeout: maximum time (in seconds) to wait for more events before sending the batch.
BATCH_TIMEOUT = float(os.getenv("ASTRAL_BATCH_TIMEOUT", "0.5"))

# Type variables for decorators and type hints
P = ParamSpec("P")
R = TypeVar("R")
Self = TypeVar("Self")
RequestT = TypeVar('RequestT', bound=AstralBaseRequest)
ResponseT = TypeVar('ResponseT', bound=AstralBaseResponse)

# -------------------------------------------------------------------------------- #
# AstralBaseLogEvent Dataclass
# -------------------------------------------------------------------------------- #


@dataclass
class AstralBaseLogEvent:
    """
    Data class representing a log event capturing information about a request/response cycle.
    This class is designed to be reused via an object pool to reduce garbage collection overhead.
    """
    log_id: str
    astral_api_key: str

    # Caller Information
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = ""

    # Model Information
    model_name: Optional[ModelName] = None
    model_provider: Optional[ModelProvider] = None

    # Resource Information
    resource_type: Optional[ResourceType] = None
    resource_subtype: Optional[str] = None
    # Request Information
    request_id: Optional[str] = None
    request: Optional[Dict[str, Any]] = None

    # Response Information
    response_id: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    content: Optional[str] = None

    # Usage Information
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None

    # Log Status
    log_status: Optional[str] = "success"

    # Latency (populated by @timeit or @atimeit if the result supports it)
    latency_ms: Optional[float] = None

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Recursively traverse a dictionary or list and convert non-serializable objects
        into their string representation to ensure the final payload is JSON serializable.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the log event to a dictionary that is JSON serializable,
        handling nested objects with custom conversion logic.
        """
        result = asdict(self)
        # Convert any remaining objects to dict if they support .dict() or .model_dump()
        for key, value in result.items():
            if value is not None:
                if hasattr(value, "dict") and callable(value.dict):
                    result[key] = value.dict()
                elif hasattr(value, "model_dump") and callable(value.model_dump):
                    result[key] = value.model_dump()
        # Ensure the entire structure is JSON serializable
        result = self._make_json_serializable(result)
        return result

# -------------------------------------------------------------------------------- #
# Utility to Convert Objects to Dict
# -------------------------------------------------------------------------------- #


def _extract_dict_from_object(obj: Any, force_json: bool = False) -> Optional[Dict[str, Any]]:
    """
    Extract a dictionary from various objects (including Pydantic models, dataclasses, etc.)
    without incurring heavy overhead if not needed.
    """

    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    if isinstance(obj, BaseModel) and hasattr(obj, "model_dump"):
        if force_json:
            return obj.model_dump_json()
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return None

# -------------------------------------------------------------------------------- #
# Event Object Pool
# -------------------------------------------------------------------------------- #


class EventObjectPool:
    """
    Pool of reusable AstralBaseLogEvent objects to reduce garbage collection pressure.
    Pre-allocates and recycles event objects, resetting their state between uses.
    """

    def __init__(self, max_size: int = 100):
        self._pool: List[AstralBaseLogEvent] = []
        self._max_size = max_size
        self._lock = threading.RLock()

    def get(self) -> AstralBaseLogEvent:
        """
        Retrieve an event object from the pool if available; otherwise, create a new one.
        The caller is responsible for populating the event fields.
        """
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return AstralBaseLogEvent(
            log_id="",
            astral_api_key=ASTRAL_API_KEY,
        )

    def put(self, event: AstralBaseLogEvent) -> None:
        """
        Reset an event object's fields to default values and return it to the pool for reuse.
        """
        event.log_id = ""
        event.user_id = None
        event.organization_id = None
        event.project_id = ""
        event.model_name = None
        event.model_provider = None
        event.resource_type = None
        event.resource_subtype = None
        event.request_id = None
        event.request = None
        event.response_id = None
        event.response = None
        event.content = None
        event.usage = None
        event.cost = None
        event.log_status = "success"
        event.latency_ms = None

        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(event)

# -------------------------------------------------------------------------------- #
# Non-Blocking Logger for Astral
# -------------------------------------------------------------------------------- #


class NonBlockingLogger:
    """
    Thread-safe, non-blocking logger for Astral that minimizes impact on user requests.
    Key features include:
    - Non-blocking enqueue of log events.
    - Asynchronous processing using a dedicated daemon thread.
    - Batched delivery of log events over HTTP to reduce network overhead.
    - Object pooling for efficient reuse of log event objects.
    - Graceful shutdown with a best-effort flush of remaining events.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        # No need for self.enabled as we'll use the global LOG_ENABLED flag

        # Initialize the object pool for log events.
        self.event_pool = EventObjectPool(max_size=100)

        # Bounded queue for log events; non-blocking on enqueue.
        self.queue = queue.Queue(maxsize=1000)

        # Statistics for dropped events due to a full queue.
        self.dropped_count = 0
        self.stats_lock = threading.RLock()

        # HTTP client for log delivery.
        self.http_client = None

        # Flag to signal shutdown.
        self.stopping = False

        # Daemon thread for processing logs.
        self.daemon_thread = None
        if LOG_ENABLED:
            self._initialize()

    def _initialize(self):
        """
        Initialize the logger by setting up the HTTP client and starting the daemon thread.
        Also, register shutdown handlers to ensure graceful termination.
        """
        global LOG_ENABLED

        logger.trace(f"{OBSERVABILITY_EMOJI} Initializing non-blocking observability for Astral")

        # Set up the HTTP client with default timeout and headers.
        self.http_client = httpx.Client(
            timeout=10.0,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
        )
        logger.trace(f"{OBSERVABILITY_EMOJI} HTTP observability client initialized")

        # Validate the API key before going further
        if not self._validate_api_key():
            logger.warning(f"{OBSERVABILITY_EMOJI} API key validation failed - observability will be disabled")
            LOG_ENABLED = False
            if self.http_client:
                self.http_client.close()
                self.http_client = None
            return

        # Start the daemon thread for processing log events in batches.
        self.daemon_thread = threading.Thread(
            target=self._process_logs,
            daemon=True,
            name="AstralObservabilityThread"
        )
        self.daemon_thread.start()

        # Register shutdown handler to flush events on exit.
        atexit.register(self._on_shutdown)
        try:
            signal.signal(signal.SIGTERM, lambda s, f: self._on_shutdown())
            signal.signal(signal.SIGINT, lambda s, f: self._on_shutdown())
        except (ValueError, RuntimeError):
            # Signal handling may not be permitted in some environments.
            pass

        if ASTRAL_BATCH_LOG_ENDPOINT:
            logger.trace(f"{OBSERVABILITY_EMOJI} Non-blocking observability initialized with both batching and single event endpoints: {ASTRAL_BATCH_LOG_ENDPOINT} and {ASTRAL_LOG_ENDPOINT}")
        else:
            logger.trace(f"{OBSERVABILITY_EMOJI} Non-blocking observability initialized with only single event endpoint: {ASTRAL_LOG_ENDPOINT}")

    def _validate_api_key(self) -> bool:
        """
        Validate the API key by making a lightweight request to the API.
        Returns True if the key is valid, False otherwise.
        """
        global LOG_ENABLED

        if not self.api_key or not self.http_client:
            logger.warning(f"{OBSERVABILITY_EMOJI} No API key provided or HTTP client not initialized")
            return False

        try:
            # Try to send a test request to validate the key
            logger.trace(f"{OBSERVABILITY_EMOJI} Validating API key with request to {ASTRAL_API_KEY_ENDPOINT}")
            response = self.http_client.head(
                ASTRAL_API_KEY_ENDPOINT,
                headers={"x-api-key": self.api_key},
                timeout=5.0
            )

            if response.status_code == 401 or response.status_code == 403:
                logger.error(f"{OBSERVABILITY_EMOJI} API KEY VALIDATION FAILED: Status {response.status_code} - Your API key appears to be invalid, inactive, or expired")
                logger.error(f"{OBSERVABILITY_EMOJI} To fix this, go to https://useastral.dev/dashboard/projects to create or copy a new API key")
                logger.error(f"{OBSERVABILITY_EMOJI} Then set it as the ASTRAL_API_KEY environment variable")
                return False

            logger.debug(f"{OBSERVABILITY_EMOJI} API key validated successfully")
            return True
        except Exception as e:
            logger.error(f"{OBSERVABILITY_EMOJI} API key validation request failed: {type(e).__name__}: {e}")
            return False

    def _process_logs(self):
        """
        Background daemon thread that processes queued log events in batches.
        It accumulates up to BATCH_SIZE events or waits for BATCH_TIMEOUT seconds before sending.
        """
        while not self.stopping:
            batch = []
            try:
                # Wait for the first event with a timeout.
                event = self.queue.get(timeout=BATCH_TIMEOUT)
                batch.append(event)
            except queue.Empty:
                continue

            # Collect additional events without blocking until the batch is full or the queue is empty.
            while len(batch) < BATCH_SIZE:
                try:
                    event = self.queue.get_nowait()
                    batch.append(event)
                except queue.Empty:
                    break

            # Send the batch of events.
            try:
                self._send_events(batch)
            except Exception as e:
                logger.error(f"{OBSERVABILITY_EMOJI} Unexpected error in _send_events: {e}")

            # Mark tasks as done and return to pool regardless of success/failure
            for event in batch:
                try:
                    self.queue.task_done()
                except Exception:
                    pass  # Ignore errors in task_done
                try:
                    # Return event to pool for reuse.
                    self.event_pool.put(event)
                except Exception as e:
                    logger.error(f"{OBSERVABILITY_EMOJI} Error returning event to pool: {e}")

    def _send_event(self, event: AstralBaseLogEvent) -> None:
        """
        Send a single log event to the single event endpoint with retry logic.
        This method is used when batch sending is not available or for single events.
        """
        global LOG_ENABLED

        if not self.http_client or not LOG_ENABLED:
            return

        try:
            payload = event.to_dict()
        except Exception as e:
            logger.error(f"{OBSERVABILITY_EMOJI} Error converting event to dict: {e}")
            return

        max_retries = 2
        retry_delay = 0.1

        for attempt in range(max_retries + 1):
            if self.stopping and attempt > 0:
                logger.trace(f"{OBSERVABILITY_EMOJI} Stopping, not retrying log send")
                return
            try:
                logger.trace(f"{OBSERVABILITY_EMOJI} Attempting to send log {payload.get('log_id')} to {ASTRAL_LOG_ENDPOINT}")
                response = self.http_client.post(
                    ASTRAL_LOG_ENDPOINT,
                    json=payload,
                    timeout=5.0
                )
                logger.trace(f"{OBSERVABILITY_EMOJI} Log {payload.get('log_id')} sent with status code {response.status_code}")
                if response.status_code < 400:
                    return  # Event sent successfully.
                elif response.status_code in (400, 401, 403):
                    try:
                        error_body = response.json()
                        logger.error(f"{OBSERVABILITY_EMOJI} Permanent error sending log event: {response.status_code} - {error_body}")
                    except Exception:
                        logger.error(f"{OBSERVABILITY_EMOJI} Permanent error sending log event: {response.status_code} - {response.text}")

                    # If we get an auth error, disable logging to prevent further failures
                    if response.status_code in (401, 403):
                        logger.error(f"{OBSERVABILITY_EMOJI} API KEY ERROR: Authentication failed. Disabling observability.")
                        logger.error(f"{OBSERVABILITY_EMOJI} To fix this, go to https://useastral.dev/dashboard/projects to create or copy a new API key")
                        LOG_ENABLED = False
                    return
                else:
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"{OBSERVABILITY_EMOJI} Failed to send log after {max_retries} retries")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"{OBSERVABILITY_EMOJI} Network error after {max_retries} retries: {e}")
            except Exception as e:
                logger.error(f"{OBSERVABILITY_EMOJI} Unexpected error sending log event: {e}")
                return

    def _send_events(self, events: List[AstralBaseLogEvent]) -> None:
        """
        Send a list of log events in a single HTTP request if a batch endpoint is configured.
        Otherwise, fall back to sending each event individually.
        Implements retry logic with exponential backoff for temporary errors.
        """
        global LOG_ENABLED

        if not self.http_client or not LOG_ENABLED:
            return

        max_retries = 2
        retry_delay = 0.1

        # Convert events to their dictionary representations.
        try:
            payload = [event.to_dict() for event in events]
        except Exception as e:
            logger.error(f"{OBSERVABILITY_EMOJI} Error converting events to dict for batch send: {e}")
            return

        # If a batch endpoint is provided and we have more than one event, use it.
        if ASTRAL_BATCH_LOG_ENDPOINT and len(payload) > 1:
            endpoint = ASTRAL_BATCH_LOG_ENDPOINT
            for attempt in range(max_retries + 1):
                if self.stopping and attempt > 0:
                    logger.trace(f"{OBSERVABILITY_EMOJI} Stopping, not retrying batch log send")
                    return
                try:
                    logger.trace(f"{OBSERVABILITY_EMOJI} Attempting batch send of {len(payload)} events")
                    response = self.http_client.post(
                        endpoint,
                        json=payload,
                        timeout=5.0
                    )
                    logger.trace(f"{OBSERVABILITY_EMOJI} Batch send completed with status code {response.status_code}")
                    if response.status_code < 400:
                        return  # Batch send successful.
                    elif response.status_code in (400, 401, 403):
                        try:
                            error_body = response.json()
                            logger.error(f"{OBSERVABILITY_EMOJI} Permanent error during batch send: {response.status_code} - {error_body}")
                        except Exception:
                            logger.error(f"{OBSERVABILITY_EMOJI} Permanent error during batch send: {response.status_code} - {response.text}")

                        # If we get an auth error, disable logging to prevent further failures
                        if response.status_code in (401, 403):
                            logger.error(f"{OBSERVABILITY_EMOJI} API KEY ERROR: Authentication failed during batch send. Disabling observability.")
                            logger.error(f"{OBSERVABILITY_EMOJI} To fix this, go to https://useastral.dev/dashboard/projects to create or copy a new API key")
                            LOG_ENABLED = False
                        return
                    else:
                        if attempt < max_retries:
                            time.sleep(retry_delay * (2 ** attempt))
                        else:
                            logger.error(f"{OBSERVABILITY_EMOJI} Failed batch send after {max_retries} retries")
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"{OBSERVABILITY_EMOJI} Network error during batch send after {max_retries} retries: {e}")
                except Exception as e:
                    logger.error(f"{OBSERVABILITY_EMOJI} Unexpected error during batch send: {e}")
                    return
        else:
            # Fallback: send events individually.
            for event in events:
                self._send_event(event)

    def log_event(self, event: AstralBaseLogEvent) -> bool:
        """
        Enqueue a log event for asynchronous processing.
        This method is completely non-blocking; if the queue is full, the event is dropped.
        Returns True if the event was successfully enqueued, False otherwise.
        """
        if not LOG_ENABLED or self.stopping:
            self.event_pool.put(event)  # Return event to pool
            if not LOG_ENABLED:
                logger.trace(f"{OBSERVABILITY_EMOJI} Skipping log event {event.log_id} - observability is disabled")
            return False

        try:
            self.queue.put_nowait(event)
            logger.trace(f"{OBSERVABILITY_EMOJI} Event queued: {event.log_id}")
            return True
        except queue.Full:
            with self.stats_lock:
                self.dropped_count += 1
                if self.dropped_count % 100 == 1:
                    logger.warning(f"{OBSERVABILITY_EMOJI} Queue full, dropped {self.dropped_count} events so far")
            self.event_pool.put(event)
            return False

    def _on_shutdown(self):
        """
        Handle graceful shutdown by attempting to flush remaining log events.
        Drains the queue and sends any remaining events in a batch if possible.
        """
        if not LOG_ENABLED or self.stopping:
            return

        logger.trace(f"{OBSERVABILITY_EMOJI} Shutting down Astral observability")
        self.stopping = True

        # Local reference to the HTTP client for flushing logs.
        http_client = self.http_client

        try:
            # Brief grace period to allow queued events to become visible.
            time.sleep(0.05)
            logger.trace(f"{OBSERVABILITY_EMOJI} Attempting to flush remaining events during shutdown")

            # Drain a number of events from the queue for flushing.
            remaining_items = []
            max_items = 50  # Process up to 50 remaining events.

            # First, check for any pending event in the queue
            while len(remaining_items) < max_items:
                try:
                    item = self.queue.get(block=True, timeout=0.05)
                    remaining_items.append(item)
                    self.queue.task_done()
                except queue.Empty:
                    break

            if remaining_items and http_client is not None:
                logger.trace(f"{OBSERVABILITY_EMOJI} Processing {len(remaining_items)} queued events during shutdown")

                # Process one by one to ensure errors are captured
                for event in remaining_items:
                    self._send_event(event)
                    self.event_pool.put(event)

                # Extra wait to allow error processing to complete
                time.sleep(0.1)

            elif http_client is None:
                logger.trace(f"{OBSERVABILITY_EMOJI} No HTTP client available, skipping flush")
            else:
                logger.trace(f"{OBSERVABILITY_EMOJI} No events to flush, closing HTTP client")
        finally:
            if http_client:
                try:
                    # Give a moment for any final logs to be processed
                    time.sleep(0.1)
                    http_client.close()
                    logger.trace(f"{OBSERVABILITY_EMOJI} HTTP client closed successfully")
                except Exception as e:
                    logger.trace(f"{OBSERVABILITY_EMOJI} Error closing HTTP client: {e}")
            self.http_client = None
            logger.trace(f"{OBSERVABILITY_EMOJI} Astral observability shutdown complete")

    def shutdown(self):
        """
        Public method to trigger a graceful shutdown of the logger.
        """
        self._on_shutdown()


# -------------------------------------------------------------------------------- #
# Global Logger Instance
# -------------------------------------------------------------------------------- #
_GLOBAL_LOGGER = NonBlockingLogger(ASTRAL_API_KEY)


def _new_event() -> AstralBaseLogEvent:
    """
    Retrieve a new or recycled log event from the object pool.
    """
    return _GLOBAL_LOGGER.event_pool.get()


def log_event_sync(event: AstralBaseLogEvent) -> None:
    """
    Synchronously enqueue a log event for non-blocking asynchronous processing.
    This function is designed to have minimal impact on the calling thread.
    """
    if not LOG_ENABLED:
        _GLOBAL_LOGGER.event_pool.put(event)  # Return event to pool
        return
    _GLOBAL_LOGGER.log_event(event)


async def flush_logs() -> None:
    """
    Best-effort asynchronous flush of log events.
    Yields control briefly to allow the background thread to process queued events.
    """
    if not LOG_ENABLED:
        return
    await asyncio.sleep(0.01)

# -------------------------------------------------------------------------------- #
# Logging Decorators
# -------------------------------------------------------------------------------- #

ResourceSubtype = Literal["standard", "structured", "json"]


def log_event(resource_subtype: ResourceSubtype) -> Callable[
    [Callable[Concatenate[Self, RequestT, P], ResponseT]],
    Callable[Concatenate[Self, RequestT, P], ResponseT]
]:
    """
    Decorator for synchronous functions to perform non-blocking logging.
    Captures pre- and post-call event data and enqueues a log event for asynchronous processing.
    The first argument of the wrapped function is expected to be an instance of AstralBaseRequest.
    """
    def decorator(func: Callable[Concatenate[Self, RequestT, P], ResponseT]) -> Callable[Concatenate[Self, RequestT, P], ResponseT]:
        @wraps(func)
        def wrapper(self: Self, request: RequestT, *args: P.args, **kwargs: P.kwargs) -> ResponseT:
            if not LOG_ENABLED:
                return func(self, request, *args, **kwargs)

            # Check if logging is enabled for the given request.
            should_store = request.astral_params.store
            if not should_store:
                logger.trace(f"{OBSERVABILITY_EMOJI} Skipping logging due to store=False in astral_params")
                return func(self, request, *args, **kwargs)

            # TODO: Resource Type + Subtype will be how we build the log events in the future

            # Capture pre-call event data.
            event = _new_event()
            event.log_id = str(uuid.uuid4())
            event.astral_api_key = ASTRAL_API_KEY

            # Set the model information.
            event.model_name = request.model
            event.model_provider = request.provider_name
            event.resource_type = request.resource_type
            event.resource_subtype = resource_subtype

            logger.trace(f"{OBSERVABILITY_EMOJI} The resource subtype is {event.resource_subtype}")

            # Set the request information.
            event.request_id = request.request_id
            event.request = _extract_dict_from_object(request)

            try:
                result = func(self, request, *args, **kwargs)

                # Capture post-call event data.
                event.response_id = result.response_id

                # Process content field: if it's a string, use it directly; otherwise, extract dict
                if hasattr(result, "content") and result.content is not None:
                    if isinstance(result.content, str):
                        # event.content = {"raw": result.content}
                        event.content = result.content
                        logger.trace(f"{OBSERVABILITY_EMOJI} The content is a string, using it directly as {event.content}")
                    else:
                        event.content = _extract_dict_from_object(result.content, force_json=True)
                        logger.trace(f"{OBSERVABILITY_EMOJI} Successfully extracted JSON representation from content: {event.content}")
                # Handle tool calls if content is None
                elif hasattr(result, "response") and hasattr(result.response, "choices") and result.response.choices:
                    first_choice = result.response.choices[0]
                    if hasattr(first_choice, "message") and hasattr(first_choice.message, "tool_calls") and first_choice.message.tool_calls:
                        tool_call = first_choice.message.tool_calls[0]  # Assuming the first tool call is relevant
                        if hasattr(tool_call, "function"):
                            logger.trace(f"{OBSERVABILITY_EMOJI} Detected tool call: type={tool_call.type}, name={tool_call.function.name}")
                            event.content = {
                                "type": tool_call.type,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments  # Store arguments as string
                            }
                            logger.trace(f"{OBSERVABILITY_EMOJI} Stored tool call info in event.content: {event.content}")
                        else:
                            logger.trace(f"{OBSERVABILITY_EMOJI} Tool call found but structure is unexpected: {tool_call}")
                    else:
                        logger.trace(f"{OBSERVABILITY_EMOJI} Content is None, but no tool calls found in response.choices[0].message")
                else:
                    logger.trace(f"{OBSERVABILITY_EMOJI} Content is None, and response structure doesn't match expected tool call path.")

                # Process response field separately (holds raw/provider response structure)
                event.response = _extract_dict_from_object(result.response)

                event.usage = _extract_dict_from_object(result.usage)
                event.cost = _extract_dict_from_object(result.cost)
                event.latency_ms = getattr(result, "latency_ms", None)
                logger.trace(f"{OBSERVABILITY_EMOJI} The event's latency is {event.latency_ms}")
                log_event_sync(event)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {type(e).__name__}: {e}")
                event.log_status = "error"
                log_event_sync(event)
                raise
        return wrapper
    return decorator


def log_event_async(resource_subtype: ResourceSubtype) -> Callable[
    [Callable[Concatenate[Self, RequestT, P], Awaitable[ResponseT]]],
    Callable[Concatenate[Self, RequestT, P], Awaitable[ResponseT]]
]:
    """
    Decorator for asynchronous functions to perform non-blocking logging.
    Captures pre- and post-call event data and enqueues a log event for asynchronous processing.
    The first argument of the wrapped function is expected to be an instance of AstralBaseRequest.
    """
    def decorator(func: Callable[Concatenate[Self, RequestT, P], Awaitable[ResponseT]]) -> Callable[Concatenate[Self, RequestT, P], Awaitable[ResponseT]]:
        @wraps(func)
        async def wrapper(self: Self, request: RequestT, *args: P.args, **kwargs: P.kwargs) -> ResponseT:
            if not LOG_ENABLED:
                return await func(self, request, *args, **kwargs)

            should_store = request.astral_params.store
            if not should_store:
                logger.trace(f"{OBSERVABILITY_EMOJI} Skipping logging due to store=False in astral_params")
                return await func(self, request, *args, **kwargs)

            # Capture pre-call event data.
            event = _new_event()
            event.log_id = str(uuid.uuid4())
            event.astral_api_key = ASTRAL_API_KEY

            # Set the model information.
            event.model_name = request.model
            event.model_provider = request.provider_name
            event.resource_type = request.resource_type
            event.resource_subtype = resource_subtype

            # Set the request information.
            event.request = _extract_dict_from_object(request)

            # Set the request information.
            event.request_id = request.request_id
            event.request = _extract_dict_from_object(request)

            try:
                result = await func(self, request, *args, **kwargs)

                # Process content field: if it's a string, use it directly; otherwise, extract dict
                if hasattr(result, "content") and result.content is not None:
                    if isinstance(result.content, str):
                        # event.content = {"raw": result.content}
                        event.content = result.content
                        logger.trace(f"{OBSERVABILITY_EMOJI} The content is a string, using it directly as {event.content}")
                    else:
                        event.content = _extract_dict_from_object(result.content, force_json=True)
                        logger.trace(f"{OBSERVABILITY_EMOJI} Successfully extracted JSON representation from content: {event.content}")
                # Handle tool calls if content is None
                elif hasattr(result, "response") and hasattr(result.response, "choices") and result.response.choices:
                    first_choice = result.response.choices[0]
                    if hasattr(first_choice, "message") and hasattr(first_choice.message, "tool_calls") and first_choice.message.tool_calls:
                        tool_call = first_choice.message.tool_calls[0]  # Assuming the first tool call is relevant
                        if hasattr(tool_call, "function"):
                            logger.trace(f"{OBSERVABILITY_EMOJI} Detected tool call (async): type={tool_call.type}, name={tool_call.function.name}")
                            event.content = {
                                "type": tool_call.type,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments  # Store arguments as string
                            }
                            logger.trace(f"{OBSERVABILITY_EMOJI} Stored tool call info in event.content (async): {event.content}")
                        else:
                            logger.trace(f"{OBSERVABILITY_EMOJI} Tool call found but structure is unexpected (async): {tool_call}")
                    else:
                        logger.trace(f"{OBSERVABILITY_EMOJI} Content is None, but no tool calls found in response.choices[0].message (async)")
                else:
                    logger.trace(f"{OBSERVABILITY_EMOJI} Content is None, and response structure doesn't match expected tool call path (async).")

                if hasattr(result, "response_id"):
                    event.response_id = result.response_id
                if hasattr(result, "response"):
                    logger.trace(f"{OBSERVABILITY_EMOJI} The response is {result.response}")
                    event.response = _extract_dict_from_object(result.response)
                if hasattr(result, "usage"):
                    event.usage = _extract_dict_from_object(result.usage)
                if hasattr(result, "cost"):
                    event.cost = _extract_dict_from_object(result.cost)
                if hasattr(result, "latency_ms"):
                    event.latency_ms = getattr(result, "latency_ms", None)

                log_event_sync(event)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__} (async): {type(e).__name__}: {e}")
                event.log_status = "error"
                log_event_sync(event)
                raise
        return wrapper
    return decorator
