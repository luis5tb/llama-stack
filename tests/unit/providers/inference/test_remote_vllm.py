# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChoiceChunk,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta as OpenAIChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
)
from openai.types.model import Model as OpenAIModel

from llama_stack.apis.agents.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
)
from llama_stack.apis.common.responses import Order
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponseEventType,
    CompletionMessage,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChoice,
    SystemMessage,
    ToolChoice,
    ToolConfig,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.models import Model
from llama_stack.models.llama.datatypes import StopReason, ToolCall
from llama_stack.providers.datatypes import HealthStatus
from llama_stack.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.vllm.vllm import (
    VLLMInferenceAdapter,
    _process_vllm_chat_completion_stream_response,
)

# These are unit test for the remote vllm provider
# implementation. This should only contain tests which are specific to
# the implementation details of those classes. More general
# (API-level) tests should be placed in tests/integration/inference/
#
# How to run this test:
#
# pytest tests/unit/providers/inference/test_remote_vllm.py \
# -v -s --tb=short --disable-warnings


@pytest.fixture(scope="module")
def mock_openai_models_list():
    with patch("openai.resources.models.AsyncModels.list", new_callable=AsyncMock) as mock_list:
        yield mock_list


@pytest.fixture(scope="module")
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config)
    inference_adapter.model_store = AsyncMock()
    await inference_adapter.initialize()
    return inference_adapter


async def test_register_model_checks_vllm(mock_openai_models_list, vllm_inference_adapter):
    async def mock_openai_models():
        yield OpenAIModel(id="foo", created=1, object="model", owned_by="test")

    mock_openai_models_list.return_value = mock_openai_models()

    foo_model = Model(identifier="foo", provider_resource_id="foo", provider_id="vllm-inference")

    await vllm_inference_adapter.register_model(foo_model)
    mock_openai_models_list.assert_called()


async def test_old_vllm_tool_choice(vllm_inference_adapter):
    """
    Test that we set tool_choice to none when no tools are in use
    to support older versions of vLLM
    """
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
    vllm_inference_adapter.model_store.get_model.return_value = mock_model

    with patch.object(vllm_inference_adapter, "_nonstream_chat_completion") as mock_nonstream_completion:
        # No tools but auto tool choice
        await vllm_inference_adapter.chat_completion(
            "mock-model",
            [],
            stream=False,
            tools=None,
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )
        mock_nonstream_completion.assert_called()
        request = mock_nonstream_completion.call_args.args[0]
        # Ensure tool_choice gets converted to none for older vLLM versions
        assert request.tool_config.tool_choice == ToolChoice.none


async def test_tool_call_response(vllm_inference_adapter):
    """Verify that tool call arguments from a CompletionMessage are correctly converted
    into the expected JSON format."""

    # Patch the client property to avoid instantiating a real AsyncOpenAI client
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_create_client.return_value = mock_client

        messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="How many?"),
            CompletionMessage(
                content="",
                stop_reason=StopReason.end_of_turn,
                tool_calls=[
                    ToolCall(
                        call_id="foo",
                        tool_name="knowledge_search",
                        arguments={"query": "How many?"},
                        arguments_json='{"query": "How many?"}',
                    )
                ],
            ),
            ToolResponseMessage(call_id="foo", content="knowledge_search found 5...."),
        ]
        await vllm_inference_adapter.chat_completion(
            "mock-model",
            messages,
            stream=False,
            tools=[],
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )

        assert mock_client.chat.completions.create.call_args.kwargs["messages"][2]["tool_calls"] == [
            {
                "id": "foo",
                "type": "function",
                "function": {"name": "knowledge_search", "arguments": '{"query": "How many?"}'},
            }
        ]


async def test_tool_call_delta_empty_tool_call_buf():
    """
    Test that we don't generate extra chunks when processing a
    tool call response that didn't call any tools. Previously we would
    emit chunks with spurious ToolCallParseStatus.succeeded or
    ToolCallParseStatus.failed when processing chunks that didn't
    actually make any tool calls.
    """

    async def mock_stream():
        delta = OpenAIChoiceDelta(content="", tool_calls=None)
        choices = [OpenAIChoiceChunk(delta=delta, finish_reason="stop", index=0)]
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 2
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "complete"
    assert chunks[1].event.stop_reason == StopReason.end_of_turn


async def test_tool_call_delta_streaming_arguments_dict():
    async def mock_stream():
        mock_chunk_1 = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="tc_1",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_2 = OpenAIChatCompletionChunk(
            id="chunk-2",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="tc_1",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments='{"number": 28, "power": 3}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_3 = OpenAIChatCompletionChunk(
            id="chunk-3",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(content="", tool_calls=None), finish_reason="tool_calls", index=0
                )
            ],
        )
        for chunk in [mock_chunk_1, mock_chunk_2, mock_chunk_3]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "progress"
    assert chunks[1].event.delta.type == "tool_call"
    assert chunks[1].event.delta.parse_status.value == "succeeded"
    assert chunks[1].event.delta.tool_call.arguments_json == '{"number": 28, "power": 3}'
    assert chunks[2].event.event_type.value == "complete"


async def test_multiple_tool_calls():
    async def mock_stream():
        mock_chunk_1 = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="",
                                index=1,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="power",
                                    arguments='{"number": 28, "power": 3}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_2 = OpenAIChatCompletionChunk(
            id="chunk-2",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(
                        content="",
                        tool_calls=[
                            OpenAIChoiceDeltaToolCall(
                                id="",
                                index=2,
                                function=OpenAIChoiceDeltaToolCallFunction(
                                    name="multiple",
                                    arguments='{"first_number": 4, "second_number": 7}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        )
        mock_chunk_3 = OpenAIChatCompletionChunk(
            id="chunk-3",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=[
                OpenAIChoiceChunk(
                    delta=OpenAIChoiceDelta(content="", tool_calls=None), finish_reason="tool_calls", index=0
                )
            ],
        )
        for chunk in [mock_chunk_1, mock_chunk_2, mock_chunk_3]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 4
    assert chunks[0].event.event_type.value == "start"
    assert chunks[1].event.event_type.value == "progress"
    assert chunks[1].event.delta.type == "tool_call"
    assert chunks[1].event.delta.parse_status.value == "succeeded"
    assert chunks[1].event.delta.tool_call.arguments_json == '{"number": 28, "power": 3}'
    assert chunks[2].event.event_type.value == "progress"
    assert chunks[2].event.delta.type == "tool_call"
    assert chunks[2].event.delta.parse_status.value == "succeeded"
    assert chunks[2].event.delta.tool_call.arguments_json == '{"first_number": 4, "second_number": 7}'
    assert chunks[3].event.event_type.value == "complete"


async def test_process_vllm_chat_completion_stream_response_no_choices():
    """
    Test that we don't error out when vLLM returns no choices for a
    completion request. This can happen when there's an error thrown
    in vLLM for example.
    """

    async def mock_stream():
        choices = []
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 1
    assert chunks[0].event.event_type.value == "start"


async def test_get_params_empty_tools(vllm_inference_adapter):
    request = ChatCompletionRequest(
        tools=[],
        model="test_model",
        messages=[UserMessage(content="test")],
    )
    params = await vllm_inference_adapter._get_params(request)
    assert "tools" not in params


async def test_process_vllm_chat_completion_stream_response_tool_call_args_last_chunk():
    """
    Tests the edge case where the model returns the arguments for the tool call in the same chunk that
    contains the finish reason (i.e., the last one).
    We want to make sure the tool call is executed in this case, and the parameters are passed correctly.
    """

    mock_tool_name = "mock_tool"
    mock_tool_arguments = {"arg1": 0, "arg2": 100}
    mock_tool_arguments_str = json.dumps(mock_tool_arguments)

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": None,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": None,
                                    "function": {
                                        "name": None,
                                        "arguments": mock_tool_arguments_str,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == mock_tool_arguments


async def test_process_vllm_chat_completion_stream_response_no_finish_reason():
    """
    Tests the edge case where the model requests a tool call and stays idle without explicitly providing the
    finish reason.
    We want to make sure that this case is recognized and handled correctly, i.e., as a valid end of message.
    """

    mock_tool_name = "mock_tool"
    mock_tool_arguments = {"arg1": 0, "arg2": 100}
    mock_tool_arguments_str = '"{\\"arg1\\": 0, \\"arg2\\": 100}"'

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": mock_tool_arguments_str,
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == mock_tool_arguments


async def test_process_vllm_chat_completion_stream_response_tool_without_args():
    """
    Tests the edge case where no arguments are provided for the tool call.
    Tool calls with no arguments should be treated as regular tool calls, which was not the case until now.
    """
    mock_tool_name = "mock_tool"

    async def mock_stream():
        mock_chunks = [
            OpenAIChatCompletionChunk(
                id="chunk-1",
                created=1,
                model="foo",
                object="chat.completion.chunk",
                choices=[
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "mock_id",
                                    "type": "function",
                                    "function": {
                                        "name": mock_tool_name,
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                        "logprobs": None,
                        "index": 0,
                    }
                ],
            ),
        ]
        for chunk in mock_chunks:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 3
    assert chunks[-1].event.event_type == ChatCompletionResponseEventType.complete
    assert chunks[-2].event.delta.type == "tool_call"
    assert chunks[-2].event.delta.tool_call.tool_name == mock_tool_name
    assert chunks[-2].event.delta.tool_call.arguments == {}


async def test_health_status_success(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection is successful.

    This test verifies that the health method returns a HealthResponse with status OK, only
    when the connection to the vLLM server is successful.
    """
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        # Create mock client and models
        mock_client = MagicMock()
        mock_models = MagicMock()

        # Create a mock async iterator that yields a model when iterated
        async def mock_list():
            for model in [MagicMock()]:
                yield model

        # Set up the models.list to return our mock async iterator
        mock_models.list.return_value = mock_list()
        mock_client.models = mock_models
        mock_create_client.return_value = mock_client

        # Call the health method
        health_response = await vllm_inference_adapter.health()
        # Verify the response
        assert health_response["status"] == HealthStatus.OK

        # Verify that models.list was called
        mock_models.list.assert_called_once()


async def test_health_status_failure(vllm_inference_adapter):
    """
    Test the health method of VLLM InferenceAdapter when the connection fails.

    This test verifies that the health method returns a HealthResponse with status ERROR
    and an appropriate error message when the connection to the vLLM server fails.
    """
    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        # Create mock client and models
        mock_client = MagicMock()
        mock_models = MagicMock()

        # Create a mock async iterator that raises an exception when iterated
        async def mock_list():
            raise Exception("Connection failed")
            yield  # Unreachable code

        # Set up the models.list to return our mock async iterator
        mock_models.list.return_value = mock_list()
        mock_client.models = mock_models
        mock_create_client.return_value = mock_client

        # Call the health method
        health_response = await vllm_inference_adapter.health()
        # Verify the response
        assert health_response["status"] == HealthStatus.ERROR
        assert "Health check failed: Connection failed" in health_response["message"]

        mock_models.list.assert_called_once()


async def test_openai_chat_completion_is_async(vllm_inference_adapter):
    """
    Verify that openai_chat_completion is async and doesn't block the event loop.

    To do this we mock the underlying inference with a sleep, start multiple
    inference calls in parallel, and ensure the total time taken is less
    than the sum of the individual sleep times.
    """
    sleep_time = 0.5

    async def mock_create(*args, **kwargs):
        await asyncio.sleep(sleep_time)
        return OpenAIChatCompletion(
            id="chatcmpl-abc123",
            created=1,
            model="mock-model",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="nothing interesting",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
        )

    async def do_inference():
        await vllm_inference_adapter.openai_chat_completion(
            "mock-model", messages=["one fish", "two fish"], stream=False
        )

    with patch.object(VLLMInferenceAdapter, "client", new_callable=PropertyMock) as mock_create_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_create_client.return_value = mock_client

        start_time = time.time()
        await asyncio.gather(do_inference(), do_inference(), do_inference(), do_inference())
        total_time = time.time() - start_time

        assert mock_create_client.call_count == 4  # no cheating
        assert total_time < (sleep_time * 2), f"Total time taken: {total_time}s exceeded expected max"


# OpenAI Responses API Tests
# These tests verify the direct passthrough functionality to vLLM's /v1/responses endpoint


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx.Response for testing."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock()
    return response


@pytest.fixture
def mock_httpx_stream_response():
    """Create a mock httpx streaming response for testing."""
    response = MagicMock()
    response.raise_for_status = MagicMock()

    async def mock_aiter_lines():
        # Simulate streaming response lines
        lines = [
            'data: {"type": "response.created", "response": {"id": "resp-123", "status": "in_progress"}}',
            'data: {"type": "response.output_text.delta", "delta": "Hello"}',
            'data: {"type": "response.completed", "response": {"id": "resp-123", "status": "completed"}}'
        ]
        for line in lines:
            yield line

    response.aiter_lines = mock_aiter_lines
    return response


async def test_create_openai_response_non_streaming(vllm_inference_adapter):
    """Test creating a non-streaming OpenAI response."""
    mock_response_data = {
        "id": "resp-123",
        "object": "response",
        "created_at": 1234567890,
        "model": "test-model",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello, world!"}]
            }
        ]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock the streaming response to return completed response
        async def mock_stream():
            yield OpenAIResponseObjectStreamResponseCompleted(
                type="response.completed",
                response=OpenAIResponseObject(**mock_response_data)
            )

        with patch.object(vllm_inference_adapter, '_create_streaming_openai_response', return_value=mock_stream()):
            response = await vllm_inference_adapter.create_openai_response(
                input="Test message",
                model="test-model",
                stream=False
            )

            assert isinstance(response, OpenAIResponseObject)
            assert response.id == "resp-123"
            assert response.status == "completed"


async def test_create_openai_response_streaming(vllm_inference_adapter):
    """Test creating a streaming OpenAI response."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock the streaming response
        async def mock_stream():
            yield OpenAIResponseObjectStreamResponseCreated(
                type="response.created",
                response=OpenAIResponseObject(
                    id="resp-123",
                    object="response",
                    created_at=1234567890,
                    model="test-model",
                    status="in_progress",
                    output=[]
                )
            )

        with patch.object(vllm_inference_adapter, '_create_streaming_openai_response', return_value=mock_stream()):
            response_gen = await vllm_inference_adapter.create_openai_response(
                input="Test message",
                model="test-model",
                stream=True
            )

            chunks = [chunk async for chunk in response_gen]
            assert len(chunks) == 1
            assert chunks[0].type == "response.created"


async def test_create_openai_response_with_tools(vllm_inference_adapter):
    """Test creating a response with various tool types including MCP."""
    function_tool = OpenAIResponseInputToolFunction(
        type="function",
        name="get_weather",
        description="Get weather information",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}}
    )

    mcp_tool = OpenAIResponseInputToolMCP(
        type="mcp",
        server_label="test-server",
        server_url="http://localhost:8080",
        require_approval="never"
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock the streaming response
        async def mock_stream():
            yield OpenAIResponseObjectStreamResponseCompleted(
                type="response.completed",
                response=OpenAIResponseObject(
                    id="resp-123",
                    object="response",
                    created_at=1234567890,
                    model="test-model",
                    status="completed",
                    output=[]
                )
            )

        with patch.object(vllm_inference_adapter, '_create_streaming_openai_response', return_value=mock_stream()):
            response = await vllm_inference_adapter.create_openai_response(
                input="What's the weather like?",
                model="test-model",
                tools=[function_tool, mcp_tool],
                stream=False
            )

            assert isinstance(response, OpenAIResponseObject)
            assert response.status == "completed"


async def test_create_streaming_openai_response_http_passthrough(vllm_inference_adapter):
    """Test the internal streaming method makes correct HTTP calls."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        # Mock the streaming context manager
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            lines = [
                'data: {"type": "response.created", "response": {"id": "resp-123", "object": "response", "created_at": 1234567890, "model": "test-model", "status": "in_progress", "output": []}}',
                'data: {"type": "response.completed", "response": {"id": "resp-123", "object": "response", "created_at": 1234567890, "model": "test-model", "status": "completed", "output": []}}'
            ]
            for line in lines:
                yield line

        mock_stream_response.aiter_lines = mock_aiter_lines
        mock_client.stream.return_value.__aenter__.return_value = mock_stream_response
        mock_client.stream.return_value.__aexit__.return_value = None

        chunks = []
        async for chunk in vllm_inference_adapter._create_streaming_openai_response(
            input="Test message",
            model="test-model",
            temperature=0.7
        ):
            chunks.append(chunk)

        # Verify HTTP call was made correctly
        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://mocked.localhost:12345/v1/responses"
        assert call_args[1]["json"]["input"] == "Test message"
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["json"]["temperature"] == 0.7
        assert call_args[1]["json"]["stream"] is True

        # Verify response parsing
        assert len(chunks) == 2
        assert chunks[0].type == "response.created"
        assert chunks[1].type == "response.completed"


async def test_get_openai_response(vllm_inference_adapter):
    """Test retrieving a specific OpenAI response by ID."""
    mock_response_data = {
        "id": "resp-123",
        "object": "response",
        "created_at": 1234567890,
        "model": "test-model",
        "status": "completed",
        "output": []
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_client.get.return_value = mock_response

        response = await vllm_inference_adapter.get_openai_response("resp-123")

        # Verify HTTP call
        mock_client.get.assert_called_once_with(
            "http://mocked.localhost:12345/v1/responses/resp-123",
            headers={}
        )

        # Verify response
        assert isinstance(response, OpenAIResponseObject)
        assert response.id == "resp-123"
        assert response.status == "completed"


async def test_list_openai_responses(vllm_inference_adapter):
    """Test listing OpenAI responses with pagination."""
    mock_list_data = {
        "object": "list",
        "data": [
            {
                "id": "resp-123",
                "object": "response",
                "created_at": 1234567890,
                "model": "test-model",
                "status": "completed",
                "output": [],
                "input": []
            }
        ],
        "has_more": False,
        "first_id": "resp-123",
        "last_id": "resp-123"
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_list_data
        mock_client.get.return_value = mock_response

        responses = await vllm_inference_adapter.list_openai_responses(
            limit=10,
            order=Order.desc,
            model="test-model"
        )

        # Verify HTTP call with parameters
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "http://mocked.localhost:12345/v1/responses"
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["model"] == "test-model"
        assert call_args[1]["params"]["order"] == "desc"

        # Verify response
        assert isinstance(responses, ListOpenAIResponseObject)
        assert len(responses.data) == 1
        assert responses.data[0].id == "resp-123"


async def test_list_openai_response_input_items(vllm_inference_adapter):
    """Test listing input items for a specific response."""
    mock_input_data = {
        "object": "list",
        "data": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "id": "msg-123"
            }
        ]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_input_data
        mock_client.get.return_value = mock_response

        input_items = await vllm_inference_adapter.list_openai_response_input_items(
            response_id="resp-123",
            limit=5
        )

        # Verify HTTP call
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "http://mocked.localhost:12345/v1/responses/resp-123/input_items"
        assert call_args[1]["params"]["limit"] == 5

        # Verify response
        assert isinstance(input_items, ListOpenAIResponseInputItem)
        assert len(input_items.data) == 1


async def test_delete_openai_response(vllm_inference_adapter):
    """Test deleting an OpenAI response."""
    mock_delete_data = {
        "id": "resp-123",
        "object": "response",
        "deleted": True
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_delete_data
        mock_client.delete.return_value = mock_response

        result = await vllm_inference_adapter.delete_openai_response("resp-123")

        # Verify HTTP call
        mock_client.delete.assert_called_once_with(
            "http://mocked.localhost:12345/v1/responses/resp-123",
            headers={}
        )

        # Verify response
        assert isinstance(result, OpenAIDeleteResponseObject)
        assert result.id == "resp-123"
        assert result.deleted is True


async def test_responses_api_with_authentication(vllm_inference_adapter):
    """Test that API token is properly included in headers."""
    # Set up adapter with API token
    vllm_inference_adapter.config.api_token = "test-token"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"id": "resp-123", "object": "response", "deleted": True}
        mock_client.delete.return_value = mock_response

        await vllm_inference_adapter.delete_openai_response("resp-123")

        # Verify authorization header is included
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"


async def test_responses_api_http_error_handling(vllm_inference_adapter):
    """Test that HTTP errors are properly propagated."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client_class.return_value.__aexit__.return_value = None

        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=MagicMock()
        )
        mock_client.get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await vllm_inference_adapter.get_openai_response("nonexistent-resp")


async def test_create_openai_response_stream_never_completes(vllm_inference_adapter):
    """Test error handling when streaming response never completes."""
    async def mock_stream():
        # Stream that never yields a completed response
        yield OpenAIResponseObjectStreamResponseCreated(
            type="response.created",
            response=OpenAIResponseObject(
                id="resp-123",
                object="response",
                created_at=1234567890,
                model="test-model",
                status="in_progress",
                output=[]
            )
        )

    with patch.object(vllm_inference_adapter, '_create_streaming_openai_response', return_value=mock_stream()):
        with pytest.raises(ValueError, match="The response stream never completed"):
            await vllm_inference_adapter.create_openai_response(
                input="Test message",
                model="test-model",
                stream=False
            )
