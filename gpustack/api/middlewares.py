from datetime import date, datetime, timezone
import json
import logging
import time
import uuid
from typing import Optional, Type, Union
from fastapi import Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from jwt import DecodeError, ExpiredSignatureError
from starlette.middleware.base import BaseHTTPMiddleware
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CompletionUsage
from openai.types.audio.transcription_create_response import (
    Transcription,
)
from openai.types.create_embedding_response import (
    Usage as EmbeddingUsage,
)
from gpustack.api.exceptions import ErrorResponse
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.routes.rerank import RerankResponse, RerankUsage
from gpustack.schemas.images import ImageGenerationChunk, ImagesResponse
from gpustack.schemas.model_usage import ModelUsage, ModelUsageLog, OperationEnum
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import JWT_TOKEN_EXPIRE_MINUTES, JWTManager
from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.server.services import (
    ModelUsageLogService,
    ModelUsageService,
    ModelUsageStatService,
)
from gpustack.api.types.openai_ext import CreateEmbeddingResponseExt, CompletionExt
from gpustack.utils.client_ip import get_client_ip


logger = logging.getLogger(__name__)


class RequestTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.start_time = datetime.now(timezone.utc)
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    reason="Internal Server Error",
                    message=f"Unexpected error occurred: {e}",
                ).model_dump(),
            )
        return response


class ModelUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        operation, response_class = get_model_usage_context(request.url.path)
        response = await call_next(request)
        if operation is not None and response_class is not None:
            return await process_request(request, response, response_class, operation)

        return response


def get_model_usage_context(path: str):
    if path in ("/v1-openai/chat/completions", "/v1/chat/completions"):
        return OperationEnum.CHAT_COMPLETION, ChatCompletion
    if path in ("/v1-openai/completions", "/v1/completions"):
        return OperationEnum.COMPLETION, CompletionExt
    if path in ("/v1-openai/embeddings", "/v1/embeddings"):
        return OperationEnum.EMBEDDING, CreateEmbeddingResponseExt
    if path in (
        "/v1-openai/images/generations",
        "/v1/images/generations",
        "/v1-openai/images/edits",
        "/v1/images/edits",
    ):
        return OperationEnum.IMAGE_GENERATION, ImagesResponse
    if path in ("/v1-openai/audio/speech", "/v1/audio/speech"):
        return OperationEnum.AUDIO_SPEECH, FileResponse
    if path in ("/v1-openai/audio/transcriptions", "/v1/audio/transcriptions"):
        return OperationEnum.AUDIO_TRANSCRIPTION, Transcription
    if path == "/v1/rerank":
        return OperationEnum.RERANK, RerankResponse
    return None, None


async def process_request(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[
            ChatCompletion,
            CompletionExt,
            CreateEmbeddingResponseExt,
            RerankResponse,
            ImagesResponse,
            FileResponse,
            Transcription,
        ]
    ],
    operation: OperationEnum,
):
    stream: bool = getattr(request.state, "stream", False)
    if stream:
        if response_class == ChatCompletion:
            response_class = ChatCompletionChunk
        if response_class == ImagesResponse:
            response_class = ImageGenerationChunk
        return await handle_streaming_response(
            request, response, response_class, operation
        )
    else:
        response_body = b"".join([chunk async for chunk in response.body_iterator])
        usage = None
        error_fields = {}
        try:
            content_type = response.headers.get("content-type", "")
            if content_type.lower().startswith("application/json"):
                response_dict = json.loads(response_body)
                if response.status_code == 200:
                    response_instance = response_class(**response_dict)
                    if hasattr(response_instance, "usage"):
                        usage = response_instance.usage
                else:
                    error_fields = extract_error_fields(response_dict)

            if response.status_code == 200:
                await record_model_usage(request, usage, operation)
        except Exception as e:
            logger.error(f"Error processing model usage: {e}")
        await record_model_usage_log(
            request,
            usage,
            operation,
            response.status_code,
            error_fields=error_fields,
        )
        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return response


async def record_model_usage(
    request: Request,
    usage: Union[CompletionUsage, EmbeddingUsage, RerankUsage, None],
    operation: OperationEnum,
):
    total_tokens = getattr(usage, 'total_tokens', 0) or 0
    prompt_tokens = getattr(usage, 'prompt_tokens', total_tokens) or total_tokens
    completion_tokens = (
        getattr(usage, 'completion_tokens', total_tokens - prompt_tokens)
        or total_tokens - prompt_tokens
    )

    user: User = request.state.user
    model: Model = request.state.model
    fields = {
        "user_id": user.id,
        "model_id": model.id,
        "date": datetime.now(timezone.utc).date(),
        "operation": operation,
    }
    model_usage = ModelUsage(
        **fields,
        completion_token_count=completion_tokens,
        prompt_token_count=prompt_tokens,
        request_count=1,
    )
    async with AsyncSession(get_engine()) as session:
        model_usage_service = ModelUsageService(session)
        current_model_usage = await model_usage_service.get_by_fields(fields)
        if current_model_usage:
            await model_usage_service.update(
                current_model_usage, completion_tokens, prompt_tokens
            )
        else:
            await model_usage_service.create(model_usage)


async def record_model_usage_log(
    request: Request,
    usage: Union[CompletionUsage, EmbeddingUsage, RerankUsage, None],
    operation: OperationEnum,
    status_code: Optional[int],
    error_fields: Optional[dict] = None,
):
    try:
        total_tokens = getattr(usage, 'total_tokens', 0) or 0
        prompt_tokens = getattr(usage, 'prompt_tokens', total_tokens) or total_tokens
        completion_tokens = (
            getattr(usage, 'completion_tokens', total_tokens - prompt_tokens)
            or total_tokens - prompt_tokens
        )
        usage_available = usage is not None
        if not usage_available:
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0

        now = datetime.now(timezone.utc)
        start_time = getattr(request.state, "start_time", now)
        duration_ms = int((now - start_time).total_seconds() * 1000)
        server_config = getattr(request.app.state, "server_config", None)
        trusted_proxy_cidrs = getattr(server_config, "trusted_proxy_cidrs", None)
        model: Optional[Model] = getattr(request.state, "model", None)
        user: Optional[User] = getattr(request.state, "user", None)
        error_fields = error_fields or {}
        success = status_code is not None and 200 <= status_code < 400

        model_usage_log = ModelUsageLog(
            request_id=get_request_id(request),
            call_time=start_time,
            date=start_time.date(),
            hour=start_time.hour,
            user_id=getattr(user, "id", None),
            api_key_id=getattr(request.state, "api_key_id", None),
            api_key_access_key=getattr(request.state, "api_key_access_key", None),
            model_id=getattr(model, "id", None),
            model_name=getattr(model, "name", None),
            operation=operation,
            source_ip=get_client_ip(request, trusted_proxy_cidrs),
            raw_forwarded_for=request.headers.get("x-forwarded-for"),
            prompt_token_count=prompt_tokens,
            completion_token_count=completion_tokens,
            total_token_count=total_tokens,
            usage_available=usage_available,
            status_code=status_code,
            success=success,
            duration_ms=duration_ms,
            ttft_ms=get_ttft_ms(request),
            tokens_per_second=getattr(usage, "tokens_per_second", None),
            error_code=error_fields.get("error_code"),
            error_type=error_fields.get("error_type"),
            error_message=sanitize_error_message(error_fields.get("error_message")),
            worker_id=getattr(request.state, "worker_id", None),
            worker_name=getattr(request.state, "worker_name", None),
            worker_ip=getattr(request.state, "worker_ip", None),
            model_instance_id=getattr(request.state, "model_instance_id", None),
        )
        async with AsyncSession(get_engine()) as session:
            model_usage_log = await ModelUsageLogService(session).add(model_usage_log)
            await ModelUsageStatService(session).record(model_usage_log)
            await session.commit()
    except Exception as e:
        logger.error(f"Error recording model usage log: {e}")


def get_request_id(request: Request) -> str:
    return (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or str(uuid.uuid4())
    )


def get_ttft_ms(request: Request) -> Optional[int]:
    first_token_time = getattr(request.state, "first_token_time", None)
    start_time = getattr(request.state, "start_time", None)
    if first_token_time is None or start_time is None:
        return None
    return int((first_token_time - start_time).total_seconds() * 1000)


def extract_error_fields(response_dict: dict) -> dict:
    error = response_dict.get("error") if isinstance(response_dict, dict) else None
    if isinstance(error, dict):
        return {
            "error_code": str(error.get("code")) if error.get("code") else None,
            "error_type": error.get("type"),
            "error_message": error.get("message"),
        }
    return {}


def sanitize_error_message(message: Optional[str]) -> Optional[str]:
    if not message:
        return None
    message = str(message)
    lowered = message.lower()
    for marker in ("authorization:", "bearer ", "api_key", "secret_key"):
        if marker in lowered:
            return "[redacted]"
    return message[:1024]


async def handle_streaming_response(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk]
    ],
    operation: OperationEnum,
):
    final_status_code = response.status_code

    async def streaming_generator():
        nonlocal final_status_code
        buffer = ""
        final_headers = {}
        async for chunk in response.body_iterator:
            try:
                if isinstance(chunk, tuple):
                    chunk_content, headers, chunk_status_code = chunk
                    final_status_code = chunk_status_code
                    final_headers = headers
                    if chunk_status_code >= 400:
                        await record_model_usage_log(
                            request,
                            None,
                            operation,
                            chunk_status_code,
                            error_fields=extract_error_fields_from_body(chunk_content),
                        )
                        request.state.model_usage_log_recorded = True
                        yield chunk
                        continue

                    async for processed_chunk in process_chunk(
                        chunk_content, request, response_class, operation, buffer
                    ):
                        if isinstance(processed_chunk, str):
                            buffer = processed_chunk
                        else:
                            yield processed_chunk, headers, chunk_status_code
                else:
                    async for processed_chunk in process_chunk(
                        chunk, request, response_class, operation, buffer
                    ):
                        if isinstance(processed_chunk, str):
                            buffer = processed_chunk
                        else:
                            yield processed_chunk
            except Exception as e:
                logger.error(f"Error processing streaming response: {e}")
                yield chunk
        if buffer:
            if isinstance(response, StreamingResponseWithStatusCode):
                yield buffer.encode("utf-8"), final_headers, final_status_code
            else:
                yield buffer.encode("utf-8")
        if not getattr(request.state, "model_usage_log_recorded", False):
            await record_model_usage_log(request, None, operation, final_status_code)

    if isinstance(response, StreamingResponseWithStatusCode):
        return StreamingResponseWithStatusCode(
            streaming_generator(), media_type=response.media_type
        )

    return StreamingResponse(
        streaming_generator(),
        status_code=response.status_code,
        headers=response.headers,
        media_type=response.media_type,
    )


def extract_error_fields_from_body(body) -> dict:
    try:
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            return extract_error_fields(json.loads(body))
    except Exception:
        return {}
    return {}


async def process_chunk(
    chunk,
    request,
    response_class,
    operation: OperationEnum,
    buffer: str = "",
):
    if not hasattr(request.state, 'first_token_time'):
        request.state.first_token_time = datetime.now(timezone.utc)

    # each chunk may contain multiple data lines
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8")
    chunk = buffer + chunk
    lines = chunk.split("\n\n")
    if not chunk.endswith("\n\n"):
        yield lines.pop()

    for event in lines:
        if not event:
            continue
        data = sse_event_data(event)
        if data is None:
            yield sse_event_bytes(event)
            continue

        if data.startswith('[DONE]'):
            yield sse_event_bytes(event)
            continue

        if '"usage":' in data:
            response_dict = None
            try:
                response_dict = json.loads(data.strip())
            except Exception as e:
                raise e
            response_chunk = response_class(**response_dict)

            if is_usage_chunk(response_chunk):
                await record_model_usage(request, response_chunk.usage, operation)

                # Fill rate metrics. These are extended info not included in OAI APIs.
                # llama-box provides them out-of-the-box. Align with other backends here.
                if should_add_metrics(response_dict):
                    add_metrics(response_dict, request, response_chunk)

                await record_model_usage_log(
                    request,
                    response_chunk.usage,
                    operation,
                    200,
                )
                request.state.model_usage_log_recorded = True

                yield sse_event_with_data(
                    event, json.dumps(response_dict, separators=(',', ':'))
                )
            else:
                yield sse_event_bytes(event)
        else:
            yield sse_event_bytes(event)


def sse_event_data(event: str) -> Optional[str]:
    data_lines = []
    for line in event.splitlines():
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if not data_lines:
        return None
    return "\n".join(data_lines)


def sse_event_bytes(event: str) -> bytes:
    return f"{event}\n\n".encode("utf-8")


def sse_event_with_data(event: str, data: str) -> bytes:
    lines = [line for line in event.splitlines() if not line.startswith("data:")]
    lines.append(f"data: {data}")
    return sse_event_bytes("\n".join(lines))


def should_add_metrics(response_dict):
    if not isinstance(response_dict, dict):
        return False

    usage = response_dict.get('usage', {})

    return 'prompt_tokens' in usage and 'tokens_per_second' not in usage


def add_metrics(response_dict, request, response_chunk):
    now = datetime.now(timezone.utc)
    time_to_first_token_ms = (
        request.state.first_token_time - request.state.start_time
    ).total_seconds() * 1000
    time_per_output_token_ms = (
        (now - request.state.first_token_time).total_seconds()
        * 1000
        / max(response_chunk.usage.completion_tokens, 1)
    )
    tokens_per_second = (
        1000 / time_per_output_token_ms if time_per_output_token_ms > 0 else 0
    )

    response_dict['usage'].update(
        {
            "time_to_first_token_ms": time_to_first_token_ms,
            "time_per_output_token_ms": time_per_output_token_ms,
            "tokens_per_second": tokens_per_second,
        }
    )


class RefreshTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        jwt_manager: JWTManager = request.app.state.jwt_manager
        token = request.cookies.get(SESSION_COOKIE_NAME)

        if token:
            try:
                payload = jwt_manager.decode_jwt_token(token)
                if payload:
                    # Check if the token is about to expire (less than 15 minutes left)
                    if payload['exp'] - time.time() < 15 * 60:
                        new_token = jwt_manager.create_jwt_token(
                            username=payload['sub']
                        )
                        response.set_cookie(
                            key=SESSION_COOKIE_NAME,
                            value=new_token,
                            httponly=True,
                            max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
                            expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
                        )
            except (ExpiredSignatureError, DecodeError):
                pass

        return response


def is_usage_chunk(
    chunk: Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk],
) -> bool:
    choices = getattr(chunk, "choices", None)

    if not choices and chunk.usage:
        return True

    for choice in choices or []:
        if choice.finish_reason is not None and chunk.usage:
            return True

    return False
