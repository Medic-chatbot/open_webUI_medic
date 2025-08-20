import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from open_webui.env import (
    ENABLE_GEMINI_API,
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY,
    SRC_LOG_LEVELS,
)
from open_webui.utils.auth import get_verified_user
from open_webui.utils.gemini import GeminiAPIError, GeminiClient, GeminiError
from pydantic import BaseModel

# Initialize Gemini client
gemini_client = GeminiClient()

router = APIRouter()
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["GEMINI"])


class GeminiConfig(BaseModel):
    api_key: str
    api_base_url: str
    enabled: bool = True


@router.get("/config")
async def get_config(user=Depends(get_verified_user)) -> GeminiConfig:
    """Get Gemini API configuration"""
    return GeminiConfig(
        api_key=GEMINI_API_KEY,
        api_base_url=GEMINI_API_BASE_URL,
        enabled=ENABLE_GEMINI_API,
    )


@router.post("/config/update")
async def update_config(
    config: GeminiConfig, user=Depends(get_verified_user)
) -> GeminiConfig:
    """Update Gemini API configuration"""
    global GEMINI_API_KEY, GEMINI_API_BASE_URL, ENABLE_GEMINI_API

    GEMINI_API_KEY = config.api_key
    GEMINI_API_BASE_URL = config.api_base_url
    ENABLE_GEMINI_API = config.enabled

    return config


class GeminiModel(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    max_tokens: Optional[int] = None
    tokens_per_message: Optional[int] = None


@router.get("/models")
async def get_models(user=Depends(get_verified_user)) -> List[GeminiModel]:
    """Get available Gemini models"""
    if not ENABLE_GEMINI_API:
        raise HTTPException(status_code=400, detail="Gemini API is not enabled")

    try:
        models = await gemini_client.list_models()
        return [
            GeminiModel(
                id=model["name"].split("/")[-1],
                name=model.get("displayName", model["name"]),
                description=model.get("description", ""),
                max_tokens=model.get("tokenLimit", 30720),
            )
            for model in models
        ]
    except GeminiError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))


class GeminiChatMessage(BaseModel):
    role: str
    content: str


class GeminiChatRequest(BaseModel):
    messages: List[GeminiChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class GeminiChatResponse(BaseModel):
    id: str
    choices: List[Dict]
    model: str
    usage: Dict


@router.post("/chat/completions")
async def chat_completions(
    request: Request, chat_request: GeminiChatRequest, user=Depends(get_verified_user)
) -> GeminiChatResponse:
    """Generate chat completion using Gemini API"""
    if not ENABLE_GEMINI_API:
        raise HTTPException(status_code=400, detail="Gemini API is not enabled")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="Gemini API key is not configured")

    try:
        messages = [
            {"role": msg.role, "content": msg.content} for msg in chat_request.messages
        ]

        if chat_request.stream:

            async def generate():
                try:
                    async for chunk in gemini_client.generate_content(
                        model=chat_request.model,
                        messages=messages,
                        temperature=chat_request.temperature,
                        max_tokens=chat_request.max_tokens,
                        stream=True,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                except GeminiError as e:
                    error_response = {"error": {"message": str(e)}}
                    if hasattr(e, "status_code"):
                        error_response["error"]["code"] = e.status_code
                    yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )
        else:
            response = await gemini_client.generate_content(
                model=chat_request.model,
                messages=messages,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                stream=False,
            )
            return GeminiChatResponse(
                id=f"gemini-{chat_request.model}-{hash(str(messages))}",
                choices=response["choices"],
                model=chat_request.model,
                usage=response["usage"],
            )

    except GeminiError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
