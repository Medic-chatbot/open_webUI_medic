import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from open_webui.env import (
    AIOHTTP_CLIENT_TIMEOUT,
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY,
    SRC_LOG_LEVELS,
)
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["GEMINI"])


from .errors import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiContextLengthExceededError,
    GeminiError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    GeminiRateLimitError,
    GeminiServerError,
    GeminiTimeoutError,
)


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.api_base_url = api_base_url or GEMINI_API_BASE_URL
        self.timeout = timeout or AIOHTTP_CLIENT_TIMEOUT or 300

        if not self.api_key:
            raise GeminiError("Gemini API key is required")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        stream: bool = False,
    ) -> aiohttp.ClientResponse:
        """Make a request to the Gemini API"""
        url = f"{self.api_base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                ) as response:
                    if not response.ok:
                        error_data = await response.json()
                        error_info = error_data.get("error", {})
                        error_message = error_info.get("message", "Unknown error")
                        error_code = response.status

                        if error_code == 401:
                            raise GeminiAuthenticationError()
                        elif error_code == 429:
                            raise GeminiRateLimitError(error_message)
                        elif (
                            error_code == 404
                            and "model not found" in error_message.lower()
                        ):
                            model = data.get("model", "unknown") if data else "unknown"
                            raise GeminiModelNotFoundError(model)
                        elif error_code == 400:
                            if "context length exceeded" in error_message.lower():
                                raise GeminiContextLengthExceededError(error_message)
                            raise GeminiInvalidRequestError(error_message)
                        elif error_code >= 500:
                            raise GeminiServerError(error_message)
                        else:
                            raise GeminiAPIError(error_message, status_code=error_code)
                    return response
            except aiohttp.ClientTimeout as e:
                raise GeminiTimeoutError(str(e))
            except aiohttp.ClientError as e:
                raise GeminiError(f"Request failed: {str(e)}")
            except asyncio.TimeoutError:
                raise GeminiTimeoutError()

    async def list_models(self) -> List[Dict]:
        """Get list of available models"""
        response = await self._make_request("GET", "")
        data = await response.json()
        return data.get("models", [])

    async def generate_content(
        self,
        model: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict, None] | Dict:
        """Generate content using Gemini API"""
        data = {
            "contents": [
                {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                for msg in messages
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        endpoint = f"{model}:generateContent"
        if stream:
            endpoint += "?alt=sse"  # Server-Sent Events for streaming

        response = await self._make_request(
            method="POST",
            endpoint=endpoint,
            data=data,
            stream=stream,
        )

        if stream:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode())
                        yield self._process_response(data)
                    except json.JSONDecodeError:
                        continue
        else:
            data = await response.json()
            return self._process_response(data)

    def _process_response(self, data: Dict) -> Dict:
        """Process and format the API response"""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise GeminiError("No response from the model")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise GeminiError("Empty response from the model")

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": parts[0].get("text", ""),
                        },
                        "finish_reason": candidates[0].get("finishReason", "stop"),
                    }
                ],
                "usage": {  # Gemini doesn't provide token counts
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        except (KeyError, IndexError) as e:
            raise GeminiError(f"Failed to process response: {str(e)}")
