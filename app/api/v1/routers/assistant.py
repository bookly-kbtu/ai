from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.deps import authenticate
from app.api.v1.schemas import ChatRequest, ChatResponse, TtsRequest
from app.core.config import Settings
from app.services.dialog_service import DialogService
from app.services.llm import LLMClient

router = APIRouter(route_class=DishkaRoute, prefix="/ai/v1/assistant", tags=["assistant"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    service: FromDishka[DialogService],
    settings: FromDishka[Settings],
) -> ChatResponse:
    auth = authenticate(request, settings)
    result = await service.chat(auth.user_id, auth.token, body.conversation_id, body.message)
    return ChatResponse(
        conversation_id=result.conversation_id,
        reply=result.reply,
        voice_reply=result.voice_reply,
        state=result.state,
        candidates=result.candidates,
        slots=result.slots,
        booking=result.booking,
    )


@router.post("/tts")
async def tts(
    request: Request,
    body: TtsRequest,
    llm: FromDishka[LLMClient],
    settings: FromDishka[Settings],
) -> StreamingResponse:
    authenticate(request, settings)
    chunks = llm.speak_stream(body.text)
    # Pull the first chunk before streaming starts: provider errors (bad key,
    # rate limit) still surface as a clean 502 instead of a broken stream.
    try:
        first = await anext(chunks)
    except StopAsyncIteration:
        first = b""

    async def body_stream():
        yield first
        async for chunk in chunks:
            yield chunk

    return StreamingResponse(body_stream(), media_type="audio/mpeg")
