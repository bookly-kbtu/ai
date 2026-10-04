from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Request

from app.api.deps import authenticate
from app.api.v1.schemas import ChatRequest, ChatResponse
from app.core.config import Settings
from app.services.dialog_service import DialogService

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
        state=result.state,
        candidates=result.candidates,
        slots=result.slots,
        booking=result.booking,
    )
