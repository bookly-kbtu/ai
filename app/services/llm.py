"""OpenAI wrapper: system prompt, tool definitions, one completion call.
The LLM only interprets speech and picks tool calls; every real action
(search, slots, booking) is a deterministic Go backend endpoint."""

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from openai import APIError, AsyncOpenAI

from app.core.exceptions import LLMError

ALMATY = ZoneInfo("Asia/Almaty")

WEEKDAYS_RU = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Найти мастеров и услуги по запросу клиента. Возвращает до 5 вариантов "
                "с ценами (price_amount в тиынах: 1000000 = 10000 тг) и длительностью."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Ключевые слова услуги по-русски: 'маникюр', 'мужская стрижка'",
                    },
                    "category_id": {
                        "type": ["string", "null"],
                        "description": "UUID категории из списка в системном промпте, если понятна",
                    },
                    "max_price_kzt": {
                        "type": ["integer", "null"],
                        "description": "Бюджет клиента в тенге (не в тиынах), если назван",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_slots",
            "description": (
                "Свободные слоты мастера на конкретную дату (локальный день Алматы). "
                "Вызывай после того, как клиент заинтересовался вариантом."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "master_id": {"type": "string"},
                    "service_id": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["master_id", "service_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_service",
            "description": "Зафиксировать выбор мастера и услуги для текущего запроса. Обязательно перед book.",
            "parameters": {
                "type": "object",
                "properties": {
                    "master_id": {"type": "string"},
                    "service_id": {"type": "string"},
                },
                "required": ["master_id", "service_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book",
            "description": (
                "Создать запись на выбранный слот. Вызывай ТОЛЬКО после явного подтверждения "
                "клиента (он сказал 'да, записывай' про конкретное время)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "starts_at": {
                        "type": "string",
                        "description": "Время начала из get_slots, строго как в ответе (RFC3339)",
                    },
                    "comment": {"type": ["string", "null"], "description": "Пожелание клиента мастеру"},
                },
                "required": ["starts_at"],
            },
        },
    },
]


def system_prompt(categories: list[dict[str, Any]]) -> str:
    now = datetime.now(ALMATY)
    category_lines = "\n".join(f"- {c['name']}: {c['id']}" for c in categories if c.get("is_active"))
    return f"""Ты — голосовой ассистент Bookly, сервиса записи к бьюти-мастерам в Казахстане.
Твоя задача — довести клиента до записи за минимум шагов. Отвечай по-русски, коротко и живо,
как в разговоре: текст будет озвучен и показан на телефоне. Без markdown, без списков длиннее
трёх пунктов, без UUID в тексте.

Структура каждого ответа: первый абзац — короткая самодостаточная реплика в 1-2 предложения,
именно её озвучат вслух. Детали, цены и перечисления — после пустой строки.

Сейчас {now.strftime('%Y-%m-%d %H:%M')}, {WEEKDAYS_RU[now.weekday()]}, часовой пояс Алматы.
Относительные даты считай от этой: «завтра», «в субботу» и т.п.

Категории услуг (для search_candidates):
{category_lines}

Правила:
1. Понял услугу — сразу search_candidates, не переспрашивай лишнего. Бюджет клиент называет в тенге.
2. Из кандидатов рекомендуй 2-3 лучших и объясни почему (цена, длительность, описание мастера).
   Цены показывай в тенге: price_amount раздели на 100.
3. Клиент выбрал вариант — select_service, затем get_slots на нужную дату и предложи 3-4 удобных
   времени (не перечисляй все слоты).
4. book — только после явного подтверждения конкретного времени клиентом. Не записывай без «да».
5. После записи подтверди: мастер, услуга, дата, время, адрес.
6. Если ничего не нашлось или слотов нет — предложи изменить запрос, дату или бюджет.
7. Ты не можешь отменять записи и отвечать на темы вне записи к мастерам — вежливо возвращай к делу."""


TTS_INSTRUCTIONS = (
    "Тёплый, дружелюбный женский голос ассистента сервиса красоты. "
    "Говори по-русски естественно и живо, в разговорном темпе, без пафоса."
)


class LLMClient:
    def __init__(self, client: AsyncOpenAI, model: str, tts_model: str, tts_voice: str) -> None:
        self._client = client
        self._model = model
        self._tts_model = tts_model
        self._tts_voice = tts_voice

    async def complete(self, messages: list[dict[str, Any]]) -> Any:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                temperature=0.3,
            )
        except APIError as exc:  # covers auth, rate limit, server errors
            raise LLMError(f"OpenAI: {exc}") from exc
        return response.choices[0].message

    async def speak_stream(self, text: str):
        """Neural TTS as an mp3 chunk stream: the browser starts playing
        the first chunks while the tail is still being generated."""
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._tts_model,
                voice=self._tts_voice,
                input=text,
                instructions=TTS_INSTRUCTIONS,
                response_format="mp3",
            ) as response:
                async for chunk in response.iter_bytes():
                    yield chunk
        except APIError as exc:
            raise LLMError(f"OpenAI TTS: {exc}") from exc
