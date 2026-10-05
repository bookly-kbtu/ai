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
                "Найти мастеров и услуги по запросу клиента. Возвращает варианты с "
                "ценами в тенге (price_kzt) и длительностью."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "ОДНО ключевое слово — НАЗВАНИЕ УСЛУГИ по-русски: 'стрижка', "
                            "'маникюр', 'брови', 'окрашивание'. Профессию переводи в "
                            "услугу: барбер → 'стрижка', лэшмейкер → 'ресницы'. Название "
                            "салона передавай ТОЛЬКО если клиент явно просит этот салон."
                        ),
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
            "name": "my_bookings",
            "description": (
                "Записи клиента (мой календарь): ближайшие и прошлые брони со статусами. "
                "Вызывай на вопросы вида 'какие у меня записи', 'когда я записан'."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": (
                "Отменить запись клиента. Вызывай ТОЛЬКО после явного подтверждения и только "
                "с id из свежего вызова my_bookings."
            ),
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "booking_slots",
            "description": (
                "Свободные слоты у того же мастера и услуги, что в существующей записи "
                "клиента (для переноса или повторной записи). id — из my_bookings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["booking_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_booking",
            "description": (
                "Перенести запись: создаёт новую бронь на starts_at из booking_slots и "
                "отменяет старую. Только после явного подтверждения клиента."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                    "starts_at": {"type": "string", "description": "RFC3339 из booking_slots"},
                },
                "required": ["booking_id", "starts_at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_again",
            "description": (
                "Повторить запись «как обычно»: новая бронь к тому же мастеру на ту же "
                "услугу на время из booking_slots. Старая запись не трогается."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {"type": "string"},
                    "starts_at": {"type": "string", "description": "RFC3339 из booking_slots"},
                },
                "required": ["booking_id", "starts_at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_favorites",
            "description": "Избранные салоны клиента (вопросы вида 'что у меня в избранном').",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_favorite",
            "description": (
                "Добавить салон из каталога в избранное по названию "
                "('добавь в избранное Laser Lux')."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Название салона"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book",
            "description": (
                "Создать запись на слот из get_slots ТОЛЬКО для нового поиска через "
                "search_candidates. Для переноса или повтора существующей записи используй "
                "reschedule_booking / book_again. Вызывай только после явного «да» клиента."
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
    del categories  # kept in the signature for call-site stability
    return f"""Ты — голосовой ассистент Bookly, сервиса записи к бьюти-мастерам в Казахстане.
Твоя задача — довести клиента до записи за минимум шагов. Отвечай коротко и живо,
как в переписке в мессенджере, НА ЯЗЫКЕ КЛИЕНТА: по-русски или по-казахски.
Без markdown, без списков длиннее трёх пунктов, без UUID в тексте.

Сейчас {now.strftime('%Y-%m-%d %H:%M')}, {WEEKDAYS_RU[now.weekday()]}, часовой пояс Алматы.
Относительные даты считай от этой: «завтра», «в субботу» и т.п.

Правила:
1. Понял услугу — сразу search_candidates, не переспрашивай лишнего. Бюджет клиент называет в тенге.
1а. Клиент назвал конкретный салон или мастера («запиши меня в Barber Janvs») — передай это
   название как query: поиск находит и по имени мастера/салона.
2. Из кандидатов рекомендуй 2-3, у которых НАЗВАНИЕ УСЛУГИ целиком соответствует просьбе.
   Частичные услуги игнорируй: «стрижка чёлки», «стрижка бороды», «стрижка ногтей» — это
   НЕ ответ на «подстричься»; депиляция — не ответ на «барбер». Лучше дороже, но по делу.
   Цены приходят уже в тенге (price_kzt) — называй как есть, не пересчитывай.
3. Клиент выбрал вариант — select_service, затем get_slots на нужную дату и предложи 3-4 удобных
   времени (не перечисляй все слоты).
4. book — только после явного подтверждения конкретного времени клиентом. Не записывай без «да».
5. После записи подтверди: мастер, услуга, дата, время, адрес.
6. Если ничего не нашлось или слотов нет — предложи изменить запрос, дату или бюджет.
7. Спросили про свои записи или календарь — вызови my_bookings и кратко перескажи ближайшие
   актуальные (pending/confirmed); отменённые упоминай только если спросят.
8. Отмена: сначала my_bookings, уточни какую запись, дождись явного «да» и только тогда
   cancel_booking с её id. После отмены подтверди словами.
8а. Перенос («перенеси запись»): my_bookings → уточни какую и на когда → booking_slots на
   нужную дату → предложи время → после «да» reschedule_booking. Старая отменится сама.
   Переносить можно только pending/confirmed; отменённую запись предложи повторить (book_again).
8б. «Запиши как обычно/как всегда»: my_bookings → определи привычную услугу (чаще всего
   или последняя завершённая) → уточни день → booking_slots → после «да» book_again.
9. Избранное: list_favorites показывает список, add_favorite добавляет салон из каталога
   по названию. Записаться в салоны из избранного нельзя — они из внешнего каталога.
10. Не отвечай на темы вне записи и красоты — вежливо возвращай к делу."""


class LLMClient:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

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

