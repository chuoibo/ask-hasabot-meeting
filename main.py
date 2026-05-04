import asyncio
import json
import time
import hashlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import  StreamingResponse

from hasa_bot.domain.meeting.meeting_context_service import get_meeting_context_service

from hasa_core.utils.logger import setup_logger


_logger = setup_logger("logs/hasabot_voicebot_v2.log")

_ASSIGNMENT_SUGGESTION_BUYTIME_INITIAL_DELAY = 0.9
_ASSIGNMENT_SUGGESTION_BUYTIME_CHAR_DELAY = 0.045

STREAM_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

router = APIRouter()

def _validate_assignment_suggestion_payload(
    *,
    session_id: str,
    event_id: str,
    suggestion: str,
    selected_text: str,
    question_text: str,
) -> None:
    if not str(session_id or "").strip():
        raise HTTPException(status_code=400, detail="session_id is required.")
    if not str(event_id or "").strip():
        raise HTTPException(status_code=400, detail="event_id is required.")
    if not str(suggestion or "").strip():
        raise HTTPException(status_code=400, detail="suggestion is required.")
    if not str(selected_text or "").strip():
        raise HTTPException(status_code=400, detail="selected_text is required.")
    if not str(question_text or "").strip():
        raise HTTPException(status_code=400, detail="question_text is required.")


def format_json_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _compact_assignment_text(value: Optional[str], *, limit: int = 72) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def normalize_audio_language(language: Optional[str]) -> str:
    if not isinstance(language, str):
        return "vi"
    normalized = (language or "vi").strip().lower()
    return normalized if normalized in ("vi", "en") else "vi"


def _build_assignment_suggestion_buytime_text(
    *,
    session_id: str,
    event_id: str,
    language: str,
    topic: str,
    selected_text: str,
    task_description: str,
    question_text: str,
) -> str:
    normalized_language = normalize_audio_language(language)
    topic_focus = _compact_assignment_text(topic, limit=44)
    selected_focus = _compact_assignment_text(selected_text, limit=52)
    task_focus = _compact_assignment_text(task_description, limit=52)
    question_focus = _compact_assignment_text(question_text, limit=52)

    seed_source = "||".join(
        [
            str(session_id or ""),
            str(event_id or ""),
            str(selected_focus or ""),
            str(question_focus or ""),
        ]
    )
    seed_value = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)

    if normalized_language == "en":
        openings = [
            "I’m reviewing the exact passage you highlighted so I can answer this precisely.",
            "Give me a moment while I cross-check this passage against the assignment context.",
            "I’m tightening the meaning of this selected passage before I answer you.",
            "I’m re-reading the relevant part so I can clarify it without changing the rest.",
        ]
        focus_options = []
        if topic_focus:
            focus_options.append(f"I’m focusing on the part related to {topic_focus}.")
        if selected_focus:
            focus_options.append("I’m narrowing this to the exact part you highlighted.")
        if task_focus:
            focus_options.append("I’m comparing it with the current task description so the clarification stays aligned.")
        focus_options.append("I’m keeping the rest of the suggestion intact and only clarifying the part you asked about.")
    else:
        openings = [
            "Mình đang rà lại đúng đoạn bạn vừa chọn để trả lời sát câu hỏi này.",
            "Để mình đối chiếu lại đoạn này với ngữ cảnh giao việc rồi làm rõ cho bạn ngay.",
            "Mình đang đọc kỹ lại phần bạn bôi chọn để giải thích đúng ý hơn.",
            "Mình đang siết lại ý của đoạn này trước khi trả lời để bạn dễ hiểu hơn.",
        ]
        focus_options = []
        if topic_focus:
            focus_options.append(f"Mình đang tập trung vào phần liên quan đến {topic_focus}.")
        if selected_focus:
            focus_options.append("Mình sẽ chỉ làm rõ đúng phần bạn vừa bôi chọn.")
        if task_focus:
            focus_options.append("Mình đang đối chiếu lại với mô tả công việc để phần giải thích khớp hơn.")
        focus_options.append("Mình sẽ giữ nguyên các phần còn lại và chỉ làm rõ chỗ bạn đang hỏi.")

    opening = openings[seed_value % len(openings)]
    followup = focus_options[(seed_value // max(1, len(openings))) % len(focus_options)]
    return f"{opening} {followup}".strip()

def _iter_assignment_buytime_chars(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    return list(normalized)

@router.post("/api/v2/meeting-summary/assignment-suggestion/ask")
async def meeting_summary_assignment_suggestion_ask_post(
    session_id: str = Body(...),
    event_id: str = Body(...),
    source_hash: str = Body(""),
    language: str = Body("vi"),
    topic: str = Body(""),
    report_detail: str = Body(""),
    detail_points: List[str] = Body(default=[]),
    summary: str = Body(""),
    task_description: str = Body(""),
    suggestion: str = Body(...),
    selected_text: str = Body(...),
    question_text: str = Body(...),
    conversation_history: List[Dict[str, Any]] = Body(default=[]),
    assigner: str = Body(""),
    assignee: str = Body(""),
    start_date: str = Body(""),
    deadline: str = Body("")
):
    _validate_assignment_suggestion_payload(
        session_id=session_id,
        event_id=event_id,
        suggestion=suggestion,
        selected_text=selected_text,
        question_text=question_text,
    )
    meeting_context_service = await get_meeting_context_service()

    async def event_generator():
        stream_started_at = time.monotonic()
        answer_parts: list[str] = []
        answer_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        first_answer_chunk_ms: Optional[int] = None
        buytime_decision = "pending"
        buytime_chars_emitted = 0

        async def produce_answer() -> None:
            nonlocal first_answer_chunk_ms
            try:
                async for chunk in meeting_context_service.stream_assignment_suggestion_answer(
                    session_id=session_id,
                    event_id=event_id,
                    source_hash=source_hash,
                    language=language,
                    topic=topic,
                    report_detail=report_detail,
                    detail_points=list(detail_points or []),
                    summary=summary,
                    task_description=task_description,
                    suggestion=suggestion,
                    selected_text=selected_text,
                    question_text=question_text,
                    conversation_history=list(conversation_history or []),
                    assigner=assigner,
                    assignee=assignee,
                    start_date=start_date,
                    deadline=deadline,
                ):
                    if not chunk:
                        continue
                    if first_answer_chunk_ms is None:
                        first_answer_chunk_ms = int((time.monotonic() - stream_started_at) * 1000)
                        _logger.info(
                            "Meeting assignment suggestion ask first answer chunk: session_id={} event_id={} first_chunk_ms={}",
                            session_id,
                            event_id,
                            first_answer_chunk_ms,
                        )
                    await answer_queue.put(("delta", chunk))
            except Exception as exc:
                await answer_queue.put(("error", exc))
            finally:
                await answer_queue.put(("eos", None))

        producer_task = asyncio.create_task(produce_answer())
        yield format_json_sse({"type": "start"})
        try:
            answer_started = False
            current_item: Optional[tuple[str, Any]] = None

            try:
                current_item = await asyncio.wait_for(
                    answer_queue.get(),
                    timeout=_ASSIGNMENT_SUGGESTION_BUYTIME_INITIAL_DELAY,
                )
                buytime_decision = "skipped"
                yield format_json_sse(
                    {"type": "status", "phase": "buytime", "state": "skipped"}
                )
            except asyncio.TimeoutError:
                buytime_decision = "shown"
                yield format_json_sse(
                    {"type": "status", "phase": "buytime", "state": "started"}
                )
                buytime_text = _build_assignment_suggestion_buytime_text(
                    session_id=session_id,
                    event_id=event_id,
                    language=language,
                    topic=topic,
                    selected_text=selected_text,
                    task_description=task_description,
                    question_text=question_text,
                )
                for buytime_char in _iter_assignment_buytime_chars(buytime_text):
                    buytime_chars_emitted += len(buytime_char)
                    yield format_json_sse({"type": "buytime", "delta": buytime_char})
                    await asyncio.sleep(_ASSIGNMENT_SUGGESTION_BUYTIME_CHAR_DELAY)
                if current_item is None:
                    current_item = await answer_queue.get()
                yield format_json_sse(
                    {"type": "status", "phase": "buytime", "state": "completed"}
                )

            while True:
                item_type, item_value = current_item if current_item is not None else await answer_queue.get()
                current_item = None

                if item_type == "delta":
                    if not answer_started:
                        answer_started = True
                        yield format_json_sse(
                            {"type": "status", "phase": "answer", "state": "started"}
                        )
                    answer_parts.append(item_value)
                    yield format_json_sse({"type": "delta", "delta": item_value})
                    continue

                if item_type == "error":
                    raise item_value

                if item_type == "eos":
                    break

            final_text = "".join(answer_parts)
            if not final_text.strip():
                raise ValueError("Empty assignment suggestion answer")
            yield format_json_sse({"type": "done", "text": final_text})
        except Exception as exc:
            _logger.exception(
                "Meeting assignment suggestion ask failed: session_id={} event_id={}",
                session_id,
                event_id,
            )
            yield format_json_sse(
                {
                    "type": "error",
                    "message": str(exc) or "Assignment suggestion ask failed.",
                }
            )
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
            _logger.info(
                "Meeting assignment suggestion ask stream completed: session_id={} event_id={} buytime_decision={} buytime_chars={} first_answer_chunk_ms={} total_duration_ms={}",
                session_id,
                event_id,
                buytime_decision,
                buytime_chars_emitted,
                "" if first_answer_chunk_ms is None else first_answer_chunk_ms,
                int((time.monotonic() - stream_started_at) * 1000),
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=STREAM_RESPONSE_HEADERS,
    )
