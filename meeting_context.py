"""Meeting-context summarization and retrieval for Hasabot voice v2."""

from __future__ import annotations

import asyncio
import audioop
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import HTTPException
from google.genai import types

from utils import (
    build_meeting_event_signature,
    clamp_meeting_confidence,
    coerce_int_or_default,
    coerce_optional_float,
    enforce_manual_roster_on_topic_updates,
    format_topic_clusters_summary,
    normalize_manual_participants,
    normalize_meeting_event_type,
    normalize_meeting_phase,
    normalize_meeting_detail_key,
    normalize_meeting_detail_points,
    normalize_meeting_topic_key,
    normalize_topic_clusters_payload,
    sanitize_meeting_deadline,
    sanitize_meeting_owner,
)

async def _noop() -> None:
    return None

_service: Optional["MeetingContextService"] = None
_init_lock = asyncio.Lock()
# session_id → WebM EBML init bytes (everything before the first Cluster).
# Used to patch seg 1+ audio so both Gemini and debug WAV receive a valid container.
_webm_init_cache: dict[str, bytes] = {}
_BACKGROUND_KEYPOINT_RETRY_DELAY_SECONDS = 2
_BACKGROUND_KEYPOINT_MAX_RETRIES = 3
_BACKGROUND_KEYPOINT_THINKING_BUDGET = -1
_DISPLAY_CONFIDENCE_THRESHOLD = 0.40
_ACTION_ITEM_CONFIDENCE_THRESHOLD = 0.55
_ASSIGNMENT_SUGGESTION_HISTORY_MAX_TURNS = 8
_ASSIGNMENT_SUGGESTION_HISTORY_MAX_CHARS = 6000
_ASSIGNMENT_SUGGESTION_HISTORY_ITEM_MAX_CHARS = 1600
_background_segment_locks: dict[str, asyncio.Lock] = {}


class MeetingContextService:
    """Create and read rolling meeting context for voice meeting assistant flows."""

    def __init__(self) -> None:
        # self.store = MeetingContextStore()
        self.model_name = "gemini-2.5-flash-lite"

    async def initialize(self) -> None:
        await _noop()

 
    @staticmethod
    def _build_assignment_document_query(
        *,
        topic: str,
        report_detail: str,
        summary: str,
        task_description: str,
        suggestion: str,
        selected_text: str,
        question_text: str,
    ) -> str:
        parts = [
            f"User question: {question_text}",
            f"Selected passage: {selected_text}",
            f"Topic: {topic}",
            f"Report detail: {report_detail}",
            f"Summary: {summary}",
            f"Task description: {task_description}",
            f"Current suggestion: {suggestion}",
        ]
        clean_parts = [
            re.sub(r"\s+", " ", str(part or "").strip())
            for part in parts
            if str(part or "").strip()
        ]
        return "\n".join(clean_parts)[:5000]

    async def stream_assignment_suggestion_answer(
        self,
        *,
        session_id: str,
        event_id: str,
        source_hash: str,
        language: str,
        topic: str,
        report_detail: str,
        detail_points: list[str],
        summary: str,
        task_description: str,
        suggestion: str,
        selected_text: str,
        question_text: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        assigner: str = "",
        assignee: str = "",
        start_date: str = "",
        deadline: str = "",
    ) -> AsyncIterator[str]:
 
        normalized_language = str(language or "vi").strip().lower() or "vi"
        meeting_brief = {}

        system_prompt = self._get_assignment_suggestion_ask_prompt(normalized_language)
        normalized_history = self._normalize_assignment_suggestion_conversation_history(
            conversation_history
        )
        user_prompt = self._build_assignment_suggestion_context_input(
            session_id=session_id,
            event_id=event_id,
            source_hash=source_hash,
            topic=topic,
            report_detail=report_detail,
            detail_points=detail_points,
            summary=summary,
            task_description=task_description,
            suggestion=suggestion,
            selected_text=selected_text,
            question_text=question_text,
            conversation_history=normalized_history,
            assigner=assigner,
            assignee=assignee,
            start_date=start_date,
            deadline=deadline,
            language=normalized_language,
            meeting_brief=meeting_brief,
        )
        print(
            "Assignment suggestion ask agent stream start: session_id={} event_id={} source_hash={} history_turns={} prompt_chars={} uploaded_docs={}",
            session_id,
            event_id,
            source_hash[:12],
            len(normalized_history),
            len(user_prompt)
        )

        from suggestion import get_assignment_suggestion_ask_agent

        agent = get_assignment_suggestion_ask_agent()
        async for chunk in agent.stream_answer(
            session_id=session_id,
            language=normalized_language,
            base_prompt=system_prompt,
            context_input=user_prompt,
            # document_query_context=document_query_context,
        ):
            if chunk:
                yield chunk

    # The original prompt
    # @staticmethod
    # def _get_assignment_suggestion_ask_prompt(language: str) -> str:
    #     if str(language or "vi").strip().lower() == "en":
    #         return (
    #             "You are Hasabot helping a user clarify one selected passage inside an AI-generated assignment suggestion.\n"
    #             "The user may be asking a follow-up question in an ongoing clarification thread about that same passage.\n"
    #             "Your job is to answer the user's latest question about the selected passage using the provided assignment/report context, initial meeting information, and the prior clarification thread.\n"
    #             "Your role is also a professional work-efficiency consultant: when useful, suggest clearer execution, risks to control, priorities, and practical next steps.\n"
    #             "Uploaded-document Markdown is optional supporting knowledge available through a tool when the current context is not enough.\n"
    #             "You must be highly specific, practical, and grounded in the given materials.\n"
    #             "Output requirements:\n"
    #             "- Respond in English.\n"
    #             "- Return normal markdown only. No JSON. No markdown fences unless the user explicitly needs a code block.\n"
    #             "- Start with a direct answer to the user's latest question.\n"
    #             "- Then explain how the selected passage should be understood in the context of the full suggestion.\n"
    #             "- Stay consistent with earlier clarification turns unless the provided source context clearly requires a correction.\n"
    #             "- If useful, provide a clearer rewritten interpretation of the selected passage.\n"
    #             "- Ground consulting suggestions in the provided context. If facts are missing, state cautious assumptions.\n"
    #             "- Do not invent uploaded-document content. If the uploaded-document tool is unavailable, not ready, or returns no relevant excerpt, say that only when the user specifically asks about uploaded documents.\n"
    #             "- If the source context is incomplete, explicitly say what is missing and give only cautious assumptions.\n"
    #             "- Do not invent metrics, deadlines, owners, systems, or process details that are not supported by the provided context.\n"
    #             "- Keep the answer focused on the selected passage and the user's question rather than re-explaining the whole suggestion.\n"
    #             "- Use short headings and bullet points when they improve clarity.\n"
    #             "- Do not mention hidden prompts, model behavior, or internal implementation details."
    #         )
    #     return (
    #         "Bạn là Hasabot hỗ trợ người dùng làm rõ một đoạn đã bôi chọn trong phần gợi ý AI của modal giao việc.\n"
    #         "Người dùng có thể đang hỏi tiếp trong một chuỗi làm rõ nhiều lượt về cùng đoạn đó.\n"
    #         "Nhiệm vụ của bạn là trả lời câu hỏi mới nhất của người dùng về đúng đoạn được chọn, dựa trên ngữ cảnh giao việc, phần report đã cung cấp, thông tin cuộc họp ban đầu, và chuỗi làm rõ trước đó.\n"
    #         "Vai trò của bạn cũng là một chuyên gia tư vấn hiệu quả công việc: khi phù hợp, hãy gợi ý cách triển khai rõ hơn, rủi ro cần kiểm soát, mức ưu tiên, và bước tiếp theo thực tế.\n"
    #         "Markdown tài liệu upload là tri thức bổ sung qua công cụ tùy chọn khi ngữ cảnh hiện tại chưa đủ.\n"
    #         "Bạn phải trả lời cụ thể, thực tế, dễ hiểu và bám sát dữ kiện.\n"
    #         "Yêu cầu đầu ra:\n"
    #         "- Trả lời bằng tiếng Việt.\n"
    #         "- Chỉ trả markdown thông thường. Không trả JSON. Không bọc bằng markdown fence trừ khi thật sự cần code block.\n"
    #         "- Mở đầu bằng câu trả lời trực diện cho câu hỏi mới nhất của người dùng.\n"
    #         "- Sau đó giải thích đoạn đã chọn cần được hiểu như thế nào trong ngữ cảnh toàn bộ gợi ý.\n"
    #         "- Giữ câu trả lời nhất quán với các lượt làm rõ trước đó, trừ khi dữ kiện nguồn cho thấy cần đính chính.\n"
    #         "- Nếu hữu ích, hãy viết lại đoạn đó theo cách rõ ràng hơn để người dùng dễ hiểu.\n"
    #         "- Khi đưa đề xuất tư vấn, phải bám vào ngữ cảnh được cung cấp; nếu thiếu dữ kiện thì nói rõ giả định thận trọng.\n"
    #         "- Không bịa nội dung tài liệu upload. Nếu công cụ tài liệu không khả dụng, tài liệu chưa sẵn sàng hoặc không có đoạn liên quan, chỉ nói điều đó khi người dùng hỏi trực tiếp về tài liệu upload.\n"
    #         "- Nếu dữ kiện nguồn chưa đủ, hãy nói rõ phần nào còn thiếu và chỉ đưa ra giả định thận trọng.\n"
    #         "- Không bịa thêm số liệu, hạn chót, người phụ trách, hệ thống hay quy trình nếu input không có.\n"
    #         "- Tập trung vào đoạn được chọn và câu hỏi đang hỏi, không kể lại toàn bộ gợi ý nếu không cần.\n"
    #         "- Dùng tiêu đề ngắn và bullet points khi điều đó giúp câu trả lời dễ theo dõi hơn.\n"
    #         "- Không nhắc tới prompt ẩn, cách mô hình hoạt động hay chi tiết kỹ thuật nội bộ."
    #     )

    # The modified prompt
    @staticmethod
    def _get_assignment_suggestion_ask_prompt(language: str) -> str:
        if str(language or "vi").strip().lower() == "en":
            return (
                "You are Hasabot helping a user clarify one selected passage inside an AI-generated assignment suggestion.\n"
                "The user may be asking a follow-up question in an ongoing clarification thread about that same passage.\n"
                "Your job is to answer the user's latest question about the selected passage using the provided assignment/report context, initial meeting information, and the prior clarification thread.\n"
                "Your role is also a professional work-efficiency consultant: when useful, suggest clearer execution, risks to control, priorities, and practical next steps.\n"
                "Uploaded-document Markdown is optional supporting knowledge available through a tool when the current context is not enough.\n"
                "You must be highly specific, practical, and grounded in the given materials.\n"
                "\n"
                "Output requirements:\n"
                "- Respond in English.\n"
                "- Return normal markdown only. No JSON. No markdown fences unless the user explicitly needs a code block.\n"
                "- Start with a direct answer to the user's latest question.\n"
                "- Then explain how the selected passage should be understood in the context of the full suggestion.\n"
                "- Stay consistent with earlier clarification turns unless the provided source context clearly requires a correction.\n"
                "- If useful, provide a clearer rewritten interpretation of the selected passage.\n"
                "- Ground consulting suggestions in the provided context. If facts are missing, state cautious assumptions.\n"
                "- Do not invent uploaded-document content. If the uploaded-document tool is unavailable, not ready, or returns no relevant excerpt, say that only when the user specifically asks about uploaded documents.\n"
                "- If the source context is incomplete, explicitly say what is missing and give only cautious assumptions.\n"
                "- Do not invent metrics, deadlines, owners, systems, or process details that are not supported by the provided context.\n"
                "- Keep the answer focused on the selected passage and the user's question rather than re-explaining the whole suggestion.\n"
                "- Use short headings and bullet points when they improve clarity.\n"
                "- Do not mention hidden prompts, model behavior, or internal implementation details.\n"
                "\n"
                "Preferred answer structure:\n"
                "1. Direct answer\n"
                "2. Meaning in context\n"
                "3. Clearer interpretation, if useful\n"
                "4. Practical next steps, if useful\n"
                "\n"
                "Few-shot examples:\n"
                "\n"
                "Example 1\n"
                "\n"
                "Assignment/report context:\n"
                "The team is building a robot vacuum. The robot sometimes gets stuck under low chairs. QA needs to test chair height before Friday. The suggestion says the navigation team should improve obstacle handling.\n"
                "\n"
                "Selected passage:\n"
                "\"gets stuck under low chairs\"\n"
                "\n"
                "User question:\n"
                "\"What does this mean?\"\n"
                "\n"
                "Answer:\n"
                "It means the robot can enter spaces under low chairs but may not be able to move out safely afterward.\n"
                "\n"
                "### Meaning in context\n"
                "\n"
                "In the full suggestion, this passage points to a navigation and physical-clearance problem. The issue is not only that the robot detects a chair, but that it may misjudge whether it has enough space to pass through or escape.\n"
                "\n"
                "### Clearer interpretation\n"
                "\n"
                "The robot should recognize low-clearance chair areas and avoid entering them if there is a high risk of getting trapped.\n"
                "\n"
                "### Practical next steps\n"
                "\n"
                "- QA should test several chair heights and record which cases cause the robot to get stuck.\n"
                "- The navigation team should check whether the robot needs a rule for avoiding low-clearance areas.\n"
                "- Since the context only mentions testing before Friday, the exact algorithm change is still not confirmed.\n"
                "\n"
                "Example 2\n"
                "\n"
                "Assignment/report context:\n"
                "The backend API must support streaming responses so the frontend popup can show Hasabot answers progressively while the answer is still being generated. The suggestion says backend should update the API response mechanism.\n"
                "\n"
                "Selected passage:\n"
                "\"streaming responses\"\n"
                "\n"
                "User question:\n"
                "\"Streaming here means what?\"\n"
                "\n"
                "Answer:\n"
                "Streaming here means the backend sends the answer in smaller chunks while it is being generated, instead of waiting until the full answer is complete.\n"
                "\n"
                "### Meaning in context\n"
                "\n"
                "In the full suggestion, this passage refers to the response behavior of the API. The frontend popup should be able to display partial Hasabot output progressively, which improves the user experience when the answer takes time to generate.\n"
                "\n"
                "### Clearer interpretation\n"
                "\n"
                "The backend should return partial text chunks to the frontend as soon as they are available, so the user can see the answer appear gradually.\n"
                "\n"
                "### Practical next steps\n"
                "\n"
                "- Confirm the expected streaming format between backend and frontend.\n"
                "- Test whether the frontend can render partial chunks correctly.\n"
                "- Avoid assuming a specific protocol such as SSE or WebSocket unless the project context explicitly states it.\n"
                "\n"
                "Now answer the real user case using the same behavior and structure."
            )

        return (
            "Bạn là Hasabot hỗ trợ người dùng làm rõ một đoạn đã bôi chọn trong phần gợi ý AI của modal giao việc.\n"
            "Người dùng có thể đang hỏi tiếp trong một chuỗi làm rõ nhiều lượt về cùng đoạn đó.\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi mới nhất của người dùng về đúng đoạn được chọn, dựa trên ngữ cảnh giao việc, phần report đã cung cấp, thông tin cuộc họp ban đầu, và chuỗi làm rõ trước đó.\n"
            "Vai trò của bạn cũng là một chuyên gia tư vấn hiệu quả công việc: khi phù hợp, hãy gợi ý cách triển khai rõ hơn, rủi ro cần kiểm soát, mức ưu tiên, và bước tiếp theo thực tế.\n"
            "Markdown tài liệu upload là tri thức bổ sung qua công cụ tùy chọn khi ngữ cảnh hiện tại chưa đủ.\n"
            "Bạn phải trả lời cụ thể, thực tế, dễ hiểu và bám sát dữ kiện.\n"
            "\n"
            "Yêu cầu đầu ra:\n"
            "- Trả lời bằng tiếng Việt.\n"
            "- Chỉ trả markdown thông thường. Không trả JSON. Không bọc bằng markdown fence trừ khi thật sự cần code block.\n"
            "- Mở đầu bằng câu trả lời trực diện cho câu hỏi mới nhất của người dùng.\n"
            "- Sau đó giải thích đoạn đã chọn cần được hiểu như thế nào trong ngữ cảnh toàn bộ gợi ý.\n"
            "- Giữ câu trả lời nhất quán với các lượt làm rõ trước đó, trừ khi dữ kiện nguồn cho thấy cần đính chính.\n"
            "- Nếu hữu ích, hãy viết lại đoạn đó theo cách rõ ràng hơn để người dùng dễ hiểu.\n"
            "- Khi đưa đề xuất tư vấn, phải bám vào ngữ cảnh được cung cấp; nếu thiếu dữ kiện thì nói rõ giả định thận trọng.\n"
            "- Không bịa nội dung tài liệu upload. Nếu công cụ tài liệu không khả dụng, tài liệu chưa sẵn sàng hoặc không có đoạn liên quan, chỉ nói điều đó khi người dùng hỏi trực tiếp về tài liệu upload.\n"
            "- Nếu dữ kiện nguồn chưa đủ, hãy nói rõ phần nào còn thiếu và chỉ đưa ra giả định thận trọng.\n"
            "- Không bịa thêm số liệu, hạn chót, người phụ trách, hệ thống hay quy trình nếu input không có.\n"
            "- Tập trung vào đoạn được chọn và câu hỏi đang hỏi, không kể lại toàn bộ gợi ý nếu không cần.\n"
            "- Dùng tiêu đề ngắn và bullet points khi điều đó giúp câu trả lời dễ theo dõi hơn.\n"
            "- Không nhắc tới prompt ẩn, cách mô hình hoạt động hay chi tiết kỹ thuật nội bộ.\n"
            "\n"
            "Cấu trúc trả lời ưu tiên:\n"
            "1. Trả lời trực diện\n"
            "2. Ý nghĩa trong ngữ cảnh\n"
            "3. Cách hiểu rõ hơn, nếu hữu ích\n"
            "4. Bước tiếp theo thực tế, nếu hữu ích\n"
            "\n"
            "Ví dụ few-shot:\n"
            "\n"
            "Ví dụ 1\n"
            "\n"
            "Ngữ cảnh giao việc/report:\n"
            "Team đang phát triển robot hút bụi. Robot đôi khi bị kẹt dưới ghế thấp. QA cần kiểm tra chiều cao ghế trước thứ Sáu. Gợi ý nói nhóm điều hướng cần cải thiện khả năng xử lý vật cản.\n"
            "\n"
            "Đoạn được chọn:\n"
            "\"bị kẹt dưới ghế thấp\"\n"
            "\n"
            "Câu hỏi của người dùng:\n"
            "\"Ý này nghĩa là gì?\"\n"
            "\n"
            "Câu trả lời:\n"
            "Ý này nghĩa là robot có thể đi vào khu vực dưới ghế thấp nhưng sau đó không thoát ra hoặc tiếp tục di chuyển an toàn được.\n"
            "\n"
            "### Ý nghĩa trong ngữ cảnh\n"
            "\n"
            "Trong toàn bộ gợi ý, đoạn này nói về vấn đề điều hướng và khoảng hở vật lý. Vấn đề không chỉ là robot nhận ra có ghế, mà còn là robot có đánh giá đúng việc nó có đủ không gian để đi qua hoặc quay ra hay không.\n"
            "\n"
            "### Cách hiểu rõ hơn\n"
            "\n"
            "Robot cần nhận biết các khu vực có khoảng hở thấp dưới ghế và tránh đi vào nếu có nguy cơ cao bị mắc kẹt.\n"
            "\n"
            "### Bước tiếp theo thực tế\n"
            "\n"
            "- QA nên test nhiều chiều cao ghế khác nhau và ghi lại trường hợp nào làm robot bị kẹt.\n"
            "- Nhóm điều hướng nên kiểm tra có cần thêm rule tránh khu vực gầm thấp hay không.\n"
            "- Vì ngữ cảnh chỉ nói cần kiểm tra trước thứ Sáu, chưa nên khẳng định cụ thể thuật toán sẽ sửa theo cách nào.\n"
            "\n"
            "Ví dụ 2\n"
            "\n"
            "Ngữ cảnh giao việc/report:\n"
            "API backend cần hỗ trợ streaming response để popup frontend có thể hiển thị câu trả lời của Hasabot từng phần trong lúc nội dung đang được sinh ra. Gợi ý nói backend cần cập nhật cơ chế trả response của API.\n"
            "\n"
            "Đoạn được chọn:\n"
            "\"streaming response\"\n"
            "\n"
            "Câu hỏi của người dùng:\n"
            "\"Streaming ở đây nghĩa là gì?\"\n"
            "\n"
            "Câu trả lời:\n"
            "Streaming ở đây nghĩa là backend gửi câu trả lời thành từng phần nhỏ trong lúc nội dung đang được tạo, thay vì đợi tạo xong toàn bộ rồi mới trả về một lần.\n"
            "\n"
            "### Ý nghĩa trong ngữ cảnh\n"
            "\n"
            "Trong toàn bộ gợi ý, đoạn này nói về cách API trả dữ liệu về cho frontend. Popup frontend cần hiển thị dần nội dung Hasabot để người dùng thấy phản hồi sớm hơn khi câu trả lời mất thời gian sinh ra.\n"
            "\n"
            "### Cách hiểu rõ hơn\n"
            "\n"
            "Backend nên trả từng chunk nội dung cho frontend ngay khi có, để người dùng thấy câu trả lời xuất hiện dần trên popup.\n"
            "\n"
            "### Bước tiếp theo thực tế\n"
            "\n"
            "- Cần thống nhất format streaming giữa backend và frontend.\n"
            "- Cần test frontend có render đúng từng chunk nội dung hay không.\n"
            "- Không nên tự giả định dùng SSE hay WebSocket nếu ngữ cảnh dự án chưa nói rõ.\n"
            "\n"
            "Bây giờ hãy trả lời case thật của người dùng với cùng hành vi và cấu trúc như trên."
        )

    @staticmethod
    def _get_assignment_suggestion_update_prompt(language: str) -> str:
        if str(language or "vi").strip().lower() == "en":
            return (
                "You revise one AI-generated assignment suggestion after a user asked for clarification about a selected passage.\n"
                "You will receive the current full suggestion, the selected passage, the user's latest question, the full clarification thread so far, the latest clarification answer, and the full assignment/report context with initial meeting information.\n"
                "Your job is to update the current suggestion so it becomes clearer exactly where the user asked, while preserving all unrelated content.\n"
                "Output requirements:\n"
                "- Respond in English.\n"
                "- Output the full updated suggestion markdown only.\n"
                "- Do not wrap the answer in markdown fences.\n"
                "- Preserve the existing structure, section order, and unrelated bullets whenever possible.\n"
                "- Modify only the part that needs clarification or the smallest surrounding area required to make the suggestion clearer and more useful.\n"
                "- Use the full clarification thread to understand what the user still found unclear.\n"
                "- Integrate the clarification naturally into the suggestion instead of appending a detached Q&A note.\n"
                "- Do not rewrite the entire suggestion unless the selected passage makes that unavoidable.\n"
                "- Do not add invented facts. If context is still incomplete, keep the wording cautious.\n"
                "- Keep the updated suggestion practical, assignment-ready, and easy to act on."
            )
        return (
            "Bạn chỉnh lại một phần gợi ý AI trong modal giao việc sau khi người dùng hỏi làm rõ một đoạn đã chọn.\n"
            "Bạn sẽ nhận được toàn bộ gợi ý hiện tại, đoạn được bôi chọn, câu hỏi mới nhất của người dùng, toàn bộ chuỗi làm rõ trước đó, câu trả lời làm rõ mới nhất, cùng toàn bộ ngữ cảnh report/giao việc và thông tin cuộc họp ban đầu.\n"
            "Nhiệm vụ của bạn là cập nhật gợi ý hiện tại để phần người dùng hỏi trở nên rõ ràng hơn, nhưng vẫn giữ nguyên những phần không liên quan.\n"
            "Yêu cầu đầu ra:\n"
            "- Trả lời bằng tiếng Việt.\n"
            "- Chỉ xuất ra toàn bộ nội dung gợi ý đã được cập nhật, ở dạng markdown hoàn chỉnh.\n"
            "- Không bọc nội dung trong markdown fence.\n"
            "- Giữ nguyên cấu trúc, thứ tự các phần, và các bullet không liên quan khi có thể.\n"
            "- Chỉ chỉnh đúng phần cần làm rõ hoặc vùng lân cận nhỏ nhất cần thiết để nội dung mạch lạc hơn.\n"
            "- Dùng toàn bộ chuỗi làm rõ để hiểu chính xác phần nào người dùng còn chưa rõ.\n"
            "- Tích hợp phần làm rõ vào ngay trong gợi ý thay vì thêm một mục hỏi đáp rời rạc.\n"
            "- Không viết lại toàn bộ gợi ý nếu không thật sự cần.\n"
            "- Không bịa thêm dữ kiện. Nếu ngữ cảnh vẫn thiếu, hãy giữ cách diễn đạt thận trọng.\n"
            "- Kết quả cuối cùng phải thực tế, dễ hành động và vẫn phù hợp để dùng cho giao việc."
        )

    @staticmethod
    def _format_assignment_meeting_brief_context(
        meeting_brief: Optional[dict[str, Any]],
        *,
        language: str,
    ) -> str:
        brief = meeting_brief or {}
        is_english = str(language or "vi").strip().lower() == "en"
        labels = (
            ("Title", "Objectives", "Description")
            if is_english
            else ("Tiêu đề", "Mục tiêu", "Mô tả")
        )
        values = [
            str(brief.get("title") or "").strip(),
            str(brief.get("objectives") or "").strip(),
            str(brief.get("description") or "").strip(),
        ]
        lines = [
            f"{label}: {value}"
            for label, value in zip(labels, values)
            if value
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_assignment_suggestion_context_input(
        *,
        session_id: str,
        event_id: str,
        source_hash: str,
        topic: str,
        report_detail: str,
        detail_points: list[str],
        summary: str,
        task_description: str,
        suggestion: str,
        selected_text: str,
        question_text: str,
        language: str,
        answer_text: str = "",
        conversation_history: Optional[list[dict[str, Any]]] = None,
        assigner: str = "",
        assignee: str = "",
        start_date: str = "",
        deadline: str = "",
        meeting_brief: Optional[dict[str, Any]] = None,
    ) -> str:
        normalized_language = str(language or "vi").strip().lower() or "vi"
        normalized_points = normalize_meeting_detail_points(detail_points)
        normalized_history = MeetingContextService._normalize_assignment_suggestion_conversation_history(
            conversation_history
        )
        brief_context = MeetingContextService._format_assignment_meeting_brief_context(
            meeting_brief,
            language=normalized_language,
        )

        def fmt(value: str, *, empty_vi: str = "(để trống)", empty_en: str = "(empty)") -> str:
            clean = MeetingContextService._normalize_assignment_suggestion_text(value)
            if clean:
                return clean
            return empty_en if normalized_language == "en" else empty_vi

        if normalized_language == "en":
            lines = [
                "Assignment refinement context",
                f"Session ID: {fmt(session_id, empty_vi='', empty_en='(missing)')}",
                f"Event ID: {fmt(event_id, empty_vi='', empty_en='(missing)')}",
                f"Source hash: {fmt(source_hash, empty_vi='', empty_en='(missing)')}",
                "",
                "Meeting report context",
                f"Topic: {fmt(topic)}",
                f"Report detail: {fmt(report_detail)}",
                "Detail points:",
            ]
            lines.extend(
                [f"- {point}" for point in normalized_points]
                or ["- (no detail points provided)"]
            )
            if brief_context:
                lines.extend(["", "Initial meeting information", brief_context])
            lines.extend(
                [
                    "",
                    "Assignment modal fields",
                    f"Summary: {fmt(summary)}",
                    f"Task description: {fmt(task_description)}",
                    f"Assigner: {fmt(assigner, empty_vi='', empty_en='(not provided)')}",
                    f"Assignee: {fmt(assignee, empty_vi='', empty_en='(not provided)')}",
                    f"Start date: {fmt(start_date, empty_vi='', empty_en='(not provided)')}",
                    f"Deadline: {fmt(deadline, empty_vi='', empty_en='(not provided)')}",
                    "",
                    "Current full AI suggestion",
                    fmt(suggestion),
                    "",
                    "User-selected passage",
                    fmt(selected_text),
                    "",
                    "Clarification thread so far",
                ]
            )
            lines.extend(
                [
                    f"- {'User' if item['role'] == 'user' else 'Hasabot'}: {item['content']}"
                    for item in normalized_history
                ]
                or ["- (no prior clarification turns)"]
            )
            lines.extend(
                [
                    "",
                    "User question",
                    fmt(question_text),
                ]
            )
            if str(answer_text or "").strip():
                lines.extend(
                    [
                        "",
                        "Clarification answer already given",
                        fmt(answer_text),
                    ]
                )
            return "\n".join(lines).strip()

        lines = [
            "Ngữ cảnh làm rõ gợi ý giao việc",
            f"Session ID: {fmt(session_id, empty_vi='(thiếu)', empty_en='')}",
            f"Event ID: {fmt(event_id, empty_vi='(thiếu)', empty_en='')}",
            f"Source hash: {fmt(source_hash, empty_vi='(thiếu)', empty_en='')}",
            "",
            "Ngữ cảnh report cuộc họp",
            f"Chủ đề: {fmt(topic)}",
            f"Chi tiết report: {fmt(report_detail)}",
            "Detail points:",
        ]
        lines.extend(
            [f"- {point}" for point in normalized_points]
            or ["- (không có detail points)"]
        )
        if brief_context:
            lines.extend(["", "Thông tin cuộc họp ban đầu", brief_context])
        lines.extend(
            [
                "",
                "Các trường trong modal giao việc",
                f"Tóm tắt: {fmt(summary)}",
                f"Mô tả công việc: {fmt(task_description)}",
                f"Người giao: {fmt(assigner, empty_vi='(chưa nhập)', empty_en='')}",
                f"Người nhận: {fmt(assignee, empty_vi='(chưa nhập)', empty_en='')}",
                f"Ngày bắt đầu: {fmt(start_date, empty_vi='(chưa nhập)', empty_en='')}",
                f"Hạn chót: {fmt(deadline, empty_vi='(chưa nhập)', empty_en='')}",
                "",
                "Toàn bộ gợi ý AI hiện tại",
                fmt(suggestion),
                "",
                "Đoạn người dùng đã bôi chọn",
                fmt(selected_text),
                "",
                "Chuỗi làm rõ trước đó",
            ]
        )
        lines.extend(
            [
                f"- {'Người dùng' if item['role'] == 'user' else 'Hasabot'}: {item['content']}"
                for item in normalized_history
            ]
            or ["- (chưa có lượt làm rõ trước đó)"]
        )
        lines.extend(
            [
                "",
                "Câu hỏi của người dùng",
                fmt(question_text),
            ]
        )
        if str(answer_text or "").strip():
            lines.extend(
                [
                    "",
                    "Câu trả lời làm rõ đã có",
                    fmt(answer_text),
                ]
            )
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_assignment_suggestion_text(value: Any) -> str:
        text = str(value or "").strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _normalize_assignment_suggestion_conversation_history(
        value: Optional[list[dict[str, Any]]],
        *,
        max_turns: int = _ASSIGNMENT_SUGGESTION_HISTORY_MAX_TURNS,
        max_chars: int = _ASSIGNMENT_SUGGESTION_HISTORY_MAX_CHARS,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = MeetingContextService._normalize_assignment_suggestion_text(
                item.get("content")
            )
            if not content:
                continue
            if len(content) > _ASSIGNMENT_SUGGESTION_HISTORY_ITEM_MAX_CHARS:
                keep_head = _ASSIGNMENT_SUGGESTION_HISTORY_ITEM_MAX_CHARS // 2
                keep_tail = _ASSIGNMENT_SUGGESTION_HISTORY_ITEM_MAX_CHARS - keep_head - 5
                content = f"{content[:keep_head]} ... {content[-keep_tail:]}"
            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )
        if max_turns > 0 and len(normalized) > max_turns:
            normalized = normalized[-max_turns:]
        if max_chars <= 0:
            return normalized

        kept: list[dict[str, str]] = []
        remaining_chars = max_chars
        for item in reversed(normalized):
            content = item["content"]
            if remaining_chars <= 0:
                break
            if len(content) > remaining_chars:
                if remaining_chars < 80:
                    break
                keep_head = max(20, remaining_chars // 2)
                keep_tail = max(20, remaining_chars - keep_head - 5)
                content = f"{content[:keep_head]} ... {content[-keep_tail:]}"
            kept.append(
                {
                    "role": item["role"],
                    "content": content,
                }
            )
            remaining_chars -= len(content)
        kept.reverse()
        return kept

    @staticmethod
    def _log_json(payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)


async def get_meeting_context_service() -> MeetingContextService:
    global _service
    if _service is not None:
        return _service

    async with _init_lock:
        if _service is None:
            service = MeetingContextService()
            await service.initialize()
            _service = service
    return _service
