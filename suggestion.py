import os
import uuid
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import RunConfig, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from utils import clean_markdown_text

from hasa_bot.domain.meeting.meeting_document_service import (
    get_meeting_document_service,
)
from hasa_core.utils.logger import setup_logger

load_dotenv()

logger = setup_logger("logs/meeting_assignment_suggestion_agent.log")

_APP_NAME = "AssignmentSuggestionAskAgent"
_AGENT_NAME = "assignment_suggestion_ask_agent"
_DEFAULT_LITELLM_MODEL = "openai/qwen-flash-200k-3"
_MAX_TOOL_QUERY_CHARS = 7000
_MAX_TOOL_RESULT_SPANS = 4
_MAX_TOOL_RESULT_SNIPPET_CHARS = 1800


def clean_streaming_text_chunk(text: str) -> str:
    if not text:
        return ""
    had_leading_space = text[:1].isspace()
    had_trailing_space = text[-1:].isspace()
    cleaned = clean_markdown_text(text)
    if not cleaned:
        return ""
    if had_leading_space and not cleaned.startswith((" ", "\n", ".", ",", ";", ":", "?", "!")):
        cleaned = f" {cleaned}"
    if had_trailing_space and not cleaned.endswith((" ", "\n")):
        cleaned = f"{cleaned} "
    return cleaned

def _configure_litellm_gateway_env() -> None:
    base_url = str(os.getenv("LANGEXTRACT_BASE_URL") or "").strip()
    if base_url:
        os.environ.setdefault("OPENAI_API_BASE", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    api_key = (
        str(os.getenv("OPENAI_API_KEY") or "").strip()
        or str(os.getenv("GOOGLE_API_KEY") or "").strip()
        or str(os.getenv("CHATGPT_API_KEY") or "").strip()
    )
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)


def _build_litellm_kwargs() -> dict[str, Any]:
    _configure_litellm_gateway_env()
    kwargs: dict[str, Any] = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    base_url = (
        str(os.getenv("LANGEXTRACT_BASE_URL") or "").strip()
        or str(os.getenv("OPENAI_API_BASE") or "").strip()
        or str(os.getenv("OPENAI_BASE_URL") or "").strip()
    )
    if base_url:
        kwargs["api_base"] = base_url

    api_key = (
        str(os.getenv("OPENAI_API_KEY") or "").strip()
        or str(os.getenv("GOOGLE_API_KEY") or "").strip()
        or str(os.getenv("CHATGPT_API_KEY") or "").strip()
    )
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


_configure_litellm_gateway_env()


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalize_language(language: str) -> str:
    return "en" if str(language or "").strip().lower() == "en" else "vi"


def _format_uploaded_document_catalog(
    documents: list[dict[str, Any]],
    *,
    language: str,
) -> str:
    lang = _normalize_language(language)
    clean_documents = [
        document
        for document in documents
        if isinstance(document, dict) and str(document.get("document_id") or "").strip()
    ]
    if not clean_documents:
        return (
            "No uploaded meeting documents are attached to this session."
            if lang == "en"
            else "Phiên này chưa có tài liệu cuộc họp được upload."
        )

    ready_count = sum(
        1
        for document in clean_documents
        if str(document.get("status") or "").strip().lower() == "ready"
    )
    if lang == "en":
        lines = [
            f"Uploaded document count: {len(clean_documents)}; ready to search: {ready_count}.",
            "Catalog:",
        ]
    else:
        lines = [
            f"Số tài liệu đã upload: {len(clean_documents)}; sẵn sàng tra cứu: {ready_count}.",
            "Danh mục:",
        ]

    for index, document in enumerate(clean_documents, start=1):
        document_id = str(document.get("document_id") or "").strip()
        filename = str(document.get("original_filename") or "").strip() or document_id
        status = str(document.get("status") or "").strip() or "pending"
        description = _truncate_text(document.get("doc_description"), limit=500)
        error_message = _truncate_text(document.get("error_message"), limit=240)
        if lang == "en":
            lines.append(
                f"{index}. id={document_id}; file={filename}; status={status}; "
                f"description={description or '(none)'}"
            )
            if error_message:
                lines.append(f"   error={error_message}")
        else:
            lines.append(
                f"{index}. id={document_id}; file={filename}; trạng thái={status}; "
                f"mô tả={description or '(không có)'}"
            )
            if error_message:
                lines.append(f"   lỗi={error_message}")
    return "\n".join(lines)


def _compact_document_resolution(resolution: Any) -> dict[str, Any]:
    if not isinstance(resolution, dict):
        return {
            "status": "error",
            "message": "Document lookup returned an invalid response.",
            "retrieved_spans": [],
        }

    compact: dict[str, Any] = {
        "status": str(resolution.get("status") or "").strip(),
        "message": str(resolution.get("message") or "").strip(),
        "document_count": resolution.get("document_count", 0),
        "ready_document_count": resolution.get("ready_document_count", 0),
        "retrieved_spans": [],
    }
    spans = [
        span
        for span in list(resolution.get("retrieved_spans") or [])
        if isinstance(span, dict) and str(span.get("snippet") or "").strip()
    ]
    for span in spans[:_MAX_TOOL_RESULT_SPANS]:
        compact["retrieved_spans"].append(
            {
                "document_id": str(span.get("document_id") or "").strip(),
                "document_name": str(span.get("document_name") or "").strip(),
                "node_title": str(span.get("node_title") or "").strip(),
                "line_num": span.get("line_num"),
                "source_trace": str(span.get("source_trace") or "").strip(),
                "snippet": _truncate_text(
                    span.get("snippet"),
                    limit=_MAX_TOOL_RESULT_SNIPPET_CHARS,
                ),
            }
        )
    return compact


def _has_function_payload(event: Any) -> bool:
    for method_name in ("get_function_calls", "get_function_responses"):
        method = getattr(event, method_name, None)
        if not callable(method):
            continue
        try:
            if method():
                return True
        except Exception:
            continue
    return False


def _extract_visible_text(event: Any) -> str:
    if _has_function_payload(event):
        return ""
    content = getattr(event, "content", None)
    if not content or not getattr(content, "parts", None):
        return ""

    text_parts: list[str] = []
    for part in list(content.parts or []):
        if getattr(part, "function_call", None) or getattr(part, "function_response", None):
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            text_parts.append(text)
    return "".join(text_parts)


def _build_agent_instruction(
    *,
    base_prompt: str,
    document_catalog_text: str,
    has_document_tool: bool,
    language: str,
) -> str:
    lang = _normalize_language(language)
    if lang == "en":
        tool_text = (
            "Optional uploaded-document tool:\n"
            "- You may call search_uploaded_meeting_documents(query) to search parsed Markdown from uploaded meeting documents.\n"
            "- Call this tool only when uploaded-document evidence can materially improve the answer, or when the user explicitly asks about uploaded documents.\n"
            "- If the selected passage, assignment context, meeting title/objectives/description, current suggestion, and prior thread are enough, answer directly and do not call the tool.\n"
            "- If you call the tool, use the returned snippets only as supporting context. Do not invent document content.\n"
            "- Do not mention internal tool calls unless the user directly asks how the answer was produced.\n"
            if has_document_tool
            else (
                "No uploaded-document search tool is available for this request because there are no uploaded documents in this session. "
                "Do not claim uploaded-document evidence."
            )
        )
        catalog_label = "Uploaded-document availability catalog"
    else:
        tool_text = (
            "Công cụ tài liệu upload tùy chọn:\n"
            "- Bạn có thể gọi search_uploaded_meeting_documents(query) để tra cứu Markdown đã parse từ tài liệu cuộc họp được upload.\n"
            "- Chỉ gọi công cụ này khi bằng chứng từ tài liệu upload thật sự giúp câu trả lời tốt hơn, hoặc khi người dùng hỏi trực tiếp về tài liệu upload.\n"
            "- Nếu đoạn bôi chọn, ngữ cảnh giao việc, tiêu đề/mục tiêu/mô tả cuộc họp, gợi ý hiện tại và chuỗi hỏi trước đã đủ, hãy trả lời trực tiếp và không gọi công cụ.\n"
            "- Nếu gọi công cụ, chỉ dùng các trích đoạn trả về như ngữ cảnh hỗ trợ. Không bịa nội dung tài liệu.\n"
            "- Không nhắc tới việc gọi tool nội bộ trừ khi người dùng hỏi trực tiếp cách tạo câu trả lời.\n"
            if has_document_tool
            else (
                "Không có công cụ tra cứu tài liệu upload cho lượt này vì phiên hiện chưa có tài liệu upload. "
                "Không được nói như thể đã có bằng chứng từ tài liệu upload."
            )
        )
        catalog_label = "Danh mục trạng thái tài liệu upload"

    return "\n\n".join(
        [
            str(base_prompt or "").strip(),
            tool_text,
            f"{catalog_label}:\n{document_catalog_text}",
        ]
    ).strip()


class AssignmentSuggestionAskAgent:
    """Popup Ask Hasabot agent with lazy uploaded-document search."""

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()

    @staticmethod
    def _build_generation_config() -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.75,
            max_output_tokens=10000,
        )

    def _build_agent(
        self,
        *,
        # session_id: str,
        language: str,
        base_prompt: str,
        # document_query_context: str,
        uploaded_documents: list[dict[str, Any]],
    ) -> LlmAgent:
        lang = _normalize_language(language)
        document_catalog_text = _format_uploaded_document_catalog(
            uploaded_documents,
            language=lang,
        )
        has_document_tool = bool(uploaded_documents)
        # tool_query_context = str(document_query_context or "").strip()

        # async def search_uploaded_meeting_documents(query: str) -> str:
            # """Search parsed Markdown from uploaded meeting documents for this popup question."""
            # clean_query = re.sub(r"\s+", " ", str(query or "").strip())
            # combined_query = "\n\n".join(
            #     part
            #     for part in [
            #         f"Agent search query: {clean_query}" if clean_query else "",
            #         "Popup Ask Hasabot context:",
            #         tool_query_context,
            #     ]
            #     if part
            # )[:_MAX_TOOL_QUERY_CHARS]
            # try:
            #     logger.info(
            #         "Assignment popup document tool called: session_id={} query_chars={}",
            #         session_id,
            #         len(combined_query),
            #     )
            #     # document_service = await get_meeting_document_service()
            #     resolution = await document_service.resolve_uploaded_document_query(
            #         session_id=session_id,
            #         question=combined_query or clean_query,
            #         language=lang,
            #     )
            #     return json.dumps(
            #         _compact_document_resolution(resolution),
            #         ensure_ascii=False,
            #     )
            # except Exception as exc:
            #     logger.exception(
            #         "Assignment popup document tool failed: session_id={} error={}",
            #         session_id,
            #         exc,
            #     )
            #     message = (
            #         f"Could not search uploaded meeting documents right now: {exc}"
            #         if lang == "en"
            #         else f"Không thể tra cứu tài liệu cuộc họp đã upload lúc này: {exc}"
            #     )
            #     return json.dumps(
            #         {
            #             "status": "error",
            #             "message": message,
            #             "retrieved_spans": [],
            #         },
            #         ensure_ascii=False,
            #     )

        # tools = [FunctionTool(search_uploaded_meeting_documents)] if has_document_tool else []
        return LlmAgent(
            name=_AGENT_NAME,
            model=LiteLlm(model=_DEFAULT_LITELLM_MODEL, **_build_litellm_kwargs()),
            description="Clarifies selected assignment-suggestion text and advises on practical work execution.",
            instruction=_build_agent_instruction(
                base_prompt=base_prompt,
                document_catalog_text=document_catalog_text,
                has_document_tool=has_document_tool,
                language=lang,
            ),
            # tools=tools,
            generate_content_config=self._build_generation_config(),
        )

    async def _collect_non_streaming_answer(
        self,
        *,
        runner: Runner,
        session_id: str,
        adk_session_id: str,
        content: genai_types.Content,
    ) -> str:
        events = runner.run_async(
            user_id=session_id,
            session_id=adk_session_id,
            new_message=content,
            run_config=RunConfig(streaming_mode=StreamingMode.NONE, max_llm_calls=12),
        )
        answer_parts: list[str] = []
        async for event in events:
            if _has_function_payload(event):
                calls = getattr(event, "get_function_calls", lambda: [])()
                if calls:
                    logger.info(
                        "Assignment popup agent tool call session_id={} tools={}",
                        session_id,
                        [getattr(call, "name", "") for call in calls],
                    )
                continue
            text = _extract_visible_text(event)
            if text:
                answer_parts.append(text)
        return "".join(answer_parts).strip()

    async def stream_answer(
        self,
        *,
        session_id: str,
        language: str,
        base_prompt: str,
        context_input: str,
        # document_query_context: str,
        uploaded_documents: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        lang = _normalize_language(language)
        agent = self._build_agent(
            # session_id=session_id,
            language=lang,
            base_prompt=base_prompt,
            # document_query_context=document_query_context,
            uploaded_documents=list(uploaded_documents or []),
        )
        runner = Runner(
            agent=agent,
            app_name=_APP_NAME,
            session_service=self.session_service,
        )
        adk_session_id = f"{session_id}:assignment-ask:{uuid.uuid4().hex[:10]}"
        await self.session_service.create_session(
            app_name=_APP_NAME,
            user_id=session_id,
            session_id=adk_session_id,
        )
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=str(context_input or "").strip())],
        )

        events = runner.run_async(
            user_id=session_id,
            session_id=adk_session_id,
            new_message=content,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=12),
        )

        content_streamed = False
        final_text_parts: list[str] = []
        async for event in events:
            if _has_function_payload(event):
                calls = getattr(event, "get_function_calls", lambda: [])()
                if calls:
                    logger.info(
                        "Assignment popup agent tool call session_id={} tools={}",
                        session_id,
                        [getattr(call, "name", "") for call in calls],
                    )
                continue

            text = _extract_visible_text(event)
            if not text:
                continue

            if getattr(event, "partial", False):
                cleaned = clean_streaming_text_chunk(text)
                if cleaned:
                    content_streamed = True
                    yield cleaned
            else:
                final_text_parts.append(text)

        final_text = "".join(final_text_parts).strip()
        if not content_streamed and final_text:
            cleaned = clean_streaming_text_chunk(final_text)
            if cleaned:
                yield cleaned
                return

        if content_streamed:
            return

        logger.warning(
            "Assignment popup agent SSE produced no visible output; retrying non-streaming: session_id={}",
            session_id,
        )
        fallback_session_id = f"{session_id}:assignment-ask-fallback:{uuid.uuid4().hex[:10]}"
        await self.session_service.create_session(
            app_name=_APP_NAME,
            user_id=session_id,
            session_id=fallback_session_id,
        )
        fallback_text = await self._collect_non_streaming_answer(
            runner=runner,
            session_id=session_id,
            adk_session_id=fallback_session_id,
            content=content,
        )
        if fallback_text:
            cleaned = clean_streaming_text_chunk(fallback_text)
            if cleaned:
                yield cleaned
                return

        raise ValueError("Assignment suggestion ask agent returned no visible output")


_assignment_suggestion_ask_agent: Optional[AssignmentSuggestionAskAgent] = None


def get_assignment_suggestion_ask_agent() -> AssignmentSuggestionAskAgent:
    global _assignment_suggestion_ask_agent
    if _assignment_suggestion_ask_agent is None:
        _assignment_suggestion_ask_agent = AssignmentSuggestionAskAgent()
    return _assignment_suggestion_ask_agent
