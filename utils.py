"""Utilities for richer clustered meeting context."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any


_RECENT_EVENTS_DISPLAY_LIMIT = 20

MEETING_EVENT_TYPES = (
    "report",
    "question",
    "answer",
    "conclusion",
    "action_item",
    "blocker",
    "discussion",
    "info",
    # legacy — kept for backward compat with stored documents
    "decision",
)
MEETING_EVENT_LABELS = {
    "report": "Report",
    "question": "Question",
    "answer": "Answer",
    "conclusion": "Conclusion",
    "action_item": "Action Item",
    "blocker": "Blocker",
    "discussion": "Discussion",
    "info": "Info",
    "decision": "Decision",
}
MEETING_EVENT_LABELS_VI = {
    "report": "Báo cáo",
    "question": "Câu hỏi",
    "answer": "Trả lời",
    "conclusion": "Kết luận",
    "action_item": "Việc cần làm",
    "blocker": "Vướng mắc",
    "discussion": "Thảo luận",
    "info": "Thông tin",
    "decision": "Quyết định",
}

MEETING_PARTICIPANT_ROLES = ("reporter", "questioner", "responder", "facilitator")
MEETING_PARTICIPANT_ROLE_LABELS = {
    "reporter": "Reporter",
    "questioner": "Questioner",
    "responder": "Responder",
    "facilitator": "Facilitator",
}
MEETING_PARTICIPANT_ROLE_LABELS_VI = {
    "reporter": "Người báo cáo",
    "questioner": "Người đặt câu hỏi",
    "responder": "Người trả lời",
    "facilitator": "Người điều hành",
}
MEETING_PHASES = ("opening", "discussion", "decision", "closing")
_UNKNOWN_METADATA_KEYS = {
    "null",
    "none",
    "n a",
    "na",
    "unknown",
    "khong ro",
    "chua ro",
    "khong xac dinh",
    "chua xac dinh",
}
_PLACEHOLDER_OWNER_KEYS = {
    "nguyen van a",
    "nguyen van b",
    "tran thi a",
    "tran thi b",
    "le van a",
    "le thi b",
    "person a",
    "person b",
    "user a",
    "user b",
}
_OWNER_ORG_HINT_KEYS = {
    "ban",
    "be",
    "bo",
    "bp",
    "bophan",
    "business",
    "care",
    "crm",
    "cs",
    "cskh",
    "customer",
    "dept",
    "department",
    "design",
    "doi",
    "engineering",
    "facilitator",
    "fe",
    "finance",
    "group",
    "hr",
    "it",
    "khoi",
    "kythuat",
    "leadership",
    "management",
    "marketing",
    "nhom",
    "ops",
    "operation",
    "operations",
    "phong",
    "product",
    "qa",
    "qc",
    "quanly",
    "responder",
    "sale",
    "sales",
    "shop",
    "success",
    "support",
    "team",
    "tech",
    "thu",
    "to",
    "vanhanh",
}

def clean_markdown_text(text: str, preserve_images: bool = False) -> str:
    """Clean markdown and LaTeX formatting from text, converting to plain Vietnamese text."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text
    latex_patterns = {
        r'\$\\rightarrow\$': ' sau đó ',
        r'\$\\to\$': ' sau đó ',
        r'\$→\$': ' sau đó ',
        r'\$\\leftarrow\$': ' trước đó ',
        r'\$←\$': ' trước đó ',
        r'\$\\Rightarrow\$': ' do đó ',
        r'\$⇒\$': ' do đó ',
        r'\$\\Leftarrow\$': ' vì ',
        r'\$⇐\$': ' vì ',
        r'\$\\leftrightarrow\$': ' và ngược lại ',
        r'\$↔\$': ' và ngược lại ',
        r'\$\\Leftrightarrow\$': ' tương đương ',
        r'\$⇔\$': ' tương đương ',
        r'\$\\times\$': ' × ',
        r'\$×\$': ' × ',
        r'\$\\div\$': ' ÷ ',
        r'\$÷\$': ' ÷ ',
        r'\$\\pm\$': ' ± ',
        r'\$±\$': ' ± ',
        r'\$\\leq\$': ' ≤ ',
        r'\$≤\$': ' ≤ ',
        r'\$\\geq\$': ' ≥ ',
        r'\$≥\$': ' ≥ ',
        r'\$\\neq\$': ' ≠ ',
        r'\$≠\$': ' ≠ ',
        r'\$\\approx\$': ' ≈ ',
        r'\$≈\$': ' ≈ ',
    }
    for pattern, replacement in latex_patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    corrupted_latex_patterns = {
        r'\$rightarrow\$': ' sau đó ',
        r'\$leftarrow\$': ' trước đó ',
        r'\$Rightarrow\$': ' do đó ',
        r'\$Leftarrow\$': ' vì ',
    }
    for pattern, replacement in corrupted_latex_patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r'\$\$[\s\S]*?\$\$', '', cleaned)
    cleaned = re.sub(r'\$([^\$\n]+)\$', r'\1', cleaned)
    cleaned = re.sub(r'\\begin\{equation\}[\s\S]*?\\end\{equation\}', '', cleaned)
    unicode_symbols = {
        '→': ' sau đó ', '←': ' trước đó ', '⇒': ' do đó ',
        '⇐': ' vì ', '↔': ' và ngược lại ', '⇔': ' tương đương ',
        '×': ' × ', '÷': ' ÷ ', '±': ' ± ',
        '≤': ' ≤ ', '≥': ' ≥ ', '≠': ' ≠ ', '≈': ' ≈ ',
    }
    for symbol, replacement in unicode_symbols.items():
        cleaned = cleaned.replace(symbol, replacement)
    return cleaned

def _normalize_ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    ).lower()
    ascii_text = ascii_text.replace("đ", "d")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def normalize_meeting_topic_key(topic: str) -> str:
    """Return a stable topic key for cluster matching."""
    return _normalize_ascii_key(topic)


def normalize_meeting_detail_key(detail: str) -> str:
    """Return a stable detail key for duplicate suppression."""
    return _normalize_ascii_key(detail)


def normalize_meeting_detail_points(points: Any, *, limit: int = 12) -> list[str]:
    """Return clean, unique detail bullets preserving spoken order."""
    if points is None:
        return []
    raw_points = points if isinstance(points, list) else [points]
    normalized_points: list[str] = []
    seen_keys: set[str] = set()
    for point in raw_points:
        clean_point = re.sub(r"\s+", " ", str(point or "").strip(" -:\n\t"))
        if not clean_point:
            continue
        point_key = normalize_meeting_detail_key(clean_point)
        if not point_key or point_key in seen_keys:
            continue
        normalized_points.append(clean_point)
        seen_keys.add(point_key)
        if len(normalized_points) >= limit:
            break
    return normalized_points


def normalize_meeting_owner_key(owner: str) -> str:
    """Return a normalized owner key for duplicate suppression."""
    return _normalize_ascii_key(owner)


def normalize_meeting_deadline_key(deadline: str) -> str:
    """Return a normalized deadline key for duplicate suppression."""
    return _normalize_ascii_key(deadline)


def build_meeting_event_signature(
    event_type: str,
    detail: str,
    owner: str,
    deadline: str,
) -> str:
    """Return the normalized event signature used for duplicate suppression."""
    return "|".join(
        [
            normalize_meeting_event_type(event_type),
            normalize_meeting_detail_key(detail),
            normalize_meeting_owner_key(owner),
            normalize_meeting_deadline_key(deadline),
        ]
    )


def build_stable_meeting_event_id(
    *,
    topic_key: str,
    event_type: str,
    detail: str,
    owner: str = "",
    deadline: str = "",
) -> str:
    """Return the stable public ID for a meeting event."""
    payload = {
        "topic_key": normalize_meeting_topic_key(topic_key),
        "event_signature": build_meeting_event_signature(
            event_type=event_type,
            detail=detail,
            owner=owner,
            deadline=deadline,
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"event-{digest}"


def normalize_meeting_event_type(event_type: str) -> str:
    normalized = _normalize_ascii_key(event_type).replace(" ", "_")
    return normalized if normalized in MEETING_EVENT_TYPES else "discussion"


def normalize_meeting_participant_role(role: str) -> str:
    normalized = _normalize_ascii_key(role).replace(" ", "_")
    return normalized if normalized in MEETING_PARTICIPANT_ROLES else "responder"


def normalize_meeting_phase(phase: str) -> str:
    normalized = _normalize_ascii_key(phase)
    return normalized if normalized in MEETING_PHASES else "discussion"


def normalize_manual_participants(participants: Any, *, limit: int = 24) -> list[str]:
    """Return a clean ordered roster of manually confirmed participant names."""
    if participants is None:
        return []
    raw_participants = participants if isinstance(participants, list) else [participants]
    normalized_names: list[str] = []
    seen_keys: set[str] = set()
    for raw_name in raw_participants:
        clean_name = re.sub(r"\s+", " ", str(raw_name or "").strip(" -:\n\t"))
        if not clean_name:
            continue
        name_key = _normalize_ascii_key(clean_name)
        if not name_key or name_key in seen_keys:
            continue
        normalized_names.append(clean_name)
        seen_keys.add(name_key)
        if len(normalized_names) >= limit:
            break
    return normalized_names


def _build_manual_participant_lookup(manual_participants: Any) -> dict[str, str]:
    return {
        _normalize_ascii_key(name): name
        for name in normalize_manual_participants(manual_participants)
        if _normalize_ascii_key(name)
    }


def canonicalize_manual_participant_name(name: str, manual_participants: Any) -> str:
    """Return the exact user-entered participant name if it exists in the roster."""
    clean_name = _clean_meeting_metadata_value(name)
    if not clean_name:
        return ""
    return _build_manual_participant_lookup(manual_participants).get(
        _normalize_ascii_key(clean_name),
        "",
    )


def normalize_meeting_participants_with_roster(
    participants: Any,
    manual_participants: Any,
) -> list[dict[str, Any]]:
    """Keep only roster-backed structured participants and canonicalize their names."""
    lookup = _build_manual_participant_lookup(manual_participants)
    if not lookup:
        return []

    merged: dict[str, dict[str, Any]] = {}
    for raw_participant in list(participants or []):
        raw_name = str((raw_participant or {}).get("name") or "").strip()
        canonical_name = lookup.get(_normalize_ascii_key(raw_name))
        if not canonical_name:
            continue

        participant_key = _normalize_ascii_key(canonical_name)
        participant_role = normalize_meeting_participant_role(
            str((raw_participant or {}).get("role") or "")
        )
        existing = merged.get(participant_key)
        if existing is None:
            merged[participant_key] = {
                "name": canonical_name,
                "role": participant_role,
                "verified": True,
            }
            continue

        if existing["role"] == "responder" and participant_role != "responder":
            existing["role"] = participant_role
        existing["verified"] = True

    return list(merged.values())


def _looks_like_meeting_group_owner(owner: str) -> bool:
    clean_owner = _clean_meeting_metadata_value(owner)
    if not clean_owner:
        return False

    owner_key = _normalize_ascii_key(clean_owner)
    if not owner_key:
        return False

    tokens = owner_key.split()
    if any(token in _OWNER_ORG_HINT_KEYS for token in tokens):
        return True

    alpha_text = re.sub(r"[^A-Za-z]", "", clean_owner)
    return bool(alpha_text and clean_owner == clean_owner.upper() and len(alpha_text) <= 6)


def canonicalize_meeting_owner_with_roster(
    owner: str,
    manual_participants: Any,
    *,
    detail: str = "",
) -> str:
    """Allow person owners only from the manual roster; keep grounded team labels."""
    clean_owner = sanitize_meeting_owner(owner, detail=detail)
    if not clean_owner:
        return ""

    canonical_name = canonicalize_manual_participant_name(
        clean_owner,
        manual_participants,
    )
    if canonical_name:
        return canonical_name

    if _looks_like_meeting_group_owner(clean_owner):
        return clean_owner

    return ""


def enforce_manual_roster_on_topic_updates(
    topic_updates: list[dict[str, Any]],
    manual_participants: Any,
) -> list[dict[str, Any]]:
    """Canonicalize structured person names in extracted updates against the manual roster."""
    roster = normalize_manual_participants(manual_participants)
    sanitized_updates: list[dict[str, Any]] = []

    for raw_update in list(topic_updates or []):
        update = dict(raw_update or {})
        detail = str(update.get("detail") or "").strip()
        update["participants"] = normalize_meeting_participants_with_roster(
            update.get("participants"),
            roster,
        )
        update["speaker"] = canonicalize_manual_participant_name(
            str(update.get("speaker") or ""),
            roster,
        )
        update["owner"] = canonicalize_meeting_owner_with_roster(
            str(update.get("owner") or ""),
            roster,
            detail=detail,
        )
        sanitized_updates.append(update)

    return sanitized_updates


def enforce_manual_roster_on_topic_clusters(
    topic_clusters: list[dict[str, Any]],
    manual_participants: Any,
) -> list[dict[str, Any]]:
    """Canonicalize structured person names in stored topic clusters against the manual roster."""
    roster = normalize_manual_participants(manual_participants)
    sanitized_clusters: list[dict[str, Any]] = []

    for raw_cluster in normalize_topic_clusters_payload(topic_clusters):
        cluster = dict(raw_cluster or {})
        cluster["participants"] = normalize_meeting_participants_with_roster(
            cluster.get("participants"),
            roster,
        )

        sanitized_events: list[dict[str, Any]] = []
        for raw_event in list(cluster.get("events") or []):
            event = dict(raw_event or {})
            detail = str(event.get("detail") or "").strip()
            speaker = canonicalize_manual_participant_name(
                str(event.get("speaker") or ""),
                roster,
            )
            event["speaker"] = speaker
            event["speaker_verified"] = bool(speaker)
            event["owner"] = canonicalize_meeting_owner_with_roster(
                str(event.get("owner") or ""),
                roster,
                detail=detail,
            )
            sanitized_events.append(event)

        cluster["events"] = sanitized_events
        sanitized_clusters.append(cluster)

    return sanitized_clusters


def get_meeting_event_label(event_type: str, language: str = "en") -> str:
    normalized_event_type = normalize_meeting_event_type(event_type)
    if str(language or "").strip().lower() == "vi":
        return MEETING_EVENT_LABELS_VI[normalized_event_type]
    return MEETING_EVENT_LABELS[normalized_event_type]


def _clean_meeting_metadata_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" -:\n\t"))


def _metadata_is_grounded_in_detail(value: str, detail: str) -> bool:
    value_key = _normalize_ascii_key(value)
    detail_key = _normalize_ascii_key(detail)
    return bool(value_key and detail_key and value_key in detail_key)


def sanitize_meeting_owner(owner: str, *, detail: str = "") -> str:
    """Suppress owners that look unknown, placeholder-like, or ungrounded."""
    clean_owner = _clean_meeting_metadata_value(owner)
    owner_key = _normalize_ascii_key(clean_owner)
    if not owner_key or owner_key in _UNKNOWN_METADATA_KEYS:
        return ""
    if owner_key in _PLACEHOLDER_OWNER_KEYS:
        return ""
    if not _metadata_is_grounded_in_detail(clean_owner, detail):
        return ""
    return clean_owner


def sanitize_meeting_deadline(deadline: str, *, detail: str = "") -> str:
    """Suppress deadlines unless they are explicitly grounded in the event detail."""
    clean_deadline = _clean_meeting_metadata_value(deadline)
    deadline_key = _normalize_ascii_key(clean_deadline)
    if not deadline_key or deadline_key in _UNKNOWN_METADATA_KEYS:
        return ""
    if not _metadata_is_grounded_in_detail(clean_deadline, detail):
        return ""
    return clean_deadline


def clamp_meeting_confidence(value: Any, *, default: float = 1.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_int_or_default(value: Any, default: int = -1) -> int:
    coerced = coerce_optional_int(value)
    return default if coerced is None else coerced


def format_meeting_time_label(seconds: Any) -> str:
    numeric = coerce_optional_float(seconds)
    if numeric is None:
        return ""
    total_seconds = max(0, int(round(numeric)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_meeting_event_detail(event: dict[str, Any]) -> str:
    event_type = normalize_meeting_event_type(str((event or {}).get("event_type") or ""))
    detail = str((event or {}).get("detail") or "").strip()
    speaker = str((event or {}).get("speaker") or "").strip()
    owner = sanitize_meeting_owner(str((event or {}).get("owner") or ""), detail=detail)
    deadline = sanitize_meeting_deadline(
        str((event or {}).get("deadline") or ""),
        detail=detail,
    )

    meta_parts: list[str] = []
    if speaker:
        meta_parts.append(f"Speaker: {speaker}")
    if owner:
        meta_parts.append(f"Owner: {owner}")
    if deadline:
        meta_parts.append(f"Deadline: {deadline}")
    if meta_parts:
        rendered = f"[{get_meeting_event_label(event_type)}] {detail} ({' | '.join(meta_parts)})"
    else:
        rendered = f"[{get_meeting_event_label(event_type)}] {detail}"

    detail_points = normalize_meeting_detail_points((event or {}).get("detail_points"))
    if detail_points:
        rendered += "\n" + "\n".join(f"  • {point}" for point in detail_points)
    return rendered


def _parse_legacy_summary_blocks(summary_text: str) -> list[dict[str, Any]]:
    blocks = [
        block.strip()
        for block in str(summary_text or "").split("\n\n")
        if block.strip()
    ]
    clusters: list[dict[str, Any]] = []

    for index, block in enumerate(blocks, start=1):
        topic = ""
        details: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower_line = line.lower()
            if lower_line.startswith("chủ đề:") or lower_line.startswith("chu de:"):
                topic = line.split(":", 1)[1].strip()
                continue
            if lower_line.startswith("topic:"):
                topic = line.split(":", 1)[1].strip()
                continue
            if lower_line.startswith("- chi tiết:") or lower_line.startswith("- chi tiet:"):
                details.append(line.split(":", 1)[1].strip())
                continue
            if line.startswith("- "):
                details.append(line[2:].strip())
                continue
            if not topic:
                topic = line
            else:
                details.append(line)

        clusters.append(
            {
                "id": f"legacy-topic-{index}",
                "topic": topic,
                "topic_key": normalize_meeting_topic_key(topic),
                "details": details,
                "aliases": [],
                "events": [
                    {
                        "id": build_stable_meeting_event_id(
                            topic_key=normalize_meeting_topic_key(topic),
                            event_type="info",
                            detail=detail,
                        ),
                        "event_type": "info",
                        "detail": detail,
                        "detail_points": [],
                        "owner": "",
                        "deadline": "",
                        "confidence": 1.0,
                        "segment_index": -1,
                        "approx_start_second": None,
                        "approx_end_second": None,
                    }
                    for detail in details
                    if detail
                ],
                "created_segment_index": -1,
                "last_updated_segment_index": -1,
                "approx_start_second": None,
                "approx_end_second": None,
                "importance_rank": index,
            }
        )

    return clusters


def normalize_topic_clusters_payload(
    topic_clusters: list[dict[str, Any]] | None,
    *,
    legacy_summary_text: str = "",
) -> list[dict[str, Any]]:
    """Normalize both legacy and modern cluster shapes for read-time consumers."""
    if topic_clusters:
        normalized: list[dict[str, Any]] = []
        for index, cluster in enumerate(topic_clusters, start=1):
            topic = str((cluster or {}).get("topic") or "").strip()
            aliases = [
                str(alias).strip()
                for alias in list((cluster or {}).get("aliases") or [])
                if str(alias).strip()
            ]
            topic_key = str((cluster or {}).get("topic_key") or "").strip()
            if topic and not topic_key:
                topic_key = normalize_meeting_topic_key(topic)

            raw_events = list((cluster or {}).get("events") or [])
            if not raw_events and (cluster or {}).get("details"):
                raw_events = [
                    {
                        "event_type": "info",
                        "detail": str(detail).strip(),
                        "owner": "",
                        "deadline": "",
                        "confidence": 1.0,
                        "segment_index": coerce_int_or_default((cluster or {}).get("last_updated_segment_index")),
                        "approx_start_second": (cluster or {}).get("approx_start_second"),
                        "approx_end_second": (cluster or {}).get("approx_end_second"),
                    }
                    for detail in list((cluster or {}).get("details") or [])
                    if str(detail).strip()
                ]

            events: list[dict[str, Any]] = []
            for event in raw_events:
                detail = str((event or {}).get("detail") or "").strip()
                if not detail:
                    continue
                event_type = normalize_meeting_event_type(str((event or {}).get("event_type") or "info"))
                owner = sanitize_meeting_owner(
                    str((event or {}).get("owner") or ""),
                    detail=detail,
                )
                deadline = sanitize_meeting_deadline(
                    str((event or {}).get("deadline") or ""),
                    detail=detail,
                )
                events.append(
                    {
                        "id": str(
                            (event or {}).get("id")
                            or build_stable_meeting_event_id(
                                topic_key=topic_key,
                                event_type=event_type,
                                detail=detail,
                                owner=owner,
                                deadline=deadline,
                            )
                        ),
                        "event_type": event_type,
                        "detail": detail,
                        "detail_points": normalize_meeting_detail_points(
                            (event or {}).get("detail_points")
                        ),
                        "speaker": str((event or {}).get("speaker") or "").strip(),
                        "answers_to_event_id": str((event or {}).get("answers_to_event_id") or "").strip(),
                        "answered_by_event_ids": [
                            str(eid).strip()
                            for eid in list((event or {}).get("answered_by_event_ids") or [])
                            if str(eid).strip()
                        ],
                        "owner": owner,
                        "deadline": deadline,
                        "confidence": clamp_meeting_confidence((event or {}).get("confidence"), default=1.0),
                        "segment_index": coerce_int_or_default((event or {}).get("segment_index")),
                        "approx_start_second": coerce_optional_float((event or {}).get("approx_start_second")),
                        "approx_end_second": coerce_optional_float((event or {}).get("approx_end_second")),
                    }
                )

            event_counts = {event_type: 0 for event_type in MEETING_EVENT_TYPES}
            for event in events:
                event_counts[event["event_type"]] += 1

            start_values = [
                event["approx_start_second"]
                for event in events
                if event.get("approx_start_second") is not None
            ]
            end_values = [
                event["approx_end_second"]
                for event in events
                if event.get("approx_end_second") is not None
            ]

            raw_participants = list((cluster or {}).get("participants") or [])
            participants = [
                {
                    "name": str((p or {}).get("name") or "").strip(),
                    "role": normalize_meeting_participant_role(str((p or {}).get("role") or "")),
                    "verified": bool((p or {}).get("verified") is True),
                }
                for p in raw_participants
                if str((p or {}).get("name") or "").strip()
            ]

            normalized.append(
                {
                    "id": str((cluster or {}).get("id") or f"topic-{index}"),
                    "topic": topic,
                    "topic_key": topic_key,
                    "aliases": aliases,
                    "participants": participants,
                    "events": events,
                    "event_counts": event_counts,
                    "created_segment_index": coerce_int_or_default((cluster or {}).get("created_segment_index")),
                    "last_updated_segment_index": coerce_int_or_default((cluster or {}).get("last_updated_segment_index")),
                    "approx_start_second": (
                        min(start_values)
                        if start_values
                        else coerce_optional_float((cluster or {}).get("approx_start_second"))
                    ),
                    "approx_end_second": (
                        max(end_values)
                        if end_values
                        else coerce_optional_float((cluster or {}).get("approx_end_second"))
                    ),
                    "importance_rank": coerce_optional_int((cluster or {}).get("importance_rank")) or index,
                }
            )
        return normalized

    if not str(legacy_summary_text or "").strip():
        return []

    parsed_legacy_clusters = _parse_legacy_summary_blocks(legacy_summary_text)
    if not parsed_legacy_clusters:
        return []

    return normalize_topic_clusters_payload(
        parsed_legacy_clusters,
        legacy_summary_text="",
    )


def format_topic_clusters_summary(topic_clusters: list[dict[str, Any]]) -> str:
    """Render clustered meeting topics into the legacy summary text format."""
    normalized_clusters = normalize_topic_clusters_payload(topic_clusters)
    blocks: list[str] = []

    for cluster in normalized_clusters:
        topic = str((cluster or {}).get("topic") or "").strip()
        events = list((cluster or {}).get("events") or [])
        lines: list[str] = []
        if topic:
            lines.append(f"Chủ đề: {topic}")
        for event in events:
            detail = str((event or {}).get("detail") or "").strip()
            if detail:
                lines.append(f"- Chi tiết: {detail}")
                for point in normalize_meeting_detail_points((event or {}).get("detail_points")):
                    lines.append(f"  - {point}")
        if lines:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()


def build_topic_cluster_items(
    topic_clusters: list[dict[str, Any]],
    *,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Convert stored topic clusters into UI-friendly items."""
    normalized_clusters = sorted(
        normalize_topic_clusters_payload(topic_clusters),
        key=lambda cluster: (
            int((cluster or {}).get("importance_rank") or 0),
            -len(list((cluster or {}).get("events") or [])),
            str((cluster or {}).get("topic") or ""),
        ),
    )
    items: list[dict[str, Any]] = []

    for index, cluster in enumerate(normalized_clusters, start=1):
        events = sorted(
            list((cluster or {}).get("events") or []),
            key=lambda event: (
                coerce_optional_float((event or {}).get("approx_start_second")) or 0.0,
                coerce_int_or_default((event or {}).get("segment_index")),
                str((event or {}).get("id") or ""),
            ),
            reverse=True,
        )
        details = [str((event or {}).get("detail") or "").strip() for event in events if str((event or {}).get("detail") or "").strip()]
        detail = "\n".join(details).strip()
        display_text = str((cluster or {}).get("topic") or "").strip() or detail or "Key point"
        if display_text and detail and display_text != detail:
            display_text = f"{display_text}\n{detail}"
        latest_segment_index = max(
            [
                coerce_int_or_default((event or {}).get("segment_index"))
                for event in events
            ]
            or [coerce_int_or_default((cluster or {}).get("last_updated_segment_index"))]
        )
        latest_event_second = max(
            [
                coerce_optional_float((event or {}).get("approx_start_second")) or 0.0
                for event in events
            ]
            or [coerce_optional_float((cluster or {}).get("approx_end_second")) or 0.0]
        )

        items.append(
            {
                "id": str((cluster or {}).get("id") or f"topic-{index}"),
                "topic": str((cluster or {}).get("topic") or "").strip(),
                "detail": detail,
                "details": details,
                "text": display_text.strip(),
                "participants": list((cluster or {}).get("participants") or []),
                "event_counts": dict((cluster or {}).get("event_counts") or {}),
                "event_count": len(events),
                "last_updated_segment_index": latest_segment_index,
                "latest_event_time_label": format_meeting_time_label(latest_event_second),
                "events": [
                    {
                        **event,
                        "speaker_verified": bool(
                            str((event or {}).get("speaker") or "").strip()
                        ),
                        "detail_points": normalize_meeting_detail_points(
                            (event or {}).get("detail_points")
                        ),
                        "event_label": get_meeting_event_label(
                            str((event or {}).get("event_type") or ""),
                            language=language,
                        ),
                        "time_label": format_meeting_time_label((event or {}).get("approx_start_second")),
                    }
                    for event in events
                ],
                "time_label": format_meeting_time_label((cluster or {}).get("approx_start_second")),
                "importance_rank": int((cluster or {}).get("importance_rank") or index),
            }
        )

    return items


def build_meeting_event_collections(
    topic_clusters: list[dict[str, Any]],
    *,
    language: str = "en",
) -> dict[str, Any]:
    normalized_clusters = normalize_topic_clusters_payload(topic_clusters)
    action_items: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    open_questions: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    recent_events: list[dict[str, Any]] = []
    event_totals = {event_type: 0 for event_type in MEETING_EVENT_TYPES}

    for cluster in normalized_clusters:
        topic = str((cluster or {}).get("topic") or "").strip()
        cluster_id = str((cluster or {}).get("id") or "").strip()
        for event in list((cluster or {}).get("events") or []):
            event_type = normalize_meeting_event_type(str((event or {}).get("event_type") or ""))
            event_totals[event_type] += 1
            enriched = {
                **event,
                "topic": topic,
                "cluster_id": cluster_id,
                "event_label": get_meeting_event_label(event_type, language=language),
                "time_label": format_meeting_time_label((event or {}).get("approx_start_second")),
            }
            recent_events.append(enriched)
            if event_type == "action_item":
                action_items.append(enriched)
            elif event_type == "blocker":
                blockers.append(enriched)
            elif event_type == "question":
                if not list((event or {}).get("answered_by_event_ids") or []):
                    open_questions.append(enriched)
            elif event_type == "conclusion":
                conclusions.append(enriched)
            elif event_type == "decision":
                decisions.append(enriched)

    recent_events.sort(
        key=lambda event: (
            coerce_optional_float((event or {}).get("approx_start_second")) or -1.0,
            coerce_int_or_default((event or {}).get("segment_index")),
        ),
        reverse=True,
    )

    return {
        "action_items": action_items,
        "blockers": blockers,
        "open_questions": open_questions,
        "conclusions": conclusions,
        "decisions": decisions,
        "recent_events": recent_events[:_RECENT_EVENTS_DISPLAY_LIMIT],
        "event_totals": event_totals,
    }


def format_topic_clusters_context(
    topic_clusters: list[dict[str, Any]],
    *,
    meeting_phase: str = "",
) -> str:
    """Render structured meeting clusters into answer-friendly text."""
    normalized_clusters = normalize_topic_clusters_payload(topic_clusters)
    lines: list[str] = []
    normalized_phase = normalize_meeting_phase(meeting_phase) if meeting_phase else ""
    if normalized_phase:
        lines.append(f"Giai đoạn cuộc họp: {normalized_phase}")

    for cluster in sorted(
        normalized_clusters,
        key=lambda item: (
            int((item or {}).get("importance_rank") or 0),
            coerce_optional_float((item or {}).get("approx_start_second")) or 0.0,
        ),
    ):
        topic = str((cluster or {}).get("topic") or "").strip() or "Chủ đề chưa đặt tên"
        topic_time = format_meeting_time_label((cluster or {}).get("approx_start_second"))
        if topic_time:
            lines.append(f"Chủ đề: {topic} (~{topic_time})")
        else:
            lines.append(f"Chủ đề: {topic}")

        events = sorted(
            list((cluster or {}).get("events") or []),
            key=lambda event: (
                coerce_optional_float((event or {}).get("approx_start_second")) or 0.0,
                coerce_int_or_default((event or {}).get("segment_index")),
            ),
        )
        for event in events:
            time_label = format_meeting_time_label((event or {}).get("approx_start_second"))
            prefix = "-"
            if time_label:
                prefix += f" ~{time_label}"
            lines.append(f"{prefix} {format_meeting_event_detail(event)}")

    return "\n".join(lines).strip()
