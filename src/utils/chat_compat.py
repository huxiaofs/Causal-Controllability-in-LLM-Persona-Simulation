"""Compatibility helpers for chat templates across model families."""
from __future__ import annotations

from typing import Dict, List


def adapt_messages(
    tokenizer,
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Merge ``system`` messages into the first ``user`` message when the
    tokenizer's chat template does not support the ``system`` role.

    Some model families (e.g. Mistral / Mixtral) only accept alternating
    ``user`` / ``assistant`` turns.  Calling ``apply_chat_template`` with a
    ``system`` message raises a Jinja error for those models.  This helper
    inspects the chat template text for a ``system`` branch and, when absent,
    folds every ``system`` message into the following ``user`` turn.
    """
    if not messages:
        return messages

    has_system = any(m["role"] == "system" for m in messages)
    if not has_system:
        return messages

    template = getattr(tokenizer, "chat_template", None) or ""
    if "system" in template:
        return messages

    # No system support: merge system contents into the next user message.
    adapted: List[Dict[str, str]] = []
    pending_system: List[str] = []
    for m in messages:
        if m["role"] == "system":
            pending_system.append(m["content"])
            continue
        if m["role"] == "user" and pending_system:
            prefix = "\n\n".join(pending_system)
            m = {"role": "user", "content": f"{prefix}\n\n{m['content']}"}
            pending_system.clear()
        adapted.append(m)
    return adapted
