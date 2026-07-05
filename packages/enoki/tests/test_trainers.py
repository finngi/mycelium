"""_build_example: prompt/completion masking under truncation.

Uses a fake character-level ChatML-ish tokenizer (no real transformers
install needed) so token counts are exactly len(text) and the truncation
arithmetic is easy to reason about precisely.
"""

from enoki.trainers import _build_example


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _messages(html: str, markdown: str) -> list[dict]:
    return [
        {"role": "system", "content": "extract"},
        {"role": "user", "content": html},
        {"role": "assistant", "content": markdown},
    ]


def test_fits_within_max_length_masks_only_the_prompt():
    tok = _FakeTokenizer()
    messages = _messages("<p>hi</p>", "hi")
    prompt_text = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, add_generation_prompt=False)

    ex = _build_example(tok, messages, max_length=10_000)

    assert ex is not None
    assert ex["input_ids"] == [ord(c) for c in full_text]
    prompt_len = len(prompt_text)
    assert ex["labels"][:prompt_len] == [-100] * prompt_len
    assert ex["labels"][prompt_len:] == ex["input_ids"][prompt_len:]


def test_oversized_prompt_is_truncated_from_the_left_completion_kept_whole():
    tok = _FakeTokenizer()
    html = "x" * 500
    markdown = "short markdown"
    messages = _messages(html, markdown)
    full_text = tok.apply_chat_template(messages, add_generation_prompt=False)
    # completion_ids = full_ids[len(prompt_ids):], and prompt_ids already
    # include the "<assistant>" generation-prompt tag, so the completion is
    # just the assistant content + its closing tag.
    completion_len = len(markdown + "</assistant>")
    max_length = completion_len + 20

    ex = _build_example(tok, messages, max_length=max_length)

    assert ex is not None
    assert len(ex["input_ids"]) == max_length
    # the completion (end of the sequence) is byte-for-byte intact and fully
    # supervised -- this is the thing the old right-truncation could corrupt.
    assert ex["input_ids"][-completion_len:] == [ord(c) for c in full_text[-completion_len:]]
    assert ex["labels"][-completion_len:] == ex["input_ids"][-completion_len:]
    # everything before the completion is masked prompt (truncated HTML).
    assert ex["labels"][:-completion_len] == [-100] * (max_length - completion_len)


def test_completion_alone_too_long_is_dropped():
    tok = _FakeTokenizer()
    messages = _messages("short html", "y" * 100)

    ex = _build_example(tok, messages, max_length=10)

    assert ex is None
