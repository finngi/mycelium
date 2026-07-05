from mcm.primitives.task import Task, register

# Output is free-text markdown, not a structured field extraction, hence
# codec="text" and a single output_field -- or the literal sentinel string
# "INVALID_PAGE" when the source HTML has no extractable main content (the
# refusal contract distilled from mycelium's readerlm-ft harvest). No scorer
# yet, same as nameparse.
htmlmd = register(
    Task(
        name="htmlmd",
        description="HTML -> Markdown main-content extraction, or the sentinel "
        "'INVALID_PAGE' when the page has no extractable content (ReaderLM-style distillation)",
        output_fields=("markdown",),
        score=None,
        codec="text",
    )
)
