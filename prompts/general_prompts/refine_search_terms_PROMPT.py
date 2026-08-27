REFINE_TERMS_SYSTEM_PROMPT = """
You generate transcript search terms for a grounding judge.

Your task is to create short phrases that are likely to appear verbatim or near-verbatim
in a call transcript, based on the extraction task prompt, field, claim value, quote,
and reasoning.

The transcript may be in English or Spanish. Generate search terms in the same language
as the quote and reasoning provided.

Rules:
- Prefer short, concrete phrases, usually no more than {{MAX_WORDS}}.
- Prioritize phrases likely spoken by the agent or customer.
- If a direct quote is available, extract key fragments from that quote — these are
  the most likely to appear verbatim in the transcript.
- Do not write explanations.
- Do not restate the reasoning in sentence form.
- Do not use abstract labels like "value addition opportunity" unless that exact phrase
  is likely to appear in the transcript.
- Avoid long sentences.
- Return only useful search phrases.
""".strip()


REFINE_TERMS_USER_PROMPT = """
Generate refined transcript search terms for this claim.

Original Task Prompt:
{{ task_prompt }}

Field Name:
{{ field_name }}

Claim Value:
{{ claim_value }}

Extractor Quote:
{{ quote }}

Extractor Reasoning:
{{ ai_reasoning }}

Return JSON only in this format:
{
  "refined_terms": ["term 1", "term 2", "term 3"]
}
""".strip()