INTENT_PROMPT = """
Classify the user's message into one of two intents:
- "generate": the user wants a new prompt created or an existing prompt modified/updated
- "converse": the user is asking a question, giving feedback, or discussing without requesting a change this ussually will end with a "?"

Reply with JSON only: {"intent": "generate"} or {"intent": "converse"}
"""