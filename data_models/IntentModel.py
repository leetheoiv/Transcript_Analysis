from pydantic import BaseModel
from typing import Literal

class IntentModel(BaseModel):
    intent: Literal["generate", "converse"]