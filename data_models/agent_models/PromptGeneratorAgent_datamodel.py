from pydantic import BaseModel, Field
from typing import Optional


class PromptModel(BaseModel):
    system_prompt: str = Field(..., description="The generated system prompt")
    user_prompt: str = Field(..., description="The generated user prompt")
    metadata_fields: list[str] = Field(default_factory=list,description="A list of any metadata needed to inject into the prompt")
    output_format: dict = Field(..., description="JSON schema dict for the SchemaGeneratorAgent")
    saved_location_of_prompt: str = Field(...,description='Location where the prompt .md file was saved')
