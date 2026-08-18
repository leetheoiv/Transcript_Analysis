from pydantic import BaseModel, Field


class SchemaGeneratorResult(BaseModel):
    model_name: str = Field(..., description="PascalCase class name inferred from the analyst's question")
    code: str = Field(..., description="Complete Python class definition including all imports")
    prompt_feedback: str = Field("", description="Empty if fields are sufficient; otherwise describes what needs fixing in the prompt")
