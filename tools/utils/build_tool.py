"""
tools/utils/build_tool.py

Defines the Tool model used to register callable tools with Pydantic-validated
inputs and OpenAI-compatible function schema generation.
"""

from typing import Callable, Type
from pydantic import BaseModel

   # Example Tools Format How Openai Expects the Format

    # tools = [

    #     {
    #         "name":"get_weather",
    #         "description":"Get the current weather in a given location",
    #         "input_schema":{
    #             "type":'object',
    #             "properties":{
    #                 'location':{
    #                     'type':'string',
    #                     'description':'the city and state'
    #                 }
    #             },
    #             'required':['location']
    #         }
    #     }
    # ]
    

class Tool(BaseModel):
    """Represents a callable tool with a Pydantic input model for validation.

    Encapsulates a tool's name, description, input schema (as a Pydantic model),
    and the callable function that implements the tool logic. Can be converted
    to an OpenAI-compatible function schema for use in chat completions.
    """

    name: str
    description: str
    input_model: Type[BaseModel]
    func: Callable

    class Config:
        arbitrary_types_allowed = True

    def to_openai_schema(self) -> dict:
        """Convert this tool to an OpenAI function-calling schema dict.

        Returns:
            Dict with type, name, description, and parameters keys
            matching the OpenAI tools format.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_model.model_json_schema()
        }
 

