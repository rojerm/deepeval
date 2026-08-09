from typing import Tuple,Optional, Union
from pydantic import BaseModel

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from dotenv import load_dotenv, find_dotenv
from deepeval.evaluate import AsyncConfig
from deepeval.models import OllamaModel
import os
import deepeval
from ollama import ChatResponse

deepeval.login(api_key=os.getenv("DeepEval_Key"))

class OllamaModelNoThink(OllamaModel):
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
            chat_model = self.load_model()
            messages = [{"role": "user", "content": prompt}]
          
            response: ChatResponse = chat_model.chat(
                model=self.name,
                messages=messages,
                format=schema.model_json_schema() if schema else None,
                options={
                    **{"temperature": self.temperature},
                    **self.generation_kwargs,
                },
                think=False
            )
            return (
                (
                    schema.model_validate_json(response.message.content)
                    if schema
                    else response.message.content
                ),
                0,
            )    
 
    async def a_generate(
            self, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Union[str, BaseModel], float]:
            chat_model = self.load_model(async_mode=True)   
            messages = [{"role": "user", "content": prompt}]
    
            response: ChatResponse = await chat_model.chat(
                model=self.name,
                messages=messages,
                format=schema.model_json_schema() if schema else None,
                options={
                    **{"temperature": self.temperature},
                    **self.generation_kwargs,
                },
                think=False
            )
            return (
                (
                    schema.model_validate_json(response.message.content)
                    if schema
                    else response.message.content
                ),
                0,
            )

Ollama_Model = OllamaModelNoThink(
    model=os.getenv("LOCAL_MODEL_NAME"),
    base_url=os.getenv("LOCAL_MODEL_BASE_URL")
)


load_dotenv(find_dotenv())

test_case1 = LLMTestCase(
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output="Paris."
)

test_case2 = LLMTestCase(
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output="London."
)

evaluate([test_case1, test_case2],
         metrics=[AnswerRelevancyMetric(model=Ollama_Model)],
         async_config=AsyncConfig(run_async=False)
         )
