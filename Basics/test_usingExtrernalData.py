from langchain_ollama import ChatOllama
import os
import deepeval
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden
import json

deepeval.login(api_key=os.getenv("DeepEval_Key"))


Ollama_Eval_Model = OllamaModel(
    model=os.getenv("LOCAL_MODEL_NAME"),
    base_url=os.getenv("LOCAL_MODEL_BASE_URL")
)


llm = ChatOllama(
    base_url=os.getenv("LOCAL_MODEL_BASE_URL"),
    model=os.getenv("LOCAL_MODEL_NAME"),
    reasoning=False
    )


def get_llm_output(prompt):
    llm_output = llm.invoke(prompt)
    return llm_output.content


test_case1 = LLMTestCase(
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output= get_llm_output("What is the capital of France?")
)   

test_case2 = LLMTestCase(
    input="What is the capital of India?",
    expected_output="The capital of India is New Delhi.",
    actual_output= get_llm_output("What is the capital of India?")
)   

test_case3 = LLMTestCase(
    input="What is the capital of Canada?",
    expected_output="The capital of Canada is Ottawa.",
    actual_output= get_llm_output("What is the capital of Canada?")
)   

golden_test_data = [
        {
            "input": "What is the capital of Srilanka?",
            "expected_output": "The capital of Sri Lanka is Colombo."
        },    
        {
            "input": "What is the capital of South Africa?",
            "expected_output": "The capital of South Africa is Cape Town."
        }
]

# Build the dataset from goldens
Dataset = EvaluationDataset()

with open("dev.json", "r") as f:
    dev_data = json.load(f)

for article in dev_data['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph["qas"]:
            input_text = qa["question"]
            expected_output_text = qa["answers"][0]["text"] if qa["answers"] else "No answer provided.'"
            Dataset.add_golden(Golden(input=input_text, expected_output=expected_output_text))


print(f"Total number of goldens in the dataset: {len(Dataset.goldens)}")            

Dataset.push(alias="Adversial QA Dataset")
