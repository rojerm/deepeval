from langchain_ollama import ChatOllama
import os
import deepeval
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden


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
goldens = [Golden(input=data["input"], expected_output=data["expected_output"]) for data in golden_test_data]
dataset = EvaluationDataset(goldens=goldens)
# dataset.push(alias="Capital Cities")

# Turn each golden into a real test case by asking your model
for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=get_llm_output(golden.input),
        )
    )

# Add your hand-built ones too (once, outside any loop)
for tc in [test_case1, test_case2, test_case3]:
    dataset.add_test_case(tc)

evaluate(
    test_cases=dataset.test_cases,
    metrics=[AnswerRelevancyMetric(model=Ollama_Eval_Model)],
    async_config=AsyncConfig(run_async=False),
)