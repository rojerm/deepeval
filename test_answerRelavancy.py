from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

test_case1 = LLMTestCase(
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output="Paris."
)

evaluate([test_case1],metrics=[AnswerRelevancyMetric()])
