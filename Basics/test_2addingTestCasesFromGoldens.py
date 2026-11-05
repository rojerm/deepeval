from pathlib import Path
import json
import os

import deepeval
from langchain_ollama import ChatOllama
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset, Golden


# Upload results to Confident AI (comment out to run fully local
# and save your 5 free test runs per week)
deepeval.login(api_key=os.getenv("DeepEval_Key"))


# How many questions to test. Keep it small while learning.
MAX_GOLDENS = 10

# dev.json lives in the project root, one level up from Basics/
DATA_FILE = Path(__file__).parent.parent / "dev.json"


# The judge model (grades the answers)
Ollama_Eval_Model = OllamaModel(
    model=os.getenv("LOCAL_MODEL_NAME"),
    base_url=os.getenv("LOCAL_MODEL_BASE_URL")
)

# The model being tested (answers the questions)
llm = ChatOllama(
    base_url=os.getenv("LOCAL_MODEL_BASE_URL"),
    model=os.getenv("LOCAL_MODEL_NAME"),
    reasoning=False
    )


def get_llm_output(prompt):
    return llm.invoke(prompt).content


# Build the dataset from goldens
#dataset = EvaluationDataset()
#dataset.pull(alias="Adversial QA Dataset")

def load_goldens(file_path, limit):
    """Read SQuAD-style JSON and return up to `limit` unique goldens."""
    with open(file_path, "r") as f:
        dev_data = json.load(f)

    goldens = []
    seen_questions = set()

    for article in dev_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]

                # Skip duplicate questions
                key = question.strip().lower()
                if key in seen_questions:
                    continue
                seen_questions.add(key)

                # First answer if there is one, otherwise None
                expected = qa["answers"][0]["text"] if qa["answers"] else None

                goldens.append(Golden(input=question, expected_output=expected))

                # Stop as soon as we have enough
                if len(goldens) >= limit:
                    return goldens

    return goldens

# Step 1: load goldens from the local file (instead of dataset.pull)
dataset = EvaluationDataset(goldens=load_goldens(DATA_FILE, MAX_GOLDENS))

print(f"Goldens loaded: {len(dataset.goldens)}")
assert len(dataset.goldens) > 0, "No goldens loaded! Check dev.json."


# Step 2: ask the model each question to turn goldens into test cases
for golden in dataset.goldens:
    dataset.add_test_case(
        LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=get_llm_output(golden.input),
        )
    )

print(f"Test cases created: {len(dataset.test_cases)}")


# Step 3: grade the answers
evaluate(
    test_cases=dataset.test_cases,
    metrics=[AnswerRelevancyMetric(model=Ollama_Eval_Model)],
    async_config=AsyncConfig(run_async=False),
)
