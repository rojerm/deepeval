from pathlib import Path
import json
import os

import deepeval
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


MAX_GOLDENS = 10
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "dev.json"


def load_goldens(file_path: Path, limit: int) -> list[Golden]:
    """Load up to `limit` unique question-answer pairs from a SQuAD JSON file."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    with file_path.open(encoding="utf-8") as file:
        dev_data = json.load(file)

    goldens = []
    seen_questions = set()

    for article in dev_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]

                key = question.strip().casefold()
                if key in seen_questions:
                    continue

                if not qa["answers"]:
                    continue

                seen_questions.add(key)
                expected = qa["answers"][0]["text"]
                goldens.append(Golden(input=question, expected_output=expected))

                if len(goldens) >= limit:
                    return goldens

    return goldens


def main():
    load_dotenv(PROJECT_ROOT / ".env.local")

    api_key = os.getenv("DeepEval_Key")
    if api_key:
        deepeval.login(api_key=api_key)

    model_name = os.getenv("LOCAL_MODEL_NAME")
    model_base_url = os.getenv("LOCAL_MODEL_BASE_URL")
    if not model_name or not model_base_url:
        raise ValueError(
            "Set LOCAL_MODEL_NAME and LOCAL_MODEL_BASE_URL in .env.local."
        )

    answer_model = ChatOllama(
        base_url=model_base_url,
        model=model_name,
        reasoning=False,
    )

    judge_model = OllamaModel(
        model=model_name,
        base_url=model_base_url,
    )

    dataset = EvaluationDataset(goldens=load_goldens(DATA_FILE, MAX_GOLDENS))
    if not dataset.goldens:
        raise ValueError(f"No answerable questions found in {DATA_FILE}.")

    print(f"Goldens loaded: {len(dataset.goldens)}")

    for golden in dataset.goldens:
        response = answer_model.invoke(golden.input)
        dataset.add_test_case(
            LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=response.content,
            )
        )

    print(f"Test cases created: {len(dataset.test_cases)}")

    evaluate(
        test_cases=dataset.test_cases,
        metrics=[AnswerRelevancyMetric(model=judge_model)],
        async_config=AsyncConfig(run_async=False),
    )


if __name__ == "__main__":
    main()