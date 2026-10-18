import json
import logging
import os
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.genai.scorers import Correctness, RelevanceToQuery

from jurisrag_arabic.rag import RAG, QueryRequest
from jurisrag_arabic.rag_config import load_rag_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

config = load_rag_config()

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

logger.info("Configured MLflow experiment: %s", experiment_name)

# Set the evaluator model environment variables so the judges use Groq instead of OpenAI.
# os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "1"
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

JUDGE = "openai:/openai/gpt-oss-20b"
rag_service = RAG()


def prepare_eval_dataset(lang: str = "ar") -> list[dict]:
    """
    Build records for MLflow GenAI evaluation.
    Each record contains:
      inputs.question
      expectations.expected_response
      expectations.reference_context
    """

    if lang not in {"ar", "en"}:
        raise ValueError("lang must be 'ar' or 'en'")

    dataset_path = Path(__file__).resolve().parent / "eval_test.json"

    logger.info("Loading evaluation dataset: %s", dataset_path)

    with dataset_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    dataset = []

    for index, item in enumerate(raw_data, start=1):
        if lang == "ar":
            question = item["input"]["question_ar"]
            expected_answer = item["expectation"]["expected_ar"]
            reference_context = item["ground_truth"]["text_ar"]
        else:
            question = item["input"]["question_en"]
            expected_answer = item["expectation"]["expected_en"]
            reference_context = item["ground_truth"]["text_en"]

        dataset.append(
            {
                "inputs": {
                    "question": question  # Key matches predict_rag argument name
                },
                "expectations": {
                    "expected_response": expected_answer,
                    "reference_context": reference_context,
                },
            }
        )

    logger.info("Prepared %d evaluation examples for language '%s'", len(dataset), lang)

    return dataset


@mlflow.trace
def predict_rag(question: str) -> str:
    """
    Run one question through JurisRAG and return its answer.
    The parameter name matches inputs.question in the dataset.
    """

    request = QueryRequest(query=question)
    response = rag_service.ask(request)
    time.sleep(3)

    return response


def run_eval(eval_lang: str = "ar") -> None:
    logger.info("Starting JurisRAG evaluation (language=%s)", eval_lang)

    dataset = prepare_eval_dataset(eval_lang)[:5]

    scorers = [Correctness(model=JUDGE), RelevanceToQuery(model=JUDGE)]

    with mlflow.start_run(run_name=f"rag-evaluation-{eval_lang}"):
        model_config = {
            "model_name": config["llm"]["model_name"],
            "temperature": config["llm"]["temperature"],
            "top_k": config["embedding_model"]["top_k"],
            "max_tokens": config["embedding_model"]["chunk_size"],
            "embedding_model_name": config["embedding_model"]["model_name"],
        }

        # register prompt
        registered_prompt = mlflow.genai.register_prompt(
            name="jurisrag_prompt",
            template=config["user_message_template"],
            model_config=model_config,
            tags={
                "task": "rag_evaluation",
                "evaluation_language": eval_lang,
            },
        )
        # mlflow.log_param("system_prompt", config["system_prompt"])
        mlflow.genai.set_prompt_alias(
            name="jurisrag_prompt",
            alias="development",
            version=registered_prompt.version,
        )

        logger.info("Evaluating %d examples", len(dataset))

        results = mlflow.genai.evaluate(
            data=dataset, predict_fn=predict_rag, scorers=scorers
        )

        logger.info("Evaluation completed.\nResults:\n%s", results)

    logger.info("Finished JurisRAG evaluation.")


if __name__ == "__main__":
    run_eval(eval_lang="ar")
