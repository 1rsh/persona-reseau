import os
import json
import asyncio
from tqdm.asyncio import tqdm_asyncio

from data import GENERATION_DIRECTORY
from data.response_models import Document
from utils.llm import LLMService
from utils.constants import OPENAI
from utils.logger import logger
from orchestrator import KnowledgeGraphExtractor
from orchestrator.evaluate import EvaluationPipeline


OUTPUT_PATH = "extracted/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

MAX_CONCURRENT_TASKS = 5

async def process_document(document: Document, kg_extractor: KnowledgeGraphExtractor, evaluator: EvaluationPipeline, semaphore: asyncio.Semaphore):
    """
    Process one document:
    1️⃣ Extract KG
    2️⃣ Evaluate (supervised + LLM)
    3️⃣ Visualize
    4️⃣ Save combined output
    """
    async with semaphore:
        try:
            # --- 1️⃣ Extract Knowledge Graph ---
            extracted_kg = await kg_extractor.extract(document.content)

            # --- 2️⃣ Run Evaluations ---
            supervised_eval = evaluator.evaluate_supervised(document.model_dump(), extracted_kg.model_dump())
            llm_eval = await evaluator.evaluate_llm(document.content, extracted_kg.model_dump())

            evaluation_results = {
                "supervised_eval": supervised_eval,
                "llm_eval": llm_eval
            }

        except Exception as e:
            logger.error(f"❌ Processing failed for document created at {document.creation_timestamp}: {e}")
            evaluation_results = {"error": str(e)}
            extracted_kg = None

        # --- 3️⃣ Visualization (optional) ---
        if extracted_kg:
            try:
                extracted_kg.visualize(
                    output_file=f"{OUTPUT_PATH}/kg_{document.creation_timestamp.replace(' ', '_').replace(':', '-')}.html"
                )
            except AssertionError:
                logger.warning(f"Visualization failed for document created at {document.creation_timestamp}")
            except ImportError:
                logger.warning("PyVis not installed; skipping visualization.")

        # --- 4️⃣ Save combined output ---
        output_data = {
            "document_metadata": document.model_dump(),
            "knowledge_graph": extracted_kg.model_dump() if extracted_kg else None,
            "evaluation": evaluation_results
        }

        out_file = f"{OUTPUT_PATH}/kg_{document.creation_timestamp.replace(' ', '_').replace(':', '-')}.json"
        with open(out_file, "w") as f:
            json.dump(output_data, f, indent=4)

        logger.debug(f"Saved extracted results to {out_file}")
        return out_file


async def main():
    # --- Initialize services ---
    llm_service = LLMService(OPENAI)
    kg_extractor = KnowledgeGraphExtractor(llm_service=llm_service)
    evaluator = EvaluationPipeline(llm_service=llm_service)

    # --- Load synthetic documents ---
    synthetic_documents: list[Document] = [
        Document(**json.load(open(f"{GENERATION_DIRECTORY}/{file}", "r")))
        for file in os.listdir(GENERATION_DIRECTORY)
        if file.endswith(".json")
    ]

    # --- Create semaphore for concurrency control ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # --- Create async tasks ---
    tasks = [
        process_document(doc, kg_extractor, evaluator, semaphore)
        for doc in synthetic_documents
    ]

    logger.info(f"Launching {len(tasks)} document pipelines in parallel (max {MAX_CONCURRENT_TASKS} concurrent)...")
    results = await tqdm_asyncio.gather(*tasks, desc="Processing all documents", colour="green")
    logger.info(f"Completed {len(results)} documents.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
