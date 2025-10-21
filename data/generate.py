import os
import time
import json
from tqdm import tqdm

from utils.llm import BaseLLMService, LLMService
from utils.constants import GPT_4O, OPENAI
from utils.logger import logger

from .__init__ import GENERATION_DIRECTORY
from .prompts import DOCUMENT_PLAN_SYSTEM_PROMPT, DOCUMENT_PLAN_USER_PROMPT, DOCUMENT_COMPOSE_SYSTEM_PROMPT
from .response_models import Document, DocumentPlan

class DocumentGenerator:
    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service
    
    async def generate(self) -> Document:
        # Generate document outline (plan, tone, style)
        try:
            messages_plan = [
                {"role": "system", "content": DOCUMENT_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": DOCUMENT_PLAN_USER_PROMPT}
            ]

            logger.debug("Generating document plan...")

            plan: DocumentPlan = await self.llm_service.call_llm_structured(
                model=GPT_4O,
                messages=messages_plan,
                response_format=DocumentPlan
            )

            logger.debug(f"Document plan generated. {plan}")

            # Compose the final document based on the generated plan
            compose_prompt = f"""
            Convert the following structured scenario into a coherent multi-entity document:
            {plan.model_dump()}
            """
            messages_compose = [
                {"role": "system", "content": DOCUMENT_COMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": compose_prompt}
            ]

            logger.debug("Composing final document...")

            document: str = await self.llm_service.call_llm(
                model=GPT_4O,
                messages=messages_compose
            )

            logger.debug("Document composition complete.")

            return Document(
                content=document,
                plan=plan,
                creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            raise e


if __name__ == "__main__":
    import asyncio

    os.makedirs(GENERATION_DIRECTORY, exist_ok=True)

    llm_service = LLMService(name=OPENAI)
    document_generator = DocumentGenerator(llm_service=llm_service)

    num_documents = input("Enter the number of documents to generate: ")

    async def main():
        tasks = [document_generator.generate() for _ in range(int(num_documents))]

        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), colour="green", desc="Generating documents"):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                continue

        return results
    
    generated_documents = asyncio.run(main())
    
    # Save generated documents to files
    for idx, doc in enumerate(generated_documents):
        with open(f"{GENERATION_DIRECTORY}/document_{idx+1}.json", "w") as f:
            json.dump(doc.model_dump(), f, indent=4)

