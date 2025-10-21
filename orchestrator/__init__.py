import asyncio
from typing import List
from utils.llm import BaseLLMService
from utils.constants import GPT_4O
from utils.logger import logger
from pydantic import BaseModel

from orchestrator.response_models import Entity, Relation, KnowledgeGraph, EntityType
from orchestrator.prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
    PERSONALITY_INFERENCE_SYSTEM_PROMPT,
)


class EntityExtractionResponse(BaseModel):
    entities: List[Entity]


class RelationExtractionResponse(BaseModel):
    relations: List[Relation]


class PersonalityInferenceResponse(BaseModel):
    personality_map: dict  # {entity_name: [traits]}

class KnowledgeGraphExtractor:
    """Extract a KnowledgeGraph (entities + relations) from text using LLM reasoning."""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

    async def extract(self, text: str) -> KnowledgeGraph:
        """Main pipeline — extract entities, relations, and enrich descriptions."""
        entities = await self._extract_entities(text)
        
        relations_task = asyncio.create_task(self._extract_relations(text, entities))
        entities_task = asyncio.create_task(self._infer_personalities(text, entities))

        relations, entities = await asyncio.gather(relations_task, entities_task)

        return KnowledgeGraph(nodes=entities, edges=relations)

    async def _extract_entities(self, text: str) -> List[Entity]:
        logger.debug("Extracting entities...")

        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        response: EntityExtractionResponse = await self.llm_service.call_llm_structured(
            model=GPT_4O,
            messages=messages,
            response_format=EntityExtractionResponse,
        )

        if not hasattr(response, 'entities') or not response.entities:
            logger.warning("No entities extracted from the text.")
            return []
        
        logger.debug(f"Extracted {len(response.entities)} entities.")

        return response.entities

    async def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        entity_names = ", ".join(e.name for e in entities)
        messages = [
            {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Entities: {entity_names}\n\nText:\n{text}"},
        ]

        logger.debug("Extracting relations...")

        response: RelationExtractionResponse = await self.llm_service.call_llm_structured(
            model=GPT_4O,
            messages=messages,
            response_format=RelationExtractionResponse,
        )

        if not hasattr(response, 'relations') or not response.relations:
            logger.warning("No relations extracted from the text.")
            return []

        logger.debug(f"Extracted {len(response.relations)} relations.")

        return response.relations

    async def _infer_personalities(self, text: str, entities: List[Entity]) -> List[Entity]:
        person_entities = [e for e in entities if e.type == EntityType.PERSON]
        if not person_entities:
            return entities  # skip if no people

        people_names = [p.name for p in person_entities]
        messages = [
            {"role": "system", "content": PERSONALITY_INFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"People: {people_names}\n\nText:\n{text}"},
        ]

        logger.debug("Inferring personality traits...")

        response: PersonalityInferenceResponse = await self.llm_service.call_llm_structured(
            model=GPT_4O,
            messages=messages,
            response_format=PersonalityInferenceResponse,
        )

        if not hasattr(response, 'personality_map') or not response.personality_map:
            logger.warning("No personality traits inferred.")
            return entities

        # attach traits to Entity.description
        for entity in entities:
            if entity.name in response.personality_map:
                traits = response.personality_map[entity.name]
                trait_text = f"Personality traits: {', '.join(traits)}"
                entity.description = (entity.description or "") + " " + trait_text

        return entities
