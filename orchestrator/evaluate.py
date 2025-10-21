from typing import List, Dict, Any
from difflib import SequenceMatcher

from utils.llm import BaseLLMService
from utils.constants import GPT_4O
from utils.logger import logger

from .response_models import LLMJudgeEvalResponse


class EvaluationPipeline:
    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

    # ---------------------- SUPERVISED EVAL ----------------------
    def evaluate_supervised(self, document: Dict[str, Any], generated_kg: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare synthetic document 'plan' vs generated KG entities & traits.
        Returns similarity metrics.
        """
        gt_entities = [e["name"].lower() for e in document["plan"]["entities"]]
        gen_entities = [e["name"].lower() for e in generated_kg["nodes"]]

        # Entity match (using fuzzy ratio > 0.8)
        entity_matches = self._fuzzy_match(gt_entities, gen_entities)
        entity_recall = len(entity_matches) / len(gt_entities) if gt_entities else 0
        entity_precision = len(entity_matches) / len(gen_entities) if gen_entities else 0

        # Personality/traits comparison
        gt_traits = {e["name"].lower(): e["traits"] for e in document["plan"]["entities"]}
        gen_traits = self._extract_traits(generated_kg)
        trait_similarity = self._trait_similarity(gt_traits, gen_traits)

        result = {
            "entity_precision": round(entity_precision, 3),
            "entity_recall": round(entity_recall, 3),
            "entity_f1": round(2 * entity_precision * entity_recall / (entity_precision + entity_recall + 1e-9), 3),
            "trait_similarity": round(trait_similarity, 3)
        }

        logger.debug(f"Supervised Evaluation Results: {result}")

        return result

    # ---------------------- LLM-AS-A-JUDGE EVAL ----------------------
    async def evaluate_llm(self, document_text: str, generated_kg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask an LLM to strictly assess the quality of the generated KG relative to the input text.
        Returns a structured response using call_llm_structured().
        """

        system_prompt = """
        You are a critical evaluator specializing in verifying Knowledge Graph quality.

        Evaluate the given Knowledge Graph against the input document based on the following strict criteria:

        1. **Entity Coverage (0–10)** — Did the graph capture ALL important entities (people, organizations, products, events, concepts)? 
           Deduct points for missing or redundant nodes.

        2. **Relation Correctness (0–10)** — Are the relations logically and semantically valid, AND directly or implicitly supported by the text?
           Deduct points for:
           - hallucinated relations
           - vague or generic relations (“associated_with” without context)
           - missing key causal or ownership relations

        3. **Personality Coherence (0–10)** — For human entities, are inferred personality traits justified by actions or tone in the text?
           Deduct points if traits are unsupported, generic, or inconsistent.

        4. **Factual Alignment (0–10)** — Does every node and edge correspond to something *factually* supported by the text?
           Deduct points for fabricated or misrepresented details.

        5. **Logical Consistency (0–10)** — Do all parts of the KG make sense together (no contradictions or cyclic inconsistencies)?

        Finally, compute **Overall Score (0–10)** as a holistic judgment, considering all aspects with a harsher weighting on factual accuracy.

        In your reasoning, cite at least one example for each major deduction.
        Return strictly structured JSON matching the required schema.
        """

        prompt = f"""
        DOCUMENT:
        {document_text}

        GENERATED_KNOWLEDGE_GRAPH:
        {generated_kg}
        """

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()}
        ]

        # ✅ Use structured LLM call
        eval_response: LLMJudgeEvalResponse = await self.llm_service.call_llm_structured(
            model=GPT_4O,
            messages=messages,
            response_format=LLMJudgeEvalResponse
        )

        logger.debug(f"LLM Judge Evaluation Response: {eval_response}")

        return eval_response.model_dump()

    # ---------------------- Helper Functions ----------------------

    def _fuzzy_match(self, gt_list, gen_list, threshold=0.8):
        matches = []
        for gt in gt_list:
            for gen in gen_list:
                ratio = SequenceMatcher(None, gt, gen).ratio()
                if ratio >= threshold:
                    matches.append((gt, gen))
                    break
        return matches

    def _extract_traits(self, kg: Dict[str, Any]) -> Dict[str, List[str]]:
        traits = {}
        for node in kg.get("nodes", []):
            desc = node.get("description", "")
            if "Personality traits:" in desc:
                entity = node["name"].lower()
                traits_list = desc.split("Personality traits:")[-1].split(",")
                traits[entity] = [t.strip() for t in traits_list if t.strip()]
        return traits

    def _trait_similarity(self, gt_traits, gen_traits):
        if not gt_traits:
            return 0.0

        sims = []
        for entity, gt_list in gt_traits.items():
            gen_list = gen_traits.get(entity, [])
            if not gen_list:
                continue
            overlap = len(set(gt_list).intersection(set(gen_list)))
            union = len(set(gt_list).union(set(gen_list)))
            sims.append(overlap / union if union > 0 else 0)
        return float(sum(sims)/len(sims)) if sims else 0.0
