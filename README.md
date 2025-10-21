# Persona Reseau - Automated Knowledge Graph Extraction

This project provides an **end-to-end knowledge graph extraction pipeline** along with a synthetic document generation pipeline.

---

## Setup

### Clone the Repository

```bash
git clone https://github.com/1rsh/persona-reseau.git
cd persona-reseau
```

### Environment Setup

```bash
make setup
```

Creates `.venv/` and installs dependencies from `requirements.txt`.

---

## Usage

Ensure your `.env` contains 
```txt
OPENAI_BASE_URL="https://api.openai.com/v1" # or any other OpenAI compatible base url
OPENAI_API_KEY="sk-xxxxxxxxxxxxxx"
```
Then, 
```bash
make run
```

Runs document generation, KG extraction, and evaluation in sequence.

Alternatively, you can run `data/generate.py` to generate synthetic documents and then `main.py` to extract knowledge graphs from those documents.

Finally, open the `.html` files in `extracted/` folder to visualize the knowledge graphs (hovering on nodes would give extra info).

---

## Example Output

```json
{
  "document_metadata": {
    "topic": "Biotech Startup",
    "entities": [
      {"name": "Dr. Emily Carter", "role": "Chief Scientist", "traits": ["visionary", "dedicated"]},
      {"name": "NanoHealth Inc.", "role": "Biotech Company"},
      {"name": "GreenTech Ventures", "role": "Investor"}
    ]
  },
  "knowledge_graph": {
    "nodes": [
      {"name": "Dr. Emily Carter", "type": "person"},
      {"name": "NanoHealth Inc.", "type": "organization"}
    ],
    "edges": [
      {"source": "Dr. Emily Carter", "relation": "leads", "target": "NanoHealth Inc."}
    ]
  },
  "evaluation": {
    "supervised_eval": {
      "entity_precision": 0.75,
      "entity_recall": 0.67,
      "entity_f1": 0.706,
      "trait_similarity": 0.62
    },
    "llm_eval": {
      "entity_coverage_score": 9,
      "relation_correctness_score": 8,
      "personality_coherence_score": 7,
      "factual_alignment_score": 9,
      "logical_consistency_score": 8,
      "overall_score": 8,
      "reasoning": "KG captures key actors and relations but omits secondary collaborations."
    }
  }
}
```
