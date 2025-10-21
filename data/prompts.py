DOCUMENT_PLAN_SYSTEM_PROMPT = """
You are an expert scenario architect for synthetic document generation.

Your task is to design realistic, diverse, and contextually rich fictional scenarios 
that could naturally exist in a wide range of real-world domains. 
Each scenario should provide a strong foundation for generating varied, human-like documents.

Guidelines:
1. Include **3–6 entities** spanning multiple types (e.g., people, organizations, places, technologies, events).
2. Establish at least **3 significant relationships** or interactions between them 
   (e.g., collaboration, conflict, discovery, negotiation, betrayal, or alliance).
3. Make each scenario **plausible but original**, avoiding clichés or repetitive tropes.
4. Include details about:
   - Setting (time, place, or socio-economic backdrop)
   - Each entity’s role and motivation
   - The primary tension, challenge, or opportunity connecting them
5. Ensure there’s potential for rich narrative development (multiple perspectives, emotional tone, ethical or strategic conflict).

Return structured JSON with the following keys:
{
  "topic": "...",
  "setting": "...",
  "entities": [
    {"name": "...", "role": "...", "traits": ["..."]},
    ...
  ],
  "key_events": ["...", "..."],
  "tone": "...",
  "style": "..."
}
"""


DOCUMENT_PLAN_USER_PROMPT = """
Generate a **fictional yet realistic** scenario with strong narrative potential.

Following are some example themes:
- Cybersecurity & Espionage
- Climate Policy & Industry Conflict
- Financial Fraud & Whistleblowing
- Urban Redevelopment & Displacement
- AI Regulation & Corporate Rivalry
- Space Exploration & Political Competition
- Sports Doping & Media Pressure
- International Diplomacy & Espionage
- Cultural Heritage & Technology

First think of a new theme not listed above, then create a unique scenario around it.

Each scenario must:
- Contain **3–6 distinct entities** (people, organizations, locations, or events)
- Feature at least **3 meaningful relationships** (e.g., partnership, rivalry, mentorship, investigation)
- Include **one underlying tension** (ethical, strategic, or emotional)
- Be suitable for generating an engaging, multi-perspective narrative

Return only structured JSON following the expected schema.
"""

DOCUMENT_COMPOSE_SYSTEM_PROMPT = """
You are an expert creative writer and document composer.

Your goal is to transform structured scenario data into a vivid, human-like narrative.  
Each output should be **3–5 cohesive paragraphs** written in a natural, varied style.

Guidelines:
1. **Variety in tone:** The narrative can be investigative, journalistic, analytical, introspective, or dramatized — 
   vary this style across generations.
2. **Show, don’t tell:** Reveal relationships and traits through dialogue, actions, or subtle details 
   instead of explicit statements.
3. **Weave perspectives:** Optionally use multiple viewpoints (e.g., a scientist, a journalist, an investor, or a citizen).
4. **Include temporal and spatial grounding** (e.g., “In late 2027, in the heart of Nairobi’s innovation district…”).
5. **Use implicit meaning:** Let readers infer relationships and emotions from tone and context.
6. Avoid listing facts mechanically — focus on narrative flow and realism.

Your writing should feel human, spontaneous, and distinct from previous samples.

Return **only the full text** of the composed document, not JSON.
"""
