from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from utils.logger import logger

class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    EVENT = "event"
    LOCATION = "location"
    CONCEPT = "concept"
    UNKNOWN = "unknown"

class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    type: EntityType = Field(..., description="Type of the entity")
    description: Optional[str] = Field(None, description="Brief description of the entity")

class Relation(BaseModel):
    source: str = Field(..., description="Source entity of the relation")
    relation: str = Field(..., description="Type of relation")
    target: str = Field(..., description="Target entity of the relation")

class PersonalityTrait(BaseModel):
    entity: str = Field(..., description="Name of the entity")
    traits: List[str] = Field(..., description="List of personality traits associated with the entity")

class KnowledgeGraph(BaseModel):
    nodes: List[Entity] = Field(..., description="List of entities in the knowledge graph")
    edges: List[Relation] = Field(..., description="List of relations between entities")

    def __str__(self):
        node_str = ',\n    '.join(repr(node) for node in self.nodes)
        edge_str = ',\n    '.join(repr(edge) for edge in self.edges)
        return (
            f"{self.__class__.__name__}(\n"
            f"  nodes=[\n    {node_str}\n  ],\n"
            f"  edges=[\n    {edge_str}\n  ]\n"
            f")"
        )

    def visualize(self, output_file: str = "knowledge_graph.html", notebook: bool = False, attempt_correction: bool = True):
        """
        Visualize the knowledge graph interactively using PyVis.
        
        Args:
            output_file (str): Path to save the HTML visualization.
            notebook (bool): If True, shows directly in Jupyter Notebook.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis not installed. Run `pip install pyvis networkx`")

        # Initialize PyVis graph
        net = Network(height="750px", width="100%", directed=True, bgcolor="#0e1117", font_color="white")
        net.barnes_hut()  # physics layout

        # Define color mapping by entity type
        color_map = {
            "person": "#ff6b6b",
            "organization": "#4ecdc4",
            "product": "#ffe66d",
            "event": "#ff9f43",
            "location": "#54a0ff",
            "concept": "#c56cf0",
            "unknown": "#d2dae2",
        }

        try:
            # Add nodes
            for node in self.nodes:
                label = f"{node.name}\n({node.type.value})"
                title = node.description or ""
                net.add_node(
                    node.name,
                    label=label,
                    title=title,
                    color=color_map.get(node.type.value, "#d2dae2"),
                )

            # Add edges
            for edge in self.edges:
                net.add_edge(edge.source, edge.target, label=edge.relation)
        except AssertionError:
            if attempt_correction:
                logger.warning("AssertionError encountered during visualization. Attempting to correct...")
                return self._attempt_correction(output_file, notebook)
            else:
                raise

        if notebook:
            return net.show(output_file)
        else:
            net.save_graph(output_file)
            logger.info(f"Knowledge graph saved to {output_file}")
    
    def _attempt_correction(self, output_file: str, notebook: bool):
        """Attempt to correct common issues in the graph and re-visualize."""
        # Simple correction: remove edges with missing nodes
        valid_node_names = {node.name for node in self.nodes}
        corrected_edges = [
            edge for edge in self.edges
            if edge.source in valid_node_names and edge.target in valid_node_names
        ]
        if len(corrected_edges) < len(self.edges):
            logger.debug(f"Removed {len(self.edges) - len(corrected_edges)} edges with missing nodes.")
            self.edges = corrected_edges
        else:
            logger.debug("No edges were removed; unable to correct the graph.")
        
        # Retry visualization without further correction attempts
        self.visualize(output_file=output_file, notebook=notebook, attempt_correction=False)


class LLMJudgeEvalResponse(BaseModel):
    entity_coverage_score: int = Field(..., ge=0, le=10, description="How well the KG captures key entities")
    relation_correctness_score: int = Field(..., ge=0, le=10, description="How accurate and text-supported relations are")
    personality_coherence_score: int = Field(..., ge=0, le=10, description="How consistent personality traits are with the text")
    factual_alignment_score: int = Field(..., ge=0, le=10, description="How factually faithful the KG is to the document content")
    logical_consistency_score: int = Field(..., ge=0, le=10, description="Whether the KG’s relationships make logical sense together")
    overall_score: int = Field(..., ge=0, le=10, description="Overall weighted impression of quality and accuracy")
    reasoning: str = Field(..., description="Brief reasoning with evidence for deductions, highlighting missing or incorrect aspects")

    def __str__(self):
        return f"{self.__class__.__name__}(entity_coverage_score={self.entity_coverage_score}, relation_correctness_score={self.relation_correctness_score}, personality_coherence_score={self.personality_coherence_score}, factual_alignment_score={self.factual_alignment_score}, logical_consistency_score={self.logical_consistency_score}, overall_score={self.overall_score}, reasoning='{self.reasoning}')"