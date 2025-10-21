from pydantic import BaseModel, Field
from typing import List, Optional

class EntityRole(BaseModel):
    name: str = Field(..., description="Name of the entity")
    role: str = Field(..., description="Role of the entity in the document")
    traits: List[str] = Field(..., description="Key traits or characteristics of the entity")

class DocumentPlan(BaseModel):
    topic: str = Field(..., description="The main topic of the document")
    setting: str = Field(..., description="The setting or context of the document")
    entities: List[EntityRole] = Field(..., description="List of entities involved in the document")
    key_events: List[str] = Field(..., description="Key events or interactions between entities")
    tone: Optional[str] = "neutral"
    style: Optional[str] = "narrative"

class Document(BaseModel):
    content: str
    plan: DocumentPlan
    creation_timestamp: Optional[str]

