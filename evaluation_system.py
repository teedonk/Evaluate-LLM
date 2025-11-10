"""
 Complete LLM Evaluation Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from datetime import datetime
import logging

from abc import ABC, abstractmethod

import re
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationStatus(Enum):
    """Evaluation result status"""
    PASSED = "passed"
    FAILED = "failed"
    FLAGGED = "flagged"  # Needs human review
    ERROR = "error"


@dataclass
class EvaluationResult:
    """Result from a single evaluation check"""
    metric_name: str
    score: float  # 0.0 to 1.0
    passed: bool
    confidence: float  # How confident we are in this result
    reasoning: str
    status: EvaluationStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'metric_name': self.metric_name,
            'score': self.score,
            'passed': self.passed,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'status': self.status.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class EvaluationContext:
    """Context needed for evaluation"""
    query: str
    response: str
    reference_docs: Optional[str] = None
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated results from all evaluators"""
    overall_score: float
    overall_status: EvaluationStatus
    individual_results: Dict[str, EvaluationResult]
    flagged_for_review: bool
    confidence: float
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)




class BaseEvaluator(ABC):
    """Abstract base class for all evaluators"""
    
    def __init__(self, name: str, weight: float = 1.0, critical: bool = False):
        """
        Args:
            name: Evaluator identifier
            weight: Weight in final score calculation
            critical: If True, failure blocks overall pass
        """
        self.name = name
        self.weight = weight
        self.critical = critical
    
    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Evaluate the response"""
        pass
    
    def should_run(self, context: EvaluationContext) -> bool:
        """Override to conditionally skip evaluation"""
        return True
    



class LengthValidator(BaseEvaluator):
    """Validates response length is appropriate"""
    
    def __init__(self, min_length: int = 10, max_length: int = 5000):
        super().__init__(name="length_validation", weight=0.5)
        self.min_length = min_length
        self.max_length = max_length
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        length = len(context.response)
        
        if length < self.min_length:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                confidence=1.0,
                reasoning=f"Response too short: {length} chars (min: {self.min_length})",
                status=EvaluationStatus.FAILED,
                metadata={'length': length}
            )
        
        if length > self.max_length:
            return EvaluationResult(
                metric_name=self.name,
                score=0.5,
                passed=False,
                confidence=1.0,
                reasoning=f"Response too long: {length} chars (max: {self.max_length})",
                status=EvaluationStatus.FLAGGED,
                metadata={'length': length}
            )
        
        # Score based on reasonable length
        optimal_length = (self.min_length + self.max_length) / 2
        score = 1.0 - abs(length - optimal_length) / optimal_length * 0.3
        score = max(0.7, min(1.0, score))
        
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            passed=True,
            confidence=1.0,
            reasoning=f"Length appropriate: {length} chars",
            status=EvaluationStatus.PASSED,
            metadata={'length': length}
        )


class EntityHallucinationDetector(BaseEvaluator):
    """Detects entity hallucinations using spaCy"""
    
    def __init__(self):
        super().__init__(name="entity_hallucination", weight=1.5, critical=True)
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def should_run(self, context: EvaluationContext) -> bool:
        return self.nlp is not None and context.reference_docs is not None
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not self.should_run(context):
            return self._skip_result()
        
        response_entities = self._extract_entities(context.response)
        reference_entities = self._extract_entities(context.reference_docs)
        
        if not response_entities:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                confidence=0.8,
                reasoning="No named entities in response to verify",
                status=EvaluationStatus.PASSED
            )
        
        # Check for unsupported entities
        unsupported = response_entities - reference_entities
        support_rate = 1 - (len(unsupported) / len(response_entities))
        
        # Be lenient with common entities (dates, generic terms)
        filtered_unsupported = [e for e in unsupported if not self._is_generic_entity(e)]
        
        if filtered_unsupported:
            score = max(0.3, support_rate)
            passed = len(filtered_unsupported) <= 1  # Allow 1 unsupported entity
            status = EvaluationStatus.FLAGGED if not passed else EvaluationStatus.PASSED
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                confidence=0.85,
                reasoning=f"Found {len(filtered_unsupported)} unsupported entities: {', '.join(list(filtered_unsupported)[:3])}",
                status=status,
                metadata={
                    'unsupported_entities': list(filtered_unsupported),
                    'total_entities': len(response_entities),
                    'support_rate': support_rate
                }
            )
        
        return EvaluationResult(
            metric_name=self.name,
            score=1.0,
            passed=True,
            confidence=0.9,
            reasoning=f"All {len(response_entities)} entities supported by reference",
            status=EvaluationStatus.PASSED,
            metadata={'total_entities': len(response_entities)}
        )
    
    def _extract_entities(self, text: str) -> set:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entities.add(ent.text.lower().strip())
        
        return entities
    
    def _is_generic_entity(self, entity: str) -> bool:
        """Check if entity is generic/common"""
        generic_terms = {'today', 'yesterday', 'tomorrow', 'now', 'currently', 'recently'}
        return entity.lower() in generic_terms or len(entity) <= 2
    
    def _skip_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name,
            score=0.5,
            passed=True,
            confidence=0.0,
            reasoning="Evaluator skipped (missing dependencies or context)",
            status=EvaluationStatus.PASSED
        )


class SemanticConsistencyChecker(BaseEvaluator):
    """Checks internal consistency using sentence embeddings"""
    
    def __init__(self):
        super().__init__(name="semantic_consistency", weight=1.0)
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.model = None
    
    def should_run(self, context: EvaluationContext) -> bool:
        return self.model is not None and len(context.response) > 100
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not self.should_run(context):
            return self._skip_result()
        
        sentences = self._split_sentences(context.response)
        
        if len(sentences) < 2:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                confidence=0.5,
                reasoning="Response too short for consistency check",
                status=EvaluationStatus.PASSED
            )
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Calculate pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = []
        contradictions = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
                
                # Very low similarity might indicate contradiction
                if sim < 0.2:
                    contradictions.append((sentences[i][:80], sentences[j][:80]))
        
        min_sim = min(similarities)
        avg_sim = np.mean(similarities)
        
        # Score based on minimum similarity (catches contradictions)
        score = max(0.0, min(1.0, (min_sim + 0.5) / 1.5))
        passed = min_sim > 0.15 and len(contradictions) == 0
        
        if not passed:
            status = EvaluationStatus.FLAGGED
            reasoning = f"Potential contradictions detected (min_sim={min_sim:.2f})"
        else:
            status = EvaluationStatus.PASSED
            reasoning = f"Response internally consistent (min_sim={min_sim:.2f}, avg={avg_sim:.2f})"
        
        return EvaluationResult(
            metric_name=self.name,
            score=score,
            passed=passed,
            confidence=0.75,
            reasoning=reasoning,
            status=status,
            metadata={
                'min_similarity': float(min_sim),
                'avg_similarity': float(avg_sim),
                'potential_contradictions': contradictions[:2]
            }
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _skip_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name,
            score=0.5,
            passed=True,
            confidence=0.0,
            reasoning="Evaluator skipped",
            status=EvaluationStatus.PASSED
        )
    



class NLIConsistencyChecker(BaseEvaluator):
    """Uses NLI model to check entailment with reference"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-small"):
        super().__init__(name="nli_consistency", weight=2.0, critical=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"NLI model loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}")
            self.model = None
    
    def should_run(self, context: EvaluationContext) -> bool:
        return self.model is not None and context.reference_docs is not None
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if not self.should_run(context):
            return self._skip_result()
        
        # Split response into claims
        sentences = self._split_sentences(context.response)
        
        entailment_scores = []
        contradictions = []
        neutral_claims = []
        
        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            score, label = self._check_entailment(context.reference_docs, sentence)
            entailment_scores.append(score)
            
            if label == "contradiction":
                contradictions.append(sentence[:100])
            elif label == "neutral" and score < 0.3:
                neutral_claims.append(sentence[:100])
        
        if not entailment_scores:
            return EvaluationResult(
                metric_name=self.name,
                score=0.5,
                passed=True,
                confidence=0.3,
                reasoning="No claims to verify",
                status=EvaluationStatus.PASSED
            )
        
        avg_score = sum(entailment_scores) / len(entailment_scores)
        
        # Determine pass/fail
        has_contradictions = len(contradictions) > 0
        mostly_supported = avg_score > 0.5
        
        if has_contradictions:
            passed = False
            status = EvaluationStatus.FAILED
            reasoning = f"Found {len(contradictions)} contradictory claims"
        elif not mostly_supported:
            passed = False
            status = EvaluationStatus.FLAGGED
            reasoning = f"Low entailment score: {avg_score:.2f}"
        else:
            passed = True
            status = EvaluationStatus.PASSED
            reasoning = f"Claims supported (entailment={avg_score:.2f})"
        
        return EvaluationResult(
            metric_name=self.name,
            score=avg_score,
            passed=passed,
            confidence=0.85,
            reasoning=reasoning,
            status=status,
            metadata={
                'entailment_score': avg_score,
                'contradictions': contradictions[:3],
                'neutral_claims': neutral_claims[:2],
                'total_claims_checked': len(entailment_scores)
            }
        )
    
    def _check_entailment(self, premise: str, hypothesis: str) -> Tuple[float, str]:
        """Check if hypothesis is entailed by premise"""
        
        inputs = self.tokenizer(
            premise[:512],  # Truncate to avoid overflow
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
        
        # Labels: typically [contradiction, neutral, entailment] or [entailment, neutral, contradiction]
        # Check model config for exact order
        labels = ["contradiction", "neutral", "entailment"]
        label_idx = torch.argmax(probs).item()
        
        # Return entailment probability
        entailment_idx = 2 if labels[2] == "entailment" else 0
        entailment_score = probs[entailment_idx].item()
        
        return entailment_score, labels[label_idx]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _skip_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name,
            score=0.5,
            passed=True,
            confidence=0.0,
            reasoning="Evaluator skipped",
            status=EvaluationStatus.PASSED
        )