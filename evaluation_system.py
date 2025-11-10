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

    
from pydantic import BaseModel, Field

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd

from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report
from scipy.stats import pearsonr, spearmanr
import numpy as np


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



class FactualityAssessment(BaseModel):
    """Structured output for factuality evaluation"""
    overall_score: int = Field(ge=1, le=5, description="Overall factuality score 1-5")
    has_hallucination: bool = Field(description="Whether response contains hallucinations")
    accuracy: int = Field(ge=1, le=5, description="Factual accuracy score")
    relevance: int = Field(ge=1, le=5, description="Relevance to query")
    completeness: int = Field(ge=1, le=5, description="Completeness of answer")
    reasoning: str = Field(description="Detailed reasoning for scores")
    problematic_claims: List[str] = Field(default_factory=list, description="Specific problematic claims")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class LLMAsJudgeEvaluator(BaseEvaluator):
    """Comprehensive evaluation using Claude/GPT-4 as judge"""
    
    def __init__(self, api_key: str, provider: str = "anthropic", model: str = None):
        super().__init__(name="llm_judge_factuality", weight=3.0, critical=True)
        
        self.provider = provider
        
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Comprehensive factuality evaluation"""
        
        prompt = self._build_prompt(context)
        
        try:
            if self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                response = self._call_openai(prompt)
            
            # Parse structured output
            assessment = self._parse_response(response)
            
            # Normalize score to 0-1
            normalized_score = assessment.overall_score / 5.0
            
            # Determine confidence
            confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.5}
            confidence = confidence_map.get(assessment.confidence.lower(), 0.7)
            
            # Determine pass/fail
            passed = (
                assessment.overall_score >= 4 and
                not assessment.has_hallucination and
                assessment.accuracy >= 4
            )
            
            if assessment.has_hallucination:
                status = EvaluationStatus.FAILED
            elif assessment.overall_score < 3:
                status = EvaluationStatus.FAILED
            elif assessment.overall_score == 3:
                status = EvaluationStatus.FLAGGED
            else:
                status = EvaluationStatus.PASSED
            
            return EvaluationResult(
                metric_name=self.name,
                score=normalized_score,
                passed=passed,
                confidence=confidence,
                reasoning=assessment.reasoning,
                status=status,
                metadata={
                    'accuracy': assessment.accuracy,
                    'relevance': assessment.relevance,
                    'completeness': assessment.completeness,
                    'problematic_claims': assessment.problematic_claims,
                    'has_hallucination': assessment.has_hallucination
                }
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                passed=False,
                confidence=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                status=EvaluationStatus.ERROR,
                metadata={'error': str(e)}
            )
    
    def _build_prompt(self, context: EvaluationContext) -> str:
        """Build comprehensive evaluation prompt"""
        
        prompt = f"""You are an expert evaluator assessing AI-generated responses for factuality, accuracy, and hallucinations.

QUERY: {context.query}

AI RESPONSE TO EVALUATE:
{context.response}
"""
        
        if context.reference_docs:
            prompt += f"""
REFERENCE DOCUMENTS (Ground Truth):
{context.reference_docs[:2000]}  
"""
        
        if context.ground_truth:
            prompt += f"""
EXPECTED ANSWER:
{context.ground_truth}
"""
        
        prompt += """

EVALUATION CRITERIA:

1. FACTUAL ACCURACY (1-5):
   - Are all factual claims correct and verifiable?
   - Are there any fabricated facts, statistics, or quotes?
   
2. HALLUCINATION CHECK:
   - Does the response contain information NOT supported by the reference documents?
   - Are there invented details, names, or events?
   
3. RELEVANCE (1-5):
   - Does the response directly answer the query?
   - Is all information provided relevant?
   
4. COMPLETENESS (1-5):
   - Does it address all aspects of the query?
   - Is any critical information missing?

SCORING RUBRIC:
5 - Excellent: Completely accurate, no hallucinations, comprehensive
4 - Good: Accurate with minor omissions, no significant hallucinations
3 - Acceptable: Mostly accurate but has issues or gaps
2 - Poor: Significant inaccuracies or unsupported claims
1 - Unacceptable: Mostly inaccurate or heavily hallucinated

CONFIDENCE LEVELS:
- "high": Very confident in evaluation (clear-cut case)
- "medium": Moderately confident (some ambiguity)
- "low": Low confidence (needs human review)

Respond ONLY with valid JSON (no markdown formatting, no code blocks):
{
    "overall_score": <1-5>,
    "has_hallucination": <true/false>,
    "accuracy": <1-5>,
    "relevance": <1-5>,
    "completeness": <1-5>,
    "reasoning": "<detailed explanation of scores, specific examples of issues>",
    "problematic_claims": ["<claim 1>", "<claim 2>"],
    "confidence": "<high/medium/low>"
}"""
        
        return prompt
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,  # Deterministic for evaluation
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> FactualityAssessment:
        """Parse JSON response into structured format"""
        # Clean response (remove markdown if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'```json\s*|\s*```', '', cleaned)
        
        data = json.loads(cleaned)
        return FactualityAssessment(**data)


class ClaimVerificationEvaluator(BaseEvaluator):
    """Extracts and verifies individual claims"""
    
    def __init__(self, api_key: str, provider: str = "anthropic"):
        super().__init__(name="claim_verification", weight=2.5, critical=True)
        
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-sonnet-4-20250514"
        else:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-4o"
        
        self.provider = provider
    
    def should_run(self, context: EvaluationContext) -> bool:
        """Only run for responses with reference docs"""
        return context.reference_docs is not None and len(context.response) > 50
    
    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        """Extract and verify claims"""
        
        if not self.should_run(context):
            return self._skip_result()
        
        try:
            # Step 1: Extract claims
            claims = self._extract_claims(context.response)
            
            if not claims or len(claims) == 0:
                return EvaluationResult(
                    metric_name=self.name,
                    score=1.0,
                    passed=True,
                    confidence=0.6,
                    reasoning="No verifiable factual claims found",
                    status=EvaluationStatus.PASSED
                )
            
            # Step 2: Verify each claim
            verified = []
            unverified = []
            
            for claim in claims[:10]:  # Limit to 10 claims for cost
                is_supported = self._verify_claim(claim, context.reference_docs)
                if is_supported:
                    verified.append(claim)
                else:
                    unverified.append(claim)
            
            # Calculate score
            total = len(verified) + len(unverified)
            score = len(verified) / total if total > 0 else 1.0
            
            # Determine status
            if score >= 0.9:
                passed = True
                status = EvaluationStatus.PASSED
                reasoning = f"All claims verified ({len(verified)}/{total})"
            elif score >= 0.7:
                passed = True
                status = EvaluationStatus.PASSED
                reasoning = f"Most claims verified ({len(verified)}/{total})"
            elif score >= 0.5:
                passed = False
                status = EvaluationStatus.FLAGGED
                reasoning = f"Some unverified claims ({len(unverified)}/{total})"
            else:
                passed = False
                status = EvaluationStatus.FAILED
                reasoning = f"Many unverified claims ({len(unverified)}/{total})"
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                confidence=0.85,
                reasoning=reasoning,
                status=status,
                metadata={
                    'total_claims': total,
                    'verified_claims': len(verified),
                    'unverified_claims': unverified[:3],
                    'verification_rate': score
                }
            )
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.5,
                passed=True,
                confidence=0.0,
                reasoning=f"Verification error: {str(e)}",
                status=EvaluationStatus.ERROR
            )
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract atomic claims from response"""
        
        prompt = f"""Extract all factual claims from this text as a JSON list. 

Requirements for each claim:
- Atomic (one fact per claim)
- Self-contained (understandable without context)
- Verifiable (can be checked against source)
- Exclude opinions and subjective statements

Text: {response}

Respond with JSON only (no markdown):
{{"claims": ["claim 1", "claim 2", ...]}}"""
        
        if self.provider == "anthropic":
            result = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = result.content[0].text
        else:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            text = result.choices[0].message.content
        
        # Parse response
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'```json\s*|\s*```', '', cleaned)
        
        data = json.loads(cleaned)
        return data.get("claims", [])
    
    def _verify_claim(self, claim: str, reference: str) -> bool:
        """Verify if claim is supported"""
        
        prompt = f"""Is this claim fully supported by the reference text?

Claim: {claim}

Reference: {reference[:1500]}

Rules:
- Return true ONLY if the claim is explicitly or clearly implied by the reference
- Return false if the claim adds information not in the reference
- Return false if uncertain

Respond with JSON only (no markdown):
{{"supported": true/false, "reasoning": "brief explanation"}}"""
        
        if self.provider == "anthropic":
            result = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            text = result.content[0].text
        else:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            text = result.choices[0].message.content
        
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'```json\s*|\s*```', '', cleaned)
        
        data = json.loads(cleaned)
        return data.get("supported", False)
    
    def _skip_result(self) -> EvaluationResult:
        return EvaluationResult(
            metric_name=self.name,
            score=0.5,
            passed=True,
            confidence=0.0,
            reasoning="Evaluator skipped (no reference docs)",
            status=EvaluationStatus.PASSED
        )
    

class EvaluationPipeline:
    """Orchestrates multi-tier evaluation"""
    
    def __init__(
        self,
        evaluators: List[BaseEvaluator],
        early_stopping: bool = True,
        cost_threshold: float = 0.05  # Max $ per evaluation
    ):
        self.evaluators = evaluators
        self.early_stopping = early_stopping
        self.cost_threshold = cost_threshold
        
        # Organize evaluators by tier
        self.tier1 = []  # Fast, free
        self.tier2 = []  # Medium, free
        self.tier3 = []  # Slow, costly
        
        for evaluator in evaluators:
            if isinstance(evaluator, (LengthValidator, EntityHallucinationDetector, SemanticConsistencyChecker)):
                self.tier1.append(evaluator)
            elif isinstance(evaluator, NLIConsistencyChecker):
                self.tier2.append(evaluator)
            else:
                self.tier3.append(evaluator)
    
    def evaluate(
        self,
        context: EvaluationContext,
        run_tier3: Optional[bool] = None
    ) -> AggregatedResult:
        """Run evaluation pipeline with smart tier execution"""
        
        start_time = time.time()
        all_results = {}
        
        # Tier 1: Always run (fast, free)
        logger.info("Running Tier 1 evaluators...")
        for evaluator in self.tier1:
            if evaluator.should_run(context):
                result = evaluator.evaluate(context)
                all_results[evaluator.name] = result
                logger.info(f"  {evaluator.name}: {result.score:.2f} ({result.status.value})")
        
        # Check if we should stop early
        tier1_score = self._calculate_weighted_score(all_results)
        critical_failure = any(
            r.status == EvaluationStatus.FAILED and self._is_critical(r.metric_name)
            for r in all_results.values()
        )
        
        if self.early_stopping and (critical_failure or tier1_score < 0.3):
            logger.info("Early stopping: Critical failure detected in Tier 1")
            return self._aggregate_results(all_results, time.time() - start_time)
        
        # Tier 2: Run if Tier 1 looks reasonable
        logger.info("Running Tier 2 evaluators...")
        for evaluator in self.tier2:
            if evaluator.should_run(context):
                result = evaluator.evaluate(context)
                all_results[evaluator.name] = result
                logger.info(f"  {evaluator.name}: {result.score:.2f} ({result.status.value})")
        
        # Tier 3: Decision logic
        tier2_score = self._calculate_weighted_score(all_results)
        
        should_run_tier3 = run_tier3 if run_tier3 is not None else self._should_run_tier3(
            tier2_score, all_results
        )
        
        if should_run_tier3:
            logger.info("Running Tier 3 evaluators (LLM-as-Judge)...")
            for evaluator in self.tier3:
                if evaluator.should_run(context):
                    result = evaluator.evaluate(context)
                    all_results[evaluator.name] = result
                    logger.info(f"  {evaluator.name}: {result.score:.2f} ({result.status.value})")
        else:
            logger.info("Skipping Tier 3 (cost optimization)")
        
        return self._aggregate_results(all_results, time.time() - start_time)
    
    def _should_run_tier3(self, current_score: float, results: Dict) -> bool:
        """Decide if Tier 3 (expensive LLM calls) should run"""
        
        # Run if borderline (needs deeper analysis)
        if 0.4 <= current_score <= 0.75:
            return True
        
        # Run if any checks flagged for review
        if any(r.status == EvaluationStatus.FLAGGED for r in results.values()):
            return True
        
        # Random 10% sampling for monitoring
        import random
        if random.random() < 0.1:
            logger.info("Running Tier 3 for random sampling")
            return True
        
        return False
    
    def _calculate_weighted_score(self, results: Dict[str, EvaluationResult]) -> float:
        """Calculate weighted average score"""
        if not results:
            return 0.5
        
        total_weight = sum(self._get_weight(name) for name in results.keys())
        if total_weight == 0:
            return sum(r.score for r in results.values()) / len(results)
        
        weighted_sum = sum(
            r.score * self._get_weight(name)
            for name, r in results.items()
        )
        
        return weighted_sum / total_weight
    
    def _get_weight(self, evaluator_name: str) -> float:
        """Get weight for evaluator"""
        for evaluator in self.evaluators:
            if evaluator.name == evaluator_name:
                return evaluator.weight
        return 1.0
    
    def _is_critical(self, evaluator_name: str) -> bool:
        """Check if evaluator is critical"""
        for evaluator in self.evaluators:
            if evaluator.name == evaluator_name:
                return evaluator.critical
        return False
    
    def _aggregate_results(
        self,
        results: Dict[str, EvaluationResult],
        elapsed_time: float
    ) -> AggregatedResult:
        """Aggregate individual results into final verdict"""
        
        overall_score = self._calculate_weighted_score(results)
        
        # Determine overall status
        has_failures = any(r.status == EvaluationStatus.FAILED for r in results.values())
        has_flags = any(r.status == EvaluationStatus.FLAGGED for r in results.values())
        
        if has_failures:
            overall_status = EvaluationStatus.FAILED
        elif has_flags:
            overall_status = EvaluationStatus.FLAGGED
        else:
            overall_status = EvaluationStatus.PASSED
        
        # Flag for human review
        flagged = (
            overall_status == EvaluationStatus.FLAGGED or
            (overall_status == EvaluationStatus.FAILED and overall_score > 0.4) or
            overall_score == 0.5  # Uncertain
        )
        
        # Calculate overall confidence
        confidences = [r.confidence for r in results.values() if r.confidence > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Generate summary
        summary = self._generate_summary(results, overall_score, overall_status)
        
        logger.info(f"Evaluation complete in {elapsed_time:.2f}s")
        logger.info(f"Overall: {overall_score:.2f} ({overall_status.value})")
        
        return AggregatedResult(
            overall_score=overall_score,
            overall_status=overall_status,
            individual_results=results,
            flagged_for_review=flagged,
            confidence=overall_confidence,
            summary=summary
        )
    
    def _generate_summary(
        self,
        results: Dict[str, EvaluationResult],
        score: float,
        status: EvaluationStatus
    ) -> str:
        """Generate human-readable summary"""
        
        if status == EvaluationStatus.PASSED:
            summary = f"✓ Response passed all checks (score: {score:.2f})"
        elif status == EvaluationStatus.FAILED:
            failures = [
                f"{name}: {r.reasoning}"
                for name, r in results.items()
                if r.status == EvaluationStatus.FAILED
            ]
            summary = f"✗ Response failed: {'; '.join(failures)}"
        else:
            flags = [
                f"{name}: {r.reasoning}"
                for name, r in results.items()
                if r.status == EvaluationStatus.FLAGGED
            ]
            summary = f"⚠ Response flagged: {'; '.join(flags)}"
        
        return summary
    

class BatchEvaluator:
    """Batch evaluation with progress tracking"""
    
    def __init__(self, pipeline: EvaluationPipeline, max_workers: int = 5):
        self.pipeline = pipeline
        self.max_workers = max_workers
    
    def evaluate_batch(
        self,
        contexts: List[EvaluationContext],
        parallel: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[AggregatedResult]:
        """Evaluate multiple contexts"""
        
        results = []
        total = len(contexts)
        
        if parallel and total > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.pipeline.evaluate, ctx): idx
                    for idx, ctx in enumerate(contexts)
                }
                
                for idx, future in enumerate(as_completed(future_to_idx)):
                    result = future.result()
                    results.append((future_to_idx[future], result))
                    
                    if progress_callback:
                        progress_callback(idx + 1, total)
            
            # Sort by original order
            results.sort(key=lambda x: x[0])
            results = [r[1] for r in results]
        else:
            for idx, ctx in enumerate(contexts):
                result = self.pipeline.evaluate(ctx)
                results.append(result)
                
                if progress_callback:
                    progress_callback(idx + 1, total)
        
        return results
    
    def generate_report(
        self,
        contexts: List[EvaluationContext],
        results: List[AggregatedResult]
    ) -> Dict:
        """Generate comprehensive evaluation report"""
        
        report = {
            'summary': {
                'total_evaluated': len(results),
                'passed': sum(1 for r in results if r.overall_status == EvaluationStatus.PASSED),
                'failed': sum(1 for r in results if r.overall_status == EvaluationStatus.FAILED),
                'flagged': sum(1 for r in results if r.overall_status == EvaluationStatus.FLAGGED),
                'avg_score': sum(r.overall_score for r in results) / len(results),
                'avg_confidence': sum(r.confidence for r in results) / len(results)
            },
            'pass_rate': sum(1 for r in results if r.overall_status == EvaluationStatus.PASSED) / len(results),
            'metric_breakdown': {},
            'failure_analysis': defaultdict(int),
            'samples': {
                'best': [],
                'worst': [],
                'flagged': []
            }
        }
        
        # Per-metric statistics
        metric_scores = defaultdict(list)
        for result in results:
            for metric_name, metric_result in result.individual_results.items():
                metric_scores[metric_name].append(metric_result.score)
                
                if metric_result.status == EvaluationStatus.FAILED:
                    report['failure_analysis'][metric_name] += 1
        
        report['metric_breakdown'] = {
            metric: {
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'failure_count': report['failure_analysis'].get(metric, 0)
            }
            for metric, scores in metric_scores.items()
        }
        
        # Sample examples
        sorted_results = sorted(
            zip(contexts, results),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        # Best examples
        for ctx, res in sorted_results[:3]:
            if res.overall_status == EvaluationStatus.PASSED:
                report['samples']['best'].append({
                    'query': ctx.query[:100],
                    'response': ctx.response[:200],
                    'score': res.overall_score,
                    'summary': res.summary
                })
        
        # Worst examples
        for ctx, res in sorted_results[-3:]:
            if res.overall_status == EvaluationStatus.FAILED:
                report['samples']['worst'].append({
                    'query': ctx.query[:100],
                    'response': ctx.response[:200],
                    'score': res.overall_score,
                    'summary': res.summary
                })
        
        # Flagged examples
        for ctx, res in zip(contexts, results):
            if res.flagged_for_review:
                report['samples']['flagged'].append({
                    'query': ctx.query[:100],
                    'response': ctx.response[:200],
                    'score': res.overall_score,
                    'summary': res.summary
                })
                
                if len(report['samples']['flagged']) >= 5:
                    break
        
        return report
    
    def export_to_dataframe(
        self,
        contexts: List[EvaluationContext],
        results: List[AggregatedResult]
    ) -> pd.DataFrame:
        """Export results to pandas DataFrame"""
        
        rows = []
        for ctx, res in zip(contexts, results):
            row = {
                'query': ctx.query,
                'response': ctx.response,
                'overall_score': res.overall_score,
                'overall_status': res.overall_status.value,
                'confidence': res.confidence,
                'flagged': res.flagged_for_review,
                'summary': res.summary
            }
            
            # Add individual metric scores
            for metric_name, metric_result in res.individual_results.items():
                row[f'{metric_name}_score'] = metric_result.score
                row[f'{metric_name}_passed'] = metric_result.passed
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    


class EvaluatorValidator:
    """Validate evaluator reliability against human judgments"""
    
    def __init__(self, pipeline: EvaluationPipeline):
        self.pipeline = pipeline
    
    def validate_against_human_labels(
        self,
        contexts: List[EvaluationContext],
        human_scores: List[float],  # 0.0 to 1.0
        human_labels: List[bool]  # Pass/Fail
    ) -> Dict:
        """Compare evaluator against human judgments"""
        
        logger.info("Validating evaluator against human labels...")
        
        # Run evaluator
        llm_results = []
        for ctx in contexts:
            result = self.pipeline.evaluate(ctx)
            llm_results.append(result)
        
        llm_scores = [r.overall_score for r in llm_results]
        llm_labels = [r.overall_status == EvaluationStatus.PASSED for r in llm_results]
        
        # Calculate correlation (scores)
        pearson_corr, pearson_p = pearsonr(human_scores, llm_scores)
        spearman_corr, spearman_p = spearmanr(human_scores, llm_scores)
        
        # Calculate agreement (binary labels)
        accuracy = accuracy_score(human_labels, llm_labels)
        kappa = cohen_kappa_score(human_labels, llm_labels)
        f1 = f1_score(human_labels, llm_labels)
        
        # Classification report
        report = classification_report(
            human_labels,
            llm_labels,
            target_names=['Fail', 'Pass'],
            output_dict=True
        )
        
        validation_results = {
            'correlation': {
                'pearson': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman': spearman_corr,
                'spearman_p_value': spearman_p
            },
            'agreement': {
                'accuracy': accuracy,
                'cohens_kappa': kappa,
                'f1_score': f1,
                'precision': report['Pass']['precision'],
                'recall': report['Pass']['recall']
            },
            'interpretation': self._interpret_results(pearson_corr, kappa, accuracy)
        }
        
        # Print report
        print("\n" + "="*60)
        print("EVALUATOR VALIDATION REPORT")
        print("="*60)
        print(f"\nCorrelation Analysis:")
        print(f"  Pearson correlation:  {pearson_corr:.3f} (p={pearson_p:.4f})")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
        
        print(f"\nAgreement Analysis:")
        print(f"  Accuracy:      {accuracy:.1%}")
        print(f"  Cohen's Kappa: {kappa:.3f}")
        print(f"  F1 Score:      {f1:.3f}")
        print(f"  Precision:     {report['Pass']['precision']:.1%}")
        print(f"  Recall:        {report['Pass']['recall']:.1%}")
        
        print(f"\nInterpretation:")
        print(f"  {validation_results['interpretation']}")
        print("="*60 + "\n")
        
        return validation_results
    
    def _interpret_results(self, correlation: float, kappa: float, accuracy: float) -> str:
        """Interpret validation metrics"""
        
        if kappa >= 0.8 and correlation >= 0.75:
            return "✓ EXCELLENT: Evaluator highly reliable, safe for production use"
        elif kappa >= 0.6 and correlation >= 0.6:
            return "⚠ MODERATE: Evaluator acceptable, recommend human sampling (10-20%)"
        elif kappa >= 0.4:
            return "⚠ FAIR: Evaluator needs improvement, use with caution (30%+ sampling)"
        else:
            return "✗ POOR: Evaluator unreliable, requires major refinement"
    
    def cross_validate_evaluators(
        self,
        contexts: List[EvaluationContext],
        evaluator_configs: List[Dict]
    ) -> Dict:
        """Compare multiple evaluator configurations"""
        
        logger.info("Cross-validating multiple evaluator configurations...")
        
        results_matrix = []
        
        for config in evaluator_configs:
            # Create pipeline with this config
            pipeline = self._create_pipeline_from_config(config)
            
            # Evaluate all contexts
            scores = []
            for ctx in contexts:
                result = pipeline.evaluate(ctx)
                scores.append(result.overall_score)
            
            results_matrix.append(scores)
        
        # Calculate inter-evaluator agreement
        correlations = []
        for i in range(len(results_matrix)):
            for j in range(i + 1, len(results_matrix)):
                corr, _ = pearsonr(results_matrix[i], results_matrix[j])
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        print(f"\nCross-Validation Results:")
        print(f"  Average inter-evaluator correlation: {avg_correlation:.3f}")
        
        if avg_correlation >= 0.8:
            print("  ✓ High agreement between evaluators")
        elif avg_correlation >= 0.6:
            print("  ⚠ Moderate agreement - some inconsistency")
        else:
            print("  ✗ Low agreement - evaluators disagree significantly")
        
        return {
            'avg_correlation': avg_correlation,
            'all_correlations': correlations,
            'results_matrix': results_matrix
        }
    
    def _create_pipeline_from_config(self, config: Dict) -> EvaluationPipeline:
        """Helper to create pipeline from config"""
        # Implementation depends on your config structure
        pass


class ProductionMonitor:
    """Monitor evaluator performance in production"""
    
    def __init__(self, pipeline: EvaluationPipeline, sampling_rate: float = 0.05):
        self.pipeline = pipeline
        self.sampling_rate = sampling_rate
        self.human_feedback = []
    
    def evaluate_with_sampling(
        self,
        context: EvaluationContext
    ) -> Tuple[AggregatedResult, bool]:
        """Evaluate and randomly flag for human review"""
        
        result = self.pipeline.evaluate(context)
        
        # Random sampling for validation
        import random
        needs_human_review = (
            result.flagged_for_review or
            random.random() < self.sampling_rate
        )
        
        return result, needs_human_review
    
    def record_human_feedback(
        self,
        context: EvaluationContext,
        llm_result: AggregatedResult,
        human_score: float,
        human_passed: bool
    ):
        """Record human judgment for monitoring"""
        
        self.human_feedback.append({
            'llm_score': llm_result.overall_score,
            'llm_passed': llm_result.overall_status == EvaluationStatus.PASSED,
            'human_score': human_score,
            'human_passed': human_passed,
            'timestamp': datetime.now()
        })
    
    def calculate_drift(self, window_size: int = 100) -> Dict:
        """Detect if evaluator is drifting from human judgment"""
        
        if len(self.human_feedback) < window_size:
            return {'status': 'insufficient_data', 'samples': len(self.human_feedback)}
        
        recent = self.human_feedback[-window_size:]
        
        llm_scores = [f['llm_score'] for f in recent]
        human_scores = [f['human_score'] for f in recent]
        
        correlation, _ = pearsonr(llm_scores, human_scores)
        
        llm_labels = [f['llm_passed'] for f in recent]
        human_labels = [f['human_passed'] for f in recent]
        
        agreement = sum(1 for l, h in zip(llm_labels, human_labels) if l == h) / len(recent)
        
        # Check for drift
        drift_detected = correlation < 0.6 or agreement < 0.7
        
        return {
            'status': 'drift_detected' if drift_detected else 'healthy',
            'correlation': correlation,
            'agreement': agreement,
            'samples': len(recent),
            'recommendation': 'Re-validate evaluator' if drift_detected else 'Continue monitoring'
        }