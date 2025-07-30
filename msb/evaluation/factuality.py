"""Factuality evaluation metrics"""

import re
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FactualityScore:
    """Factuality evaluation score"""
    accuracy: float
    hallucination_rate: float
    source_reliability: float
    overall_factuality: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.overall_factuality,
            "accuracy": self.accuracy,
            "hallucination_rate": self.hallucination_rate,
            "source_reliability": self.source_reliability,
            "details": self.details
        }


class FactualityEvaluator:
    """Evaluator for factuality metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fact_check_model = config.get("fact_check_model", "gpt-4")
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.check_sources = config.get("check_sources", True)
        
        # Known facts database (simplified)
        self._load_known_facts()
    
    def _load_known_facts(self):
        """Load database of known facts for verification"""
        self.known_facts = {
            "capitals": {
                "france": "paris",
                "japan": "tokyo",
                "usa": "washington dc",
                "uk": "london",
                "germany": "berlin",
                "china": "beijing",
                "india": "new delhi",
                "brazil": "brasilia",
                "canada": "ottawa",
                "australia": "canberra"
            },
            "math": {
                "pi": 3.14159,
                "e": 2.71828,
                "golden_ratio": 1.618
            },
            "science": {
                "water_formula": "H2O",
                "speed_of_light": "299792458 m/s",
                "earth_age": "4.5 billion years",
                "solar_system_planets": 8
            }
        }
        
        # Common hallucination patterns
        self.hallucination_patterns = [
            re.compile(r'As of my knowledge cutoff in \d{4}', re.IGNORECASE),
            re.compile(r'I don\'t have access to real-time', re.IGNORECASE),
            re.compile(r'Based on my training data', re.IGNORECASE)
        ]
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> FactualityScore:
        """
        Evaluate factuality of a model response
        
        Args:
            prompt: Input prompt
            response: Model response
            context: Additional context
            language: Language code
            
        Returns:
            FactualityScore object
        """
        # Extract factual claims from response
        claims = self._extract_factual_claims(response)
        
        # Verify claims against known facts
        accuracy_score = self._verify_claims(claims)
        
        # Check for hallucination indicators
        hallucination_score = self._check_hallucinations(response)
        
        # Evaluate source citations if applicable
        source_score = self._evaluate_sources(response) if self.check_sources else 1.0
        
        # Calculate overall factuality
        overall_factuality = (
            0.5 * accuracy_score +
            0.3 * (1 - hallucination_score) +
            0.2 * source_score
        )
        
        return FactualityScore(
            accuracy=accuracy_score,
            hallucination_rate=hallucination_score,
            source_reliability=source_score,
            overall_factuality=overall_factuality,
            details={
                "claims_found": len(claims),
                "verified_claims": sum(1 for c in claims if self._verify_single_claim(c)),
                "language": language,
                "prompt_type": self._classify_prompt_type(prompt)
            }
        )
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains factual indicators
            factual_indicators = [
                r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
                r'\bequals\b', r'\bcontains\b', r'\bhas\b',
                r'\d+', r'%', r'degrees', r'meters', r'years'
            ]
            
            if any(re.search(indicator, sentence, re.IGNORECASE) for indicator in factual_indicators):
                claims.append(sentence)
        
        return claims
    
    def _verify_claims(self, claims: List[str]) -> float:
        """Verify factual claims against known facts"""
        if not claims:
            return 1.0  # No claims to verify
        
        verified = 0
        total = len(claims)
        
        for claim in claims:
            if self._verify_single_claim(claim):
                verified += 1
        
        return verified / total
    
    def _verify_single_claim(self, claim: str) -> bool:
        """Verify a single factual claim"""
        claim_lower = claim.lower()
        
        # Check against known facts
        for category, facts in self.known_facts.items():
            for key, value in facts.items():
                if key in claim_lower and str(value).lower() in claim_lower:
                    return True
                # Check for contradictions
                if key in claim_lower and str(value).lower() not in claim_lower:
                    # Additional check for numerical values
                    if isinstance(value, (int, float)):
                        # Extract numbers from claim
                        numbers = re.findall(r'\d+\.?\d*', claim)
                        if numbers:
                            for num in numbers:
                                if abs(float(num) - value) / value < 0.1:  # 10% tolerance
                                    return True
                    return False
        
        # If no known fact to verify against, assume neutral
        return True
    
    def _check_hallucinations(self, text: str) -> float:
        """Check for hallucination indicators"""
        hallucination_count = 0
        
        # Check for explicit hallucination patterns
        for pattern in self.hallucination_patterns:
            if pattern.search(text):
                hallucination_count += 1
        
        # Check for uncertain language
        uncertainty_phrases = [
            "might be", "could be", "possibly", "perhaps",
            "I think", "I believe", "it seems", "apparently"
        ]
        
        text_lower = text.lower()
        for phrase in uncertainty_phrases:
            if phrase in text_lower:
                hallucination_count += 0.5
        
        # Check for made-up citations
        fake_citation_pattern = re.compile(r'\[\d+\]|\(\w+,\s*\d{4}\)')
        fake_citations = fake_citation_pattern.findall(text)
        if fake_citations and not self._verify_citations(fake_citations):
            hallucination_count += len(fake_citations) * 0.2
        
        # Normalize score
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        hallucination_rate = min(1.0, hallucination_count / (word_count * 0.05))
        return hallucination_rate
    
    def _evaluate_sources(self, text: str) -> float:
        """Evaluate source citation quality"""
        # Look for citation patterns
        citations = re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)|https?://\S+', text)
        
        if not citations:
            # No citations - neutral score
            return 0.5
        
        # Verify citations (simplified)
        valid_citations = 0
        for citation in citations:
            if self._verify_citations([citation]):
                valid_citations += 1
        
        return valid_citations / len(citations)
    
    def _verify_citations(self, citations: List[str]) -> bool:
        """Verify if citations are valid (simplified check)"""
        for citation in citations:
            # Check if it's a URL
            if citation.startswith(('http://', 'https://')):
                # In real implementation, would check if URL exists
                return True
            # Check if it's a standard academic citation
            if re.match(r'\(\w+,\s*\d{4}\)', citation):
                return True
        return False
    
    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify the type of prompt for better evaluation"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["what", "who", "where", "when", "which"]):
            return "factual_question"
        elif any(word in prompt_lower for word in ["explain", "describe", "how"]):
            return "explanation"
        elif any(word in prompt_lower for word in ["opinion", "think", "feel"]):
            return "opinion"
        else:
            return "general"