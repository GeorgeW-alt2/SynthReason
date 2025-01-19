from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
import random
import numpy as np
from collections import defaultdict

@dataclass
class LogicalProposition:
    statement: str
    confidence: float
    domain: str
    premises: Set[str]
    conclusion: str
    inference_type: str
    probability: float = 1.0
    causal_strength: float = 1.0
    temporal_order: Optional[int] = None
    supporting_evidence: List[str] = None

class AdvancedInferenceEngine:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.inference_rules = {
            'modus_ponens': {
                'rule': lambda p, q: f"If {p} implies {q}, and {p} is true, then {q} is true",
                'probability': 0.95
            },
            'modus_tollens': {
                'rule': lambda p, q: f"If {p} implies {q}, and {q} is false, then {p} is false",
                'probability': 0.90
            },
            'abductive': {
                'rule': lambda p, q: f"Given {q}, {p} is a possible explanation because {p} typically leads to {q}",
                'probability': 0.75
            }
        }

        self.cognitive_domains = {
            'perception': {
                'processes': ['sensation', 'attention', 'recognition', 'integration'],
                'properties': ['parallel', 'hierarchical', 'selective', 'adaptive'],
                'mechanisms': ['bottom-up', 'top-down', 'predictive', 'feature-binding']
            },
            'memory': {
                'processes': ['encoding', 'storage', 'retrieval', 'consolidation'],
                'properties': ['distributed', 'associative', 'reconstructive', 'plastic'],
                'mechanisms': ['rehearsal', 'elaboration', 'chunking', 'spreading-activation']
            },
            'learning': {
                'processes': ['acquisition', 'reinforcement', 'generalization', 'adaptation'],
                'properties': ['incremental', 'hierarchical', 'context-dependent', 'error-driven'],
                'mechanisms': ['hebbian', 'error-correction', 'competitive', 'self-organizing']
            }
        }
        
        self._initialize_knowledge_structures()

    def _initialize_knowledge_structures(self):
        """Initialize knowledge graph with domain concepts."""
        for domain, content in self.cognitive_domains.items():
            for category, items in content.items():
                for item in items:
                    self.knowledge_graph.add_node(
                        item,
                        type='concept',
                        domain=domain,
                        category=category
                    )

    def _determine_inference_domain(self, premises: List[str]) -> str:
        """Determine the cognitive domain of an inference from its premises."""
        domain_scores = defaultdict(float)
        
        for premise in premises:
            words = premise.lower().split()
            for domain, content in self.cognitive_domains.items():
                score = 0.0
                for category in content.values():
                    for item in category:
                        if item.lower() in words:
                            score += 1.0
                if score > 0:
                    domain_scores[domain] += score
                    
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def _calculate_inference_confidence(self, premises: List[str], rule_type: str) -> float:
        """Calculate confidence score for an inference."""
        rule_confidences = {
            'modus_ponens': 0.95,
            'modus_tollens': 0.90,
            'abductive': 0.75
        }
        
        base_confidence = rule_confidences.get(rule_type, 0.60)
        premise_adjustment = max(0, 1.0 - (len(premises) - 1) * 0.1)
        
        return round(base_confidence * premise_adjustment, 3)

    def _extract_concepts(self, statement: str) -> Set[str]:
        """Extract cognitive concepts from a statement."""
        concepts = set()
        words = statement.lower().split()
        
        for domain_data in self.cognitive_domains.values():
            for category_items in domain_data.values():
                for item in category_items:
                    if item.lower() in words:
                        concepts.add(item)
        return concepts

    def generate_inference(self, premises: List[str], inference_type: str) -> LogicalProposition:
        """Generate a logical inference from given premises."""
        if not premises:
            return None
            
        rule_info = self.inference_rules.get(inference_type)
        if not rule_info:
            return None
            
        rule = rule_info['rule']
        base_probability = rule_info['probability']
        
        if len(premises) >= 2:
            conclusion = rule(premises[0], premises[1])
        else:
            # Generate a reasonable consequent for single premise
            domain = self._determine_inference_domain(premises)
            if domain in self.cognitive_domains:
                process = random.choice(self.cognitive_domains[domain]['processes'])
                conclusion = rule(premises[0], f"enhanced {process}")
            else:
                conclusion = rule(premises[0], "leads to cognitive enhancement")
        
        confidence = self._calculate_inference_confidence(premises, inference_type)
        domain = self._determine_inference_domain(premises)
        
        return LogicalProposition(
            statement=" AND ".join(premises),
            confidence=confidence,
            domain=domain,
            premises=set(premises),
            conclusion=conclusion,
            inference_type=inference_type,
            probability=base_probability
        )

    def chain_inferences(self, initial_statement: str, depth: int = 3) -> List[LogicalProposition]:
        """Generate a chain of inferences starting from an initial statement."""
        inference_chain = []
        current_premises = [initial_statement]
        
        for _ in range(depth):
            inference_type = random.choice(list(self.inference_rules.keys()))
            inference = self.generate_inference(current_premises, inference_type)
            
            if inference:
                inference_chain.append(inference)
                current_premises = [inference.conclusion]
        
        return inference_chain

    def find_related_inferences(self, concept: str) -> List[LogicalProposition]:
        """Find inferences related to a given concept."""
        related_inferences = []
        concepts = self._extract_concepts(concept)
        
        for c in concepts:
            if c in self.knowledge_graph:
                neighbors = list(self.knowledge_graph.neighbors(c))
                if neighbors:
                    inference = self.generate_inference(
                        [c],
                        random.choice(list(self.inference_rules.keys()))
                    )
                    if inference:
                        related_inferences.append(inference)
                        
        return related_inferences

def main():
    engine = AdvancedInferenceEngine()
    
    print("Advanced Cognitive Inference Engine\n")
    
    initial_statements = [
        "Attention modulates perception",
        "Learning requires memory consolidation",
        "Neural plasticity enables adaptation"
    ]
    
    print("Generating inference chains...\n")
    for statement in initial_statements:
        print(f"\nInitial Statement: {statement}")
        print("-" * 80)
        
        inference_chain = engine.chain_inferences(statement, depth=3)
        
        for i, inference in enumerate(inference_chain, 1):
            print(f"\nInference {i}:")
            print(f"Type: {inference.inference_type}")
            print(f"Premises: {', '.join(inference.premises)}")
            print(f"Conclusion: {inference.conclusion}")
            print(f"Confidence: {inference.confidence:.3f}")
            print(f"Probability: {inference.probability:.3f}")
            print(f"Domain: {inference.domain}")
            
        print("-" * 80)

if __name__ == "__main__":
    main()