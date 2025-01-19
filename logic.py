import random
import pickle
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import numpy as np

class LogicalStatementGenerator:
    def __init__(self):
        # Logical operators
        self.operators = {
            'unary': ['NOT', 'NECESSARILY', 'POSSIBLY'],
            'binary': ['AND', 'OR', 'IMPLIES', 'IFF', 'XOR'],
            'quantifiers': ['FORALL', 'EXISTS', 'UNIQUE']
        }
        
        # Cognitive domain vocabularies
        self.vocabulary = {
            'concepts': Counter({
                'knowledge': 1, 'belief': 1, 'truth': 1, 'perception': 1,
                'consciousness': 1, 'thought': 1, 'memory': 1, 'learning': 1,
                'reasoning': 1, 'intelligence': 1, 'understanding': 1,
                'awareness': 1, 'attention': 1, 'intention': 1, 'decision': 1
            }),
            
            'predicates': Counter({
                'implies': 1, 'causes': 1, 'leads_to': 1, 'requires': 1,
                'depends_on': 1, 'influences': 1, 'correlates_with': 1,
                'precedes': 1, 'follows': 1, 'contains': 1, 'consists_of': 1,
                'exhibits': 1, 'demonstrates': 1, 'manifests': 1, 'emerges_from': 1
            }),
            
            'properties': Counter({
                'recursive': 1, 'hierarchical': 1, 'parallel': 1, 'sequential': 1,
                'distributed': 1, 'centralized': 1, 'adaptive': 1, 'stable': 1,
                'dynamic': 1, 'static': 1, 'explicit': 1, 'implicit': 1,
                'conscious': 1, 'unconscious': 1, 'automatic': 1
            }),
            
            'modifiers': Counter({
                'necessarily': 1, 'possibly': 1, 'probably': 1, 'definitely': 1,
                'always': 1, 'never': 1, 'sometimes': 1, 'often': 1,
                'partially': 1, 'fully': 1, 'gradually': 1, 'immediately': 1,
                'directly': 1, 'indirectly': 1, 'systematically': 1
            })
        }
        
        # Logical templates for different types of statements
        self.templates = {
            'simple': [
                ('CONCEPT', 'PREDICATE', 'CONCEPT'),
                ('PROPERTY', 'CONCEPT', 'PREDICATE', 'CONCEPT'),
                ('MODIFIER', 'CONCEPT', 'PREDICATE', 'CONCEPT')
            ],
            'compound': [
                ('(', 'CONCEPT', 'PREDICATE', 'CONCEPT', ')', 'BINARY_OP', 
                 '(', 'CONCEPT', 'PREDICATE', 'CONCEPT', ')'),
                ('UNARY_OP', '(', 'CONCEPT', 'PREDICATE', 'CONCEPT', ')'),
                ('QUANTIFIER', 'CONCEPT', '(', 'PREDICATE', 'CONCEPT', ')')
            ]
        }
        
        # Axiom schemas
        self.axiom_schemas = [
            "If {concept1} {predicate} {concept2}, then {concept2} {inverse_predicate} {concept1}",
            "For all {concept}, there exists a {property} such that {concept} {predicate} it",
            "Either {concept1} {predicate} {concept2} or {concept2} {predicate} {concept1}",
            "If {concept1} is {property}, then it {predicate} {concept2}"
        ]

    def weighted_choice(self, counter: Counter) -> str:
        """Select an item based on its frequency weight."""
        if not counter:
            return ""
        items = list(counter.keys())
        weights = list(counter.values())
        total = sum(weights)
        probabilities = [w/total for w in weights]
        return random.choices(items, weights=probabilities, k=1)[0]

    def generate_atomic_proposition(self) -> str:
        """Generate an atomic logical proposition."""
        concept1 = self.weighted_choice(self.vocabulary['concepts'])
        predicate = self.weighted_choice(self.vocabulary['predicates'])
        concept2 = self.weighted_choice(self.vocabulary['concepts'])
        property = self.weighted_choice(self.vocabulary['properties'])
        
        if random.random() < 0.3:
            return f"{property} {concept1} {predicate} {concept2}"
        return f"{concept1} {predicate} {concept2}"

    def generate_compound_statement(self) -> str:
        """Generate a compound logical statement."""
        op = random.choice(self.operators['binary'])
        prop1 = self.generate_atomic_proposition()
        prop2 = self.generate_atomic_proposition()
        
        if random.random() < 0.3:
            modifier = self.weighted_choice(self.vocabulary['modifiers'])
            return f"{modifier} ({prop1} {op} {prop2})"
        return f"({prop1} {op} {prop2})"

    def generate_quantified_statement(self) -> str:
        """Generate a quantified logical statement."""
        quantifier = random.choice(self.operators['quantifiers'])
        concept = self.weighted_choice(self.vocabulary['concepts'])
        property = self.weighted_choice(self.vocabulary['properties'])
        predicate = self.weighted_choice(self.vocabulary['predicates'])
        
        return f"{quantifier} {concept} ({property} {predicate})"

    def generate_axiom(self) -> str:
        """Generate an axiom using the schemas."""
        schema = random.choice(self.axiom_schemas)
        
        # Fill in the placeholders
        filled_schema = schema.format(
            concept1=self.weighted_choice(self.vocabulary['concepts']),
            concept2=self.weighted_choice(self.vocabulary['concepts']),
            predicate=self.weighted_choice(self.vocabulary['predicates']),
            property=self.weighted_choice(self.vocabulary['properties']),
            inverse_predicate=self.weighted_choice(self.vocabulary['predicates'])
        )
        
        return filled_schema

    def generate_logical_theory(self, num_statements: int = 3) -> List[str]:
        """Generate a coherent set of logical statements forming a theory."""
        theory = []
        
        # Start with an axiom
        theory.append(f"Axiom 1: {self.generate_axiom()}")
        
        # Add derived statements
        for i in range(1, num_statements):
            if random.random() < 0.4:
                stmt = self.generate_compound_statement()
            elif random.random() < 0.7:
                stmt = self.generate_quantified_statement()
            else:
                stmt = self.generate_atomic_proposition()
                
            theory.append(f"Statement {i+1}: {stmt}")
            
        return theory

    def learn_from_statement(self, statement: str) -> None:
        """Learn from an input logical statement."""
        words = statement.lower().split()
        
        # Update vocabulary frequencies based on word patterns
        for i, word in enumerate(words):
            if word in self.operators['binary'] or word in self.operators['unary']:
                continue
                
            if i > 0 and words[i-1] in ['is', 'are', 'becomes']:
                self.vocabulary['properties'][word] += 1
            elif i > 0 and words[i-1] in ['that', 'which', 'who']:
                self.vocabulary['predicates'][word] += 1
            else:
                self.vocabulary['concepts'][word] += 1

def main():
    generator = LogicalStatementGenerator()
    
    print("Logical Statement Generator")
    print("Generating a cognitive theory...\n")
    
    theory = generator.generate_logical_theory(5)
    for statement in theory:
        print(statement)
        print()
    
    while True:
        choice = input("\nGenerate another theory? (y/n): ").lower()
        if choice != 'y':
            break
            
        num_statements = int(input("How many statements? (1-10): "))
        num_statements = max(1, min(10, num_statements))
        
        print("\nGenerating new theory...\n")
        theory = generator.generate_logical_theory(num_statements)
        for statement in theory:
            print(statement)
            print()

if __name__ == "__main__":
    main()