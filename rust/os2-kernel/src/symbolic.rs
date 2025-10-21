use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SymbolicError {
    #[error("facts must be grounded (no variables)")]
    FactMustBeGrounded,
    #[error("rule predicates must have matching arity")]
    RuleArityMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolicTerm {
    Atom(String),
    Variable(String),
}

impl SymbolicTerm {
    pub fn atom(value: impl Into<String>) -> Self {
        Self::Atom(value.into())
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }

    fn is_variable(&self) -> bool {
        matches!(self, SymbolicTerm::Variable(_))
    }

    #[allow(dead_code)]
    fn resolve(&self, bindings: &HashMap<String, String>) -> Option<String> {
        match self {
            SymbolicTerm::Atom(value) => Some(value.clone()),
            SymbolicTerm::Variable(name) => bindings.get(name).cloned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolicAtom {
    pub predicate: String,
    pub terms: Vec<SymbolicTerm>,
}

impl SymbolicAtom {
    pub fn new(predicate: impl Into<String>, terms: Vec<SymbolicTerm>) -> Self {
        Self {
            predicate: predicate.into(),
            terms,
        }
    }

    pub fn is_grounded(&self) -> bool {
        self.terms.iter().all(|term| !term.is_variable())
    }

    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    fn bind(&self, bindings: &HashMap<String, String>) -> SymbolicAtom {
        let terms = self
            .terms
            .iter()
            .map(|term| match term {
                SymbolicTerm::Atom(value) => SymbolicTerm::Atom(value.clone()),
                SymbolicTerm::Variable(name) => bindings
                    .get(name)
                    .map(|value| SymbolicTerm::Atom(value.clone()))
                    .unwrap_or_else(|| SymbolicTerm::Variable(name.clone())),
            })
            .collect();
        SymbolicAtom {
            predicate: self.predicate.clone(),
            terms,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicRule {
    pub antecedents: Vec<SymbolicAtom>,
    pub consequent: SymbolicAtom,
}

impl SymbolicRule {
    pub fn new(antecedents: Vec<SymbolicAtom>, consequent: SymbolicAtom) -> Self {
        Self {
            antecedents,
            consequent,
        }
    }

    fn validate(&self) -> Result<(), SymbolicError> {
        for atom in &self.antecedents {
            if atom.arity() != self.consequent.arity() && !self.consequent.terms.is_empty() {
                return Err(SymbolicError::RuleArityMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymbolicDerivation {
    pub consequent: SymbolicAtom,
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, Default, Clone)]
pub struct SymbolicLogicEngine {
    facts: HashSet<SymbolicAtom>,
    rules: Vec<SymbolicRule>,
    derivations: Vec<SymbolicDerivation>,
}

impl SymbolicLogicEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assert_fact(&mut self, fact: SymbolicAtom) -> Result<bool, SymbolicError> {
        if !fact.is_grounded() {
            return Err(SymbolicError::FactMustBeGrounded);
        }
        let inserted = self.facts.insert(fact);
        if inserted {
            self.run_inference();
        }
        Ok(inserted)
    }

    pub fn assert_rule(&mut self, rule: SymbolicRule) -> Result<(), SymbolicError> {
        rule.validate()?;
        self.rules.push(rule);
        self.run_inference();
        Ok(())
    }

    pub fn query(&self, pattern: &SymbolicAtom) -> Vec<HashMap<String, String>> {
        let mut results = Vec::new();
        for fact in &self.facts {
            if let Some(bindings) = unify(pattern, fact) {
                results.push(bindings);
            }
        }
        results
    }

    pub fn derivations(&self) -> &[SymbolicDerivation] {
        &self.derivations
    }

    fn run_inference(&mut self) {
        let mut queue: VecDeque<&SymbolicRule> = self.rules.iter().collect();
        let mut new_derivations = Vec::new();

        while let Some(rule) = queue.pop_front() {
            let bindings_list = self.match_antecedents(&rule.antecedents);
            for bindings in bindings_list {
                let grounded = rule.consequent.bind(&bindings);
                if grounded.is_grounded() && self.facts.insert(grounded.clone()) {
                    new_derivations.push(SymbolicDerivation {
                        consequent: grounded,
                        bindings: bindings.clone(),
                    });
                    queue.extend(self.rules.iter());
                }
            }
        }

        if !new_derivations.is_empty() {
            self.derivations.extend(new_derivations);
        }
    }

    fn match_antecedents(&self, antecedents: &[SymbolicAtom]) -> Vec<HashMap<String, String>> {
        let mut bindings_list = vec![HashMap::new()];
        for antecedent in antecedents {
            let mut next_bindings = Vec::new();
            for bindings in bindings_list {
                let bound_atom = antecedent.bind(&bindings);
                for fact in &self.facts {
                    if let Some(mut candidate) = unify(&bound_atom, fact) {
                        candidate.extend(bindings.clone());
                        next_bindings.push(candidate);
                    }
                }
            }
            bindings_list = next_bindings;
            if bindings_list.is_empty() {
                break;
            }
        }
        bindings_list
    }
}

fn unify(pattern: &SymbolicAtom, fact: &SymbolicAtom) -> Option<HashMap<String, String>> {
    if pattern.predicate != fact.predicate || pattern.arity() != fact.arity() {
        return None;
    }

    let mut bindings = HashMap::new();
    for (pattern_term, fact_term) in pattern.terms.iter().zip(fact.terms.iter()) {
        match (pattern_term, fact_term) {
            (SymbolicTerm::Atom(a), SymbolicTerm::Atom(b)) if a == b => {}
            (SymbolicTerm::Atom(_), SymbolicTerm::Atom(_)) => return None,
            (SymbolicTerm::Atom(_), SymbolicTerm::Variable(_)) => return None,
            (SymbolicTerm::Variable(name), SymbolicTerm::Atom(value)) => match bindings.get(name) {
                Some(bound) if bound == value => {}
                Some(_) => return None,
                None => {
                    bindings.insert(name.clone(), value.clone());
                }
            },
            (SymbolicTerm::Variable(name), SymbolicTerm::Variable(other)) => {
                if let Some(bound) = bindings.get(name) {
                    bindings.insert(other.clone(), bound.clone());
                }
            }
        }
    }
    Some(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(predicate: &str, terms: &[&str]) -> SymbolicAtom {
        SymbolicAtom::new(
            predicate,
            terms
                .iter()
                .map(|term| SymbolicTerm::Atom((*term).to_string()))
                .collect(),
        )
    }

    #[test]
    fn inserting_and_querying_fact() {
        let mut engine = SymbolicLogicEngine::new();
        engine
            .assert_fact(fact("parent", &["alice", "bob"]))
            .expect("fact");

        let pattern = SymbolicAtom::new(
            "parent",
            vec![
                SymbolicTerm::Variable("x".into()),
                SymbolicTerm::Atom("bob".into()),
            ],
        );
        let results = engine.query(&pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["x"], "alice");
    }

    #[test]
    fn forward_chaining_infers_new_fact() {
        let mut engine = SymbolicLogicEngine::new();
        engine
            .assert_fact(fact("parent", &["alice", "bob"]))
            .expect("fact");
        engine
            .assert_fact(fact("parent", &["bob", "carol"]))
            .expect("fact");

        let grandparent_rule = SymbolicRule::new(
            vec![
                SymbolicAtom::new(
                    "parent",
                    vec![
                        SymbolicTerm::Variable("x".into()),
                        SymbolicTerm::Variable("y".into()),
                    ],
                ),
                SymbolicAtom::new(
                    "parent",
                    vec![
                        SymbolicTerm::Variable("y".into()),
                        SymbolicTerm::Variable("z".into()),
                    ],
                ),
            ],
            SymbolicAtom::new(
                "grandparent",
                vec![
                    SymbolicTerm::Variable("x".into()),
                    SymbolicTerm::Variable("z".into()),
                ],
            ),
        );

        engine.assert_rule(grandparent_rule).expect("rule");

        let query = SymbolicAtom::new(
            "grandparent",
            vec![
                SymbolicTerm::Atom("alice".into()),
                SymbolicTerm::Variable("desc".into()),
            ],
        );
        let results = engine.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["desc"], "carol");
        assert!(
            engine
                .derivations()
                .iter()
                .any(|derivation| derivation.consequent.predicate == "grandparent")
        );
    }
}
