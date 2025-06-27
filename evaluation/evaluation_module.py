# evaluation/evaluation_module.py
import json
import os
import sys
from datetime import datetime, timezone

# Ensure project root is in sys.path for robust module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from database.idea_db import IdeaDatabase
except ImportError as e:
    print(f"Error importing IdeaDatabase: {e}")
    # This is a critical dependency. If it cannot be imported, the module cannot function.
    # Consider how the application should handle this. For now, it will raise an error on instantiation.
    # A proper setup (e.g., pyproject.toml) would make these imports more reliable.
    raise


class EvaluationModule:
    def __init__(self, idea_database: IdeaDatabase, evaluator_name: str = "BasicEvaluatorV1"):
        if not isinstance(idea_database, IdeaDatabase):
            # This check is important because the rest of the class relies on the IdeaDatabase interface.
            raise ValueError("EvaluationModule requires a valid instance of IdeaDatabase.")
        self.idea_db = idea_database
        self.evaluator_name = evaluator_name

        self.criteria = {
            "novelty": {
                "description": "How new or original is the idea?",
                "scoring_logic": lambda idea_content: self._simple_keyword_score(idea_content, ["new", "original", "innovative", "unique", "novel", "fresh", "groundbreaking"], 0.1, 1.0)
            },
            "feasibility": {
                "description": "How practical or achievable is the idea?",
                "scoring_logic": lambda idea_content: self._simple_keyword_score(idea_content, ["feasible", "practical", "achievable", "realistic", "viable", "doable"], 0.1, 1.0, negative_keywords=["impossible", "unrealistic", "cannot", "impractical"])
            },
            "impact": {
                "description": "What is the potential positive effect or significance?",
                "scoring_logic": lambda idea_content: self._simple_keyword_score(idea_content, ["impactful", "significant", "benefit", "transform", "improve", "solution", "valuable"], 0.1, 1.0)
            },
            "clarity": {
                "description": "How clearly and understandably is the idea expressed?",
                "scoring_logic": lambda idea_content: 1.0 if len(idea_content.split()) > 10 else (0.5 if len(idea_content.split()) > 5 else 0.2) # Score based on word count
            }
        }

    def _simple_keyword_score(self, text: str, keywords: list, base_score=0.1, max_score=1.0, negative_keywords=None) -> float:
        if negative_keywords is None:
            negative_keywords = []

        score = base_score
        text_lower = text.lower()

        for neg_kw in negative_keywords:
            if neg_kw in text_lower:
                return base_score

        found_keywords_count = 0
        for kw in keywords:
            if kw in text_lower:
                found_keywords_count += 1

        if keywords: # Avoid division by zero if keywords list is empty
            score_increase_factor = found_keywords_count / len(keywords)
            score += (max_score - base_score) * score_increase_factor

        return min(score, max_score)

    def evaluate_idea(self, idea_id: str) -> bool:
        idea = self.idea_db.get_idea_by_id(idea_id)
        if not idea:
            print(f"Error: Idea with ID {idea_id} not found for evaluation.")
            return False

        print(f"Evaluating idea ID: {idea_id} - Content: '{idea['content'][:60]}...'")

        evaluation_successful = True
        for criteria_name, criteria_details in self.criteria.items():
            current_score = criteria_details["scoring_logic"](idea["content"])
            notes = f"Scored based on predefined logic for '{criteria_name}' by {self.evaluator_name}."

            if not self.idea_db.update_idea_evaluation(
                idea_id=idea_id,
                criteria_name=criteria_name,
                score=round(current_score, 2),
                evaluator_name=self.evaluator_name,
                notes=notes
            ):
                evaluation_successful = False
                print(f"Failed to update {criteria_name} for idea {idea_id}")

        if evaluation_successful:
            print(f"Successfully evaluated and updated all criteria for idea ID: {idea_id}")
        else:
            print(f"Some criteria failed to update for idea ID: {idea_id}")

        return evaluation_successful

    def evaluate_all_pending_ideas(self, re_evaluate_all=False):
        all_ideas = self.idea_db.get_all_ideas()
        evaluated_count = 0
        for idea in all_ideas:
            needs_evaluation = False
            if re_evaluate_all:
                needs_evaluation = True
            elif "evaluation_scores" not in idea or not idea["evaluation_scores"]:
                needs_evaluation = True
            else:
                # Check if all current criteria are present in the scores
                # This is a simple check; could be more sophisticated (e.g., check evaluator, timestamp)
                current_criteria_keys = set(self.criteria.keys())
                evaluated_criteria_keys = set(idea["evaluation_scores"].keys())
                if not current_criteria_keys.issubset(evaluated_criteria_keys):
                    needs_evaluation = True

            if needs_evaluation:
                print(f"Idea ID: {idea['id']} requires evaluation (re_evaluate_all={re_evaluate_all}).")
                self.evaluate_idea(idea['id'])
                evaluated_count += 1
        print(f"Evaluation process completed. Processed {len(all_ideas)} ideas, evaluated/re-evaluated {evaluated_count} ideas.")


# Example Usage
if __name__ == '__main__':
    # Setup a test database
    # Using a fixed name for simplicity in example, ensure 'database' directory exists or db_path is adjusted
    # For scripts in subdirectories, relative paths like '../database/...' might be needed if 'database' is a sibling
    db_directory = os.path.join(project_root, 'database')
    os.makedirs(db_directory, exist_ok=True) # Ensure database directory exists
    test_db_file = os.path.join(db_directory, 'eval_module_test_ideas.json')

    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    idea_db_instance = IdeaDatabase(db_path=test_db_file)
    print(f"Using test database: {test_db_file}")

    # Add some ideas
    idea_data_1 = idea_db_instance.add_idea("A new, innovative, and unique concept for modular smartphones. Highly impactful.", "AgentSmith", "NoveltyFocus")
    idea_data_2 = idea_db_instance.add_idea("A practical and feasible plan to build community gardens. This is achievable.", "AgentPractical", "FeasibilityFocus")
    idea_data_3 = idea_db_instance.add_idea("This is a very short idea.", "AgentShort", "Brevity")
    idea_data_4 = idea_db_instance.add_idea("An impossible and unrealistic idea about anti-gravity boots for cats.", "AgentSilly", "ImpossibleDreams")

    eval_module_instance = EvaluationModule(idea_database=idea_db_instance, evaluator_name="TestEvalModule")

    print("\n--- Evaluating specific idea (ID: idea_data_1['id']) ---")
    if idea_data_1:
        eval_module_instance.evaluate_idea(idea_data_1['id'])
        evaluated_idea = idea_db_instance.get_idea_by_id(idea_data_1['id'])
        print("Evaluated Idea 1 Details:")
        print(json.dumps(evaluated_idea, indent=2))

    print("\n--- Evaluating all pending ideas (first pass) ---")
    eval_module_instance.evaluate_all_pending_ideas() # re_evaluate_all is False by default

    print("\n--- Evaluating all ideas (second pass, re-evaluating all) ---")
    eval_module_instance.evaluate_all_pending_ideas(re_evaluate_all=True)

    all_final_ideas = idea_db_instance.get_all_ideas()
    print("\n--- All Ideas After All Evaluations ---")
    for current_idea_item in all_final_ideas:
        print(json.dumps(current_idea_item, indent=2))
        print("-" * 20)

    print(f"Test run complete. Final database at: {test_db_file}")
