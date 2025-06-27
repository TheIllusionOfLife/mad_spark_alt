# database/idea_db.py
import json
import os
import uuid
from datetime import datetime, timezone

class IdeaDatabase:
    def __init__(self, db_path='database/ideas.json'):
        self.db_path = db_path
        # Ensure the directory for the db_path exists
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._ensure_db_file_exists()

    def _ensure_db_file_exists(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump([], f) # Initialize with an empty list

    def _load_ideas(self) -> list:
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                ideas = json.load(f)
            return ideas
        except FileNotFoundError:
            # If file not found (e.g. after a manual deletion), re-initialize
            print(f"Warning: Database file {self.db_path} not found. Initializing new one.")
            self._ensure_db_file_exists()
            return []
        except json.JSONDecodeError:
            # If file is corrupted, log error, maybe backup and re-initialize
            print(f"Error: Could not decode JSON from {self.db_path}. Check file integrity.")
            # For now, returning empty list to prevent crash, but a backup/restore strategy might be needed
            return []


    def _save_ideas(self, ideas: list):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(ideas, f, indent=4, ensure_ascii=False)

    def add_idea(self, content: str, source_agent: str, thinking_method: str,
                 prompt_used: str = "", parent_ids: list = None,
                 metadata: dict = None) -> dict:
        '''
        Adds a new idea to the database.
        '''
        if parent_ids is None:
            parent_ids = []
        if metadata is None:
            metadata = {}

        ideas = self._load_ideas()

        new_idea = {
            "id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "content": content,
            "source_agent": source_agent,
            "thinking_method": thinking_method,
            "prompt_used": prompt_used,
            "evaluation_scores": {},
            "generation": 0,
            "parent_ids": parent_ids,
            "metadata": metadata
        }

        ideas.append(new_idea)
        self._save_ideas(ideas)
        print(f"Idea added with ID: {new_idea['id']}")
        return new_idea

    def get_idea_by_id(self, idea_id: str) -> dict | None:
        ideas = self._load_ideas()
        for idea in ideas:
            if idea["id"] == idea_id:
                return idea
        return None

    def get_all_ideas(self) -> list:
        return self._load_ideas()

    def update_idea_evaluation(self, idea_id: str, criteria_name: str, score: float, evaluator_name: str, notes: str = "") -> bool:
        '''
        Updates or adds an evaluation score for a specific criterion of an idea.
        '''
        ideas = self._load_ideas()
        idea_found = False
        for idea_obj in ideas: # Renamed to avoid conflict with outer 'idea' if any
            if idea_obj["id"] == idea_id:
                if "evaluation_scores" not in idea_obj:
                    idea_obj["evaluation_scores"] = {}

                idea_obj["evaluation_scores"][criteria_name] = {
                    "score": score,
                    "evaluator": evaluator_name,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "notes": notes
                }
                idea_found = True
                break

        if idea_found:
            self._save_ideas(ideas)
            print(f"Evaluation updated for idea ID: {idea_id}, criterion: {criteria_name}")
            return True
        else:
            print(f"Error: Idea ID {idea_id} not found for updating evaluation.")
            return False

    def get_ideas_by_filter(self, filter_func) -> list:
        ideas = self._load_ideas()
        return [idea_item for idea_item in ideas if filter_func(idea_item)]

# Example Usage
if __name__ == '__main__':
    # Use a clearly named test database file
    test_db_file_path = 'database/test_ideas_db.json'

    # Clean up previous test file if it exists
    if os.path.exists(test_db_file_path):
        os.remove(test_db_file_path)
        print(f"Removed old test database file: {test_db_file_path}")

    # Initialize IdeaDatabase with the specific test file path
    idea_db_instance = IdeaDatabase(db_path=test_db_file_path)
    print(f"Using test database file: {idea_db_instance.db_path}")

    # Add some ideas
    idea1_data = idea_db_instance.add_idea(
        content="Use solar panels that look like roof tiles for aesthetic solar power.",
        source_agent="HumanInputAgent",
        thinking_method="Analogy",
        prompt_used="How can we make solar panels less intrusive?",
        metadata={"keywords": ["solar", "aesthetics", "roofing"]}
    )

    idea2_data = idea_db_instance.add_idea(
        content="Develop a bioluminescent paint for nighttime visibility on roads.",
        source_agent="AbductionAgent",
        thinking_method="Abduction",
        prompt_used="Observation: Roads are hard to see at night without power. What if paint could glow?",
        parent_ids=[],
        metadata={"keywords": ["bioluminescence", "safety", "roads"]}
    )

    print("\n--- All Ideas in DB ---")
    current_ideas = idea_db_instance.get_all_ideas()
    for current_idea in current_ideas:
        print(json.dumps(current_idea, indent=2))

    # Update evaluation for an idea
    if current_ideas:
        first_idea_id = current_ideas[0]['id']
        print(f"\n--- Updating evaluation for idea: {first_idea_id} ---")
        idea_db_instance.update_idea_evaluation(
            idea_id=first_idea_id,
            criteria_name="Novelty",
            score=0.9,
            evaluator_name="AIReviewerV1",
            notes="Highly novel combination of existing tech."
        )
        idea_db_instance.update_idea_evaluation(
            idea_id=first_idea_id,
            criteria_name="Feasibility",
            score=0.7,
            evaluator_name="EngineerReview",
            notes="Requires specific tile manufacturing but possible."
        )

        print("\n--- Updated Idea ---")
        updated_single_idea = idea_db_instance.get_idea_by_id(first_idea_id)
        print(json.dumps(updated_single_idea, indent=2))

    # Example of filtering ideas
    print("\n--- Solar Ideas ---")
    def solar_filter(idea_to_check):
        return "solar" in idea_to_check["content"].lower() or \
               ("keywords" in idea_to_check["metadata"] and "solar" in idea_to_check["metadata"]["keywords"])

    solar_ideas_list = idea_db_instance.get_ideas_by_filter(solar_filter)
    for solar_idea in solar_ideas_list:
        print(json.dumps(solar_idea, indent=2))

    # It's good practice to allow cleanup of test files,
    # but for automated runs, sometimes it's better to inspect the file after.
    # print(f"Test database content saved to: {test_db_file_path}")
