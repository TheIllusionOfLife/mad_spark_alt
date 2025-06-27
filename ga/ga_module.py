# ga/ga_module.py
import random
import os
import sys
import json # For example usage with IdeaDatabase

# Ensure project root is in sys.path for robust module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from database.idea_db import IdeaDatabase # For example usage
    from agents.llm_client import MockLLMClient # For potential LLM-assisted mutation/crossover
except ImportError as e:
    print(f"Warning: Could not import IdeaDatabase or MockLLMClient for ga_module.py example usage: {e}")
    # Define dummy classes if imports fail, so example can still be outlined
    if 'IdeaDatabase' not in globals():
        class IdeaDatabase: # type: ignore
            def __init__(self, db_path): pass
            def get_all_ideas(self): return []
            def add_idea(self, content, source_agent, thinking_method, parent_ids, metadata): pass
            def update_idea_evaluation(self, idea_id, criteria, score, evaluator): pass # Added for example
    if 'MockLLMClient' not in globals():
        class MockLLMClient: # type: ignore
            def __init__(self): pass
            def generate_text(self, prompt, model="mock"): return "LLM-generated variation of: " + prompt[:30]


class GeneticAlgorithmModule:
    def __init__(self, idea_database: IdeaDatabase, llm_client=None, generation_number: int = 0):
        self.idea_db = idea_database
        self.llm_client = llm_client if llm_client else MockLLMClient()
        self.current_generation = generation_number

    def _get_fitness_score(self, idea: dict) -> float:
        if not idea or "evaluation_scores" not in idea or not idea["evaluation_scores"]:
            return 0.0

        novelty = idea["evaluation_scores"].get("novelty", {}).get("score", 0.0)
        impact = idea["evaluation_scores"].get("impact", {}).get("score", 0.0)
        feasibility = idea["evaluation_scores"].get("feasibility", {}).get("score", 0.0)

        fitness = (float(novelty) * 0.4) + (float(impact) * 0.4) + (float(feasibility) * 0.2)
        return round(fitness, 3)


    def selection(self, population: list[dict], num_parents: int) -> list[dict]:
        if not population:
            return []

        for idea in population:
            if "fitness" not in idea:
                idea["fitness"] = self._get_fitness_score(idea)

        # Filter out individuals with no fitness score or fitness of 0 before sorting, if desired
        # or handle them as lowest priority. For now, they'll sort to the bottom.
        fittest_population = [p for p in population if p.get("fitness", 0.0) > 0]
        if not fittest_population: # If all have 0 fitness, return a random sample or top of original
            fittest_population = population

        sorted_population = sorted(fittest_population, key=lambda x: x.get("fitness", 0.0), reverse=True)

        selected_parents = sorted_population[:num_parents]

        if not selected_parents:
            return []
        # If fewer parents selected than requested (due to small population), can either use them or try to fill with replacement
        # For now, just return what was selected.
        if len(selected_parents) < num_parents:
             print(f"Warning: Selected {len(selected_parents)} parents (less than requested {num_parents}) due to population size or fitness scores.")
        return selected_parents

    def crossover_text_simple_split(self, parent1_content: str, parent2_content: str) -> tuple[str, str]:
        if not parent1_content or not parent2_content:
            return parent1_content or "", parent2_content or ""

        p1_words = parent1_content.split()
        p2_words = parent2_content.split()

        if not p1_words or not p2_words:
             return parent1_content, parent2_content

        # Ensure split points are valid
        split_point1 = random.randint(1, len(p1_words)) if len(p1_words) > 0 else 0
        split_point2 = random.randint(1, len(p2_words)) if len(p2_words) > 0 else 0

        child1_content = " ".join(p1_words[:split_point1]) + " " + " ".join(p2_words[split_point2:])
        child2_content = " ".join(p2_words[:split_point2]) + " " + " ".join(p1_words[split_point1:])

        return child1_content.strip(), child2_content.strip()

    def crossover_llm_assisted(self, parent1_content: str, parent2_content: str,
                               prompt_template="Combine the core concepts of these two ideas into a new, synthesized idea: IDEA1: '{idea1}' IDEA2: '{idea2}'") -> str:
        if not self.llm_client:
            print("Warning: LLM client not available for LLM-assisted crossover. Falling back to simple split.")
            return self.crossover_text_simple_split(parent1_content, parent2_content)[0]

        prompt = prompt_template.format(idea1=parent1_content, idea2=parent2_content)
        combined_idea_content = self.llm_client.generate_text(prompt)
        return combined_idea_content


    def mutate_text_char_swap(self, text_content: str, mutation_intensity: float = 0.05) -> str:
        if not text_content: return ""
        chars = list(text_content)
        num_swaps = max(1, int(len(chars) * mutation_intensity)) # Ensure at least one swap if intensity > 0

        for _ in range(num_swaps):
            if len(chars) < 2: break # Cannot swap if less than 2 chars
            idx1, idx2 = random.sample(range(len(chars)), 2)
            chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
        return "".join(chars)

    def mutate_llm_assisted(self, text_content: str,
                            prompt_template="Introduce a surprising variation or a creative twist to the following idea, while retaining its core essence: IDEA: '{idea}'",
                            mutation_strength="moderate") -> str:
        if not self.llm_client:
            print("Warning: LLM client not available for LLM-assisted mutation. Falling back to char swap.")
            return self.mutate_text_char_swap(text_content)

        prompt = prompt_template.format(idea=text_content, strength=mutation_strength)
        mutated_idea_content = self.llm_client.generate_text(prompt)
        return mutated_idea_content

    def run_ga_cycle(self, population_size_target: int = 20, num_parents_to_select: int = 10, offspring_to_generate: int = 10, overall_mutation_chance: float = 0.2, use_llm_crossover=False, use_llm_mutation=False):
        print(f"\n--- Running GA Cycle: Generation {self.current_generation + 1} ---")
        current_population = self.idea_db.get_all_ideas()
        if not current_population:
            print("Population is empty. Cannot run GA cycle.")
            return

        actual_num_parents = min(num_parents_to_select, len(current_population))
        if actual_num_parents < 2 and len(current_population) >=2 :
            actual_num_parents = 2
        elif actual_num_parents < 2:
            print("Not enough individuals in population for selection and meaningful crossover.")
            return

        parents = self.selection(current_population, actual_num_parents)
        if not parents or len(parents) < 2:
            print(f"Not enough parents selected ({len(parents)}) for crossover. GA cycle cannot proceed robustly.")
            return

        print(f"Selected {len(parents)} parents for crossover.")

        new_offspring_data = []
        for i in range(offspring_to_generate // 2):
            p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0]) # Ensure 2 parents

            if use_llm_crossover:
                c1_content = self.crossover_llm_assisted(p1['content'], p2['content'])
                c2_content = self.crossover_llm_assisted(p2['content'], p1['content']) # Or make this more diverse
            else:
                c1_content, c2_content = self.crossover_text_simple_split(p1['content'], p2['content'])

            new_offspring_data.append({"content": c1_content, "parents": [p1['id'], p2['id']]})
            new_offspring_data.append({"content": c2_content, "parents": [p1['id'], p2['id']]})

        print(f"Created {len(new_offspring_data)} offspring through crossover.")

        final_offspring_to_add = []
        for off_data in new_offspring_data:
            mutated_content = off_data["content"]
            mutation_info = {"applied": False, "original": ""}
            if random.random() < overall_mutation_chance:
                original_for_mutation = off_data["content"]
                if use_llm_mutation:
                    mutated_content = self.mutate_llm_assisted(original_for_mutation)
                else:
                    mutated_content = self.mutate_text_char_swap(original_for_mutation, mutation_intensity=0.05)

                mutation_info["applied"] = True
                mutation_info["original"] = original_for_mutation
                print(f"Mutated offspring. Original: '{original_for_mutation[:30]}...' -> Mutated: '{mutated_content[:30]}...'")

            final_offspring_to_add.append({
                "content": mutated_content,
                "parents": off_data["parents"],
                "mutation_info": mutation_info
            })

        self.current_generation += 1
        for child_data in final_offspring_to_add:
            self.idea_db.add_idea(
                content=child_data["content"],
                source_agent="GeneticAlgorithmModule",
                thinking_method=f"{'llm_' if use_llm_crossover else 'simple_'}crossover_{'llm_' if use_llm_mutation and child_data['mutation_info']['applied'] else 'simple_'}mutation",
                parent_ids=child_data["parents"],
                metadata={
                    "generation": self.current_generation,
                    "mutation_applied": child_data['mutation_info']['applied'],
                    "original_if_mutated": child_data['mutation_info']['original'] if child_data['mutation_info']['applied'] else "N/A"
                }
            )
        print(f"Added {len(final_offspring_to_add)} new ideas to database from generation {self.current_generation}.")


# Example Usage (Conceptual)
if __name__ == '__main__':
    db_dir = os.path.join(project_root, 'database')
    os.makedirs(db_dir, exist_ok=True)
    test_ga_db_path = os.path.join(db_dir, 'ga_module_test_ideas.json')

    if os.path.exists(test_ga_db_path):
        os.remove(test_ga_db_path)

    idea_db_instance = IdeaDatabase(db_path=test_ga_db_path)
    print(f"Using GA test database: {test_ga_db_path}")

    # Add initial ideas
    initial_ideas_content = [
        "Solar powered roads that also clean the air using photocatalytic converters.",
        "AI chef personalizing meals for optimal gut health and taste preferences daily.",
        "Self-healing underwater cities built from bio-concrete for marine research.",
        "Adaptive clothing that changes color and texture based on biometric mood data.",
        "A global mesh network of atmospheric CO2 capture drones powered by solar energy."
    ]
    for content in initial_ideas_content:
        idea_db_instance.add_idea(content, "Seed", "InitialSeed", [], {"generation": 0})

    # Mock evaluations
    all_ideas_list = idea_db_instance.get_all_ideas()
    mock_scores = [
        {"novelty": 0.9, "impact": 0.8, "feasibility": 0.5},
        {"novelty": 0.7, "impact": 0.9, "feasibility": 0.8},
        {"novelty": 0.8, "impact": 0.6, "feasibility": 0.3},
        {"novelty": 0.6, "impact": 0.5, "feasibility": 0.7},
        {"novelty": 0.9, "impact": 0.9, "feasibility": 0.4}
    ]
    for i, idea_item in enumerate(all_ideas_list):
        if i < len(mock_scores):
            for crit, score_val in mock_scores[i].items():
                idea_db_instance.update_idea_evaluation(idea_item['id'], crit, score_val, "mock_evaluator")

    ga_module_instance = GeneticAlgorithmModule(idea_database=idea_db_instance, llm_client=MockLLMClient())

    print("\n--- TESTING SIMPLE GA CYCLE (NO LLM) ---")
    ga_module_instance.run_ga_cycle(num_parents_to_select=4, offspring_to_generate=4, overall_mutation_chance=0.5, use_llm_crossover=False, use_llm_mutation=False)

    print("\n--- Ideas after 1st GA cycle (simple): ---")
    ideas_after_simple_ga = idea_db_instance.get_all_ideas()
    for idea_idx, idea_item_simple in enumerate(ideas_after_simple_ga):
        meta = idea_item_simple.get('metadata',{})
        print(f"Idea {idea_idx + 1}: {idea_item_simple['content'][:70]}... (Gen: {meta.get('generation')}, Fitness: {ga_module_instance._get_fitness_score(idea_item_simple)})")

    # Add more distinct ideas for LLM cycle
    idea_db_instance.add_idea("Universal basic income funded by carbon tax and AI productivity gains.", "Seed", "InitialSeed", [], {"generation": 0})
    idea_db_instance.add_idea("Personalized education paths generated by AI tutors adapting in real-time.", "Seed", "InitialSeed", [], {"generation": 0})

    all_ideas_list_updated = idea_db_instance.get_all_ideas()
    new_mock_scores = [
        {"novelty": 0.95, "impact": 0.95, "feasibility": 0.4},
        {"novelty": 0.85, "impact": 0.9, "feasibility": 0.6}
    ]
    # Apply scores to the newly added ideas (assuming they are at the end)
    start_index_for_new_scores = len(all_ideas_list_updated) - len(new_mock_scores)
    for i in range(len(new_mock_scores)):
        idea_to_score_idx = start_index_for_new_scores + i
        if idea_to_score_idx < len(all_ideas_list_updated):
            idea_id_to_score = all_ideas_list_updated[idea_to_score_idx]['id']
            for crit, score_val in new_mock_scores[i].items():
                 idea_db_instance.update_idea_evaluation(idea_id_to_score, crit, score_val, "mock_evaluator_new")


    print("\n\n--- TESTING LLM-ASSISTED GA CYCLE ---")
    ga_module_instance.run_ga_cycle(num_parents_to_select=5, offspring_to_generate=4, overall_mutation_chance=0.6, use_llm_crossover=True, use_llm_mutation=True)

    print("\n--- Ideas after LLM-assisted GA cycle: ---")
    ideas_after_llm_ga = idea_db_instance.get_all_ideas()
    for idea_idx, idea_item_llm in enumerate(ideas_after_llm_ga):
        meta_llm = idea_item_llm.get('metadata',{})
        print(f"Idea {idea_idx + 1}: {idea_item_llm['content'][:70]}... (Gen: {meta_llm.get('generation')}, Method: {idea_item_llm.get('thinking_method')}, Fitness: {ga_module_instance._get_fitness_score(idea_item_llm)})")

    print(f"\nGA test run complete. Final database at: {test_ga_db_path}")
