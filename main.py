# main.py
import os
import json
from agents.llm_client import MockLLMClient
from agents.qadi_agents import QADIQuestionAgent, QADIAnswerAgent, QADIDeeperQuestionAgent, QADIInsightAgent
from agents.abduction_agent import AbductionAgent
from database.idea_db import IdeaDatabase
from evaluation.evaluation_module import EvaluationModule
from ga.ga_module import GeneticAlgorithmModule

def full_qadi_cycle_example(theme: str, idea_db: IdeaDatabase, llm_client):
    print(f"\n--- Running Full QADI Cycle for theme: '{theme}' ---")

    # 1. Question Agent
    q_agent = QADIQuestionAgent(llm_client=llm_client)
    # In a real system, the agent's execute method would make an LLM call.
    # Here, it returns a simulated response which includes the prompt.
    # For this example, let's extract a placeholder question from the simulated response.
    question_prompt_sim_resp = q_agent.execute(theme=theme)
    # This is a mock extraction. A real system would parse LLM output.
    mock_question = f"What are innovative solutions for {theme}?"
    print(f"Generated Question (Simulated): {mock_question}")
    idea_db.add_idea(content=mock_question, source_agent="QADIQuestionAgent", thinking_method="qadi_question", prompt_used=question_prompt_sim_resp)

    # 2. Answer Agent
    a_agent = QADIAnswerAgent(llm_client=llm_client)
    answer_prompt_sim_resp = a_agent.execute(question=mock_question)
    mock_answer = f"One innovative solution for {theme} is using AI-driven personalized learning."
    print(f"Generated Answer (Simulated): {mock_answer}")
    idea_db.add_idea(content=mock_answer, source_agent="QADIAnswerAgent", thinking_method="qadi_answer", prompt_used=answer_prompt_sim_resp)

    # 3. Deeper Question Agent
    dq_agent = QADIDeeperQuestionAgent(llm_client=llm_client)
    deeper_q_prompt_sim_resp = dq_agent.execute(answer=mock_answer)
    mock_deeper_question = f"How can AI ensure {theme} solutions are equitable?"
    print(f"Generated Deeper Question (Simulated): {mock_deeper_question}")
    idea_db.add_idea(content=mock_deeper_question, source_agent="QADIDeeperQuestionAgent", thinking_method="qadi_deeper_question", prompt_used=deeper_q_prompt_sim_resp)

    # (For a full cycle, we'd need an answer to the deeper question. Let's simulate that too)
    mock_deeper_answer = f"AI can ensure equity in {theme} by using diverse datasets and fairness algorithms."
    print(f"Generated Deeper Answer (Simulated): {mock_deeper_answer}")
    # Not adding this to DB as it's an intermediate step for the Insight agent.

    # 4. Insight Agent
    i_agent = QADIInsightAgent(llm_client=llm_client)
    insight_prompt_sim_resp = i_agent.execute(
        original_question=mock_question,
        answer_to_original_question=mock_answer,
        deeper_question=mock_deeper_question,
        answer_to_deeper_question=mock_deeper_answer
    )
    mock_insight = f"Key insight for {theme}: Combine AI-driven personalization with robust equity frameworks using diverse data."
    print(f"Generated Insight (Simulated): {mock_insight}")
    idea_db.add_idea(content=mock_insight, source_agent="QADIInsightAgent", thinking_method="qadi_insight", prompt_used=insight_prompt_sim_resp)

    return mock_insight


def abduction_example(observation: str, idea_db: IdeaDatabase, llm_client):
    print(f"\n--- Running Abduction Example for observation: '{observation}' ---")
    ab_agent = AbductionAgent(llm_client=llm_client)
    # Similar to QADI, execute returns a simulated response.
    abduction_prompt_sim_resp = ab_agent.execute(observation=observation)
    mock_abduction = f"The most plausible explanation for '{observation}' is the increasing adoption of remote work."
    print(f"Generated Abduction (Simulated): {mock_abduction}")
    idea_db.add_idea(content=mock_abduction, source_agent="AbductionAgent", thinking_method="abduction", prompt_used=abduction_prompt_sim_resp)
    return mock_abduction

def run_demo():
    print("--- Mad Spark Alternative - System Demo ---")

    # Setup
    # Ensure database directory exists for the JSON file
    db_dir = "database"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "main_demo_ideas.json")
    if os.path.exists(db_path):
        os.remove(db_path) # Clean slate for demo

    llm_client = MockLLMClient()
    idea_database = IdeaDatabase(db_path=db_path)
    evaluation_module = EvaluationModule(idea_database=idea_database, evaluator_name="DemoEvaluator")
    ga_module = GeneticAlgorithmModule(idea_database=idea_database, llm_client=llm_client)

    # 1. Generate some initial ideas using different agents
    full_qadi_cycle_example(theme="sustainable urban transport", idea_db=idea_database, llm_client=llm_client)
    abduction_example(observation="city parks are becoming more crowded on weekdays", idea_db=idea_database, llm_client=llm_client)

    # Add a few more manually for variety for GA
    idea_database.add_idea("Drone-based delivery systems for last-mile logistics.", "ManualSeed", "Brainstorm")
    idea_database.add_idea("Gamified public transport app to encourage off-peak travel.", "ManualSeed", "Brainstorm")


    # 2. Evaluate all ideas
    print("\n--- Evaluating all generated ideas ---")
    evaluation_module.evaluate_all_pending_ideas(re_evaluate_all=True)

    print("\n--- Ideas after initial generation and evaluation: ---")
    for idea in idea_database.get_all_ideas():
        fitness = ga_module._get_fitness_score(idea) # Use GA's fitness for display
        print(f"- {idea['content'][:50]}... (Scores: {idea.get('evaluation_scores', {})}, Fitness: {fitness})")


    # 3. Run a GA cycle (simple, no LLM for this part of demo)
    print("\n--- Running a Genetic Algorithm Cycle (Simple) ---")
    ga_module.run_ga_cycle(
        num_parents_to_select=4,
        offspring_to_generate=4,
        overall_mutation_chance=0.5,
        use_llm_crossover=False,
        use_llm_mutation=False
    )

    # 4. Evaluate the new generation of ideas
    print("\n--- Evaluating ideas after GA cycle ---")
    evaluation_module.evaluate_all_pending_ideas(re_evaluate_all=True) # Re-evaluate all to score new ones

    print("\n--- All Ideas after GA cycle and re-evaluation: ---")
    final_ideas = idea_database.get_all_ideas()
    for idea in final_ideas:
        fitness = ga_module._get_fitness_score(idea)
        print(f"- Gen: {idea.get('metadata',{}).get('generation', 0)} - {idea['content'][:50]}... (Scores: {idea.get('evaluation_scores', {})}, Fitness: {fitness})")

    print(f"\nDemo complete. Database at: {db_path}")
    print(f"Total ideas in DB: {len(final_ideas)}")

if __name__ == "__main__":
    run_demo()
