# Mad Spark Alternative System Architecture

## Overall Concept

The Mad Spark Alternative system is designed to facilitate creative idea generation and refinement through a multi-agent approach. It leverages diverse "thinking methods" (prompts and logic) assigned to different agents. These agents generate ideas, which are then stored, evaluated, and potentially evolved using a genetic algorithm. The core interaction involves agents applying thinking methods, ideas being stored and scored, and a genetic algorithm module optionally refining these ideas over generations.

## Components

### Thinking Method Library

*   **Storage:** This library will store various thinking methods. Each method can range from a simple prompt template to more complex chains of prompts or conditional logic. Initially, these might be stored as structured text files (e.g., JSON or YAML) or Python modules within the `thinking_methods/` directory.
*   **Provision:** The Multi-Agent Orchestrator will access this library to retrieve and assign specific thinking methods to agents based on the current strategy or phase of idea generation. Each method will have a unique identifier.

### Multi-Agent Orchestration

*   **Agent Management:** The orchestrator is responsible for instantiating and managing a pool of agents. Each agent can be configured with specific characteristics or capabilities, though initially, they might be largely homogeneous, differentiated primarily by the thinking methods assigned to them.
*   **Thinking Method Assignment:** The orchestrator assigns thinking methods from the library to individual agents. This assignment can be static or dynamic, potentially changing based on the performance of methods or the stage of the creative process.
*   **Information Flow (QADI Cycle):** The orchestrator manages the flow of information, which can be conceptualized through a Question-Answer-Decision-Instruction (QADI) cycle or a simpler variant.
    *   **Question:** The orchestrator might pose a challenge or a question.
    *   **Answer (Idea Generation):** An agent, using its assigned thinking method (e.g., by formatting a prompt and sending it to an LLM), generates an idea (the "answer").
    *   **Decision/Storage:** The generated idea is passed to the Idea Database.
    *   **Instruction:** The orchestrator might then instruct the Evaluation Module to assess the new idea or instruct an agent to build upon an existing idea.

### Idea Database

*   **Schema:** The database will store ideas with the following attributes:
    *   `idea_id`: Unique identifier for the idea (e.g., UUID).
    *   `content`: The actual text or description of the idea.
    *   `source_agent_id`: Identifier of the agent that generated the idea.
    *   `thinking_method_id`: Identifier of the thinking method used to generate the idea.
    *   `evaluation_scores`: A dictionary or structured field to store various evaluation scores (e.g., `{'originality': 0.8, 'feasibility': 0.6}`).
    *   `generation_number`: An integer indicating the generation this idea belongs to (relevant for GA).
    *   `parent_idea_ids`: A list of `idea_id`s from which this idea was derived (for GA tracking).
    *   `timestamp`: When the idea was created/updated.
*   **Access:** The database will provide CRUD (Create, Read, Update, Delete) operations. Agents will primarily add new ideas. The Evaluation Module will update ideas with scores. The GA Module will read ideas for selection and add new (evolved) ideas. Initially, this could be a simple file-based database (e.g., JSONL, SQLite) located in the `database/` directory.

### Evaluation Module

*   **Idea Reception:** This module will be triggered by the orchestrator or by new idea events in the database. It retrieves ideas needing evaluation.
*   **Criteria Application:**
    *   **Initial:** Evaluation might be manual (a human provides scores through a simple interface or by editing a file) or based on very simple heuristics (e.g., keyword counts, length).
    *   **Later:** This could involve more sophisticated methods, including using another LLM with specific evaluation prompts, or integrating predefined metrics.
*   **Score Storage:** The module updates the `evaluation_scores` field for the corresponding idea in the Idea Database. This module will reside in `evaluation/`.

### Genetic Algorithm (GA) Module

*   **Interaction with Idea Database:**
    *   **Selection:** The GA module will query the Idea Database to select parent ideas based on their `evaluation_scores`. Higher-scoring ideas will have a higher probability of being selected.
*   **Crossover (Conceptual):** For text-based ideas, crossover could involve:
    *   Combining sections of parent ideas (e.g., sentence A from parent 1 + sentence B from parent 2).
    *   Identifying key concepts or themes from parent ideas and generating a new idea that synthesizes them (potentially using an LLM).
*   **Mutation (Conceptual):** Mutation could involve:
    *   Altering words or phrases in an idea (e.g., using synonyms, antonyms).
    *   Adding, removing, or rephrasing sentences.
    *   Using an LLM to "slightly alter" or "suggest variations" of an existing idea.
*   The GA module will be developed in the `ga/` directory. New ideas generated by the GA will be added to the Idea Database, typically marked with an incremented `generation_number` and linked to their `parent_idea_ids`.

## Data Flow

1.  **Initialization:** The Multi-Agent Orchestrator is configured with a set of agents and available thinking methods from the Thinking Method Library.
2.  **Idea Generation Trigger:** The Orchestrator selects an agent and assigns it a thinking method (e.g., a specific prompt template from the library). It might also provide context, like a problem statement or existing high-quality ideas.
3.  **Agent Processing:** The agent utilizes the assigned thinking method. If it's a prompt, the agent might interact with an external LLM, sending the prompt and receiving generated text (the idea).
4.  **Idea Storage:** The newly generated idea is sent to the Idea Database and stored with its metadata (source agent, thinking method used, initial generation number, etc.).
5.  **Evaluation Trigger:** The Orchestrator (or a database trigger) notifies the Evaluation Module about the new idea.
6.  **Idea Assessment:** The Evaluation Module retrieves the idea from the database. It applies its evaluation criteria (manual or automated) and calculates scores.
7.  **Score Update:** The Evaluation Module updates the idea's record in the Idea Database with the new evaluation scores.
8.  **GA Cycle (Optional/Iterative):**
    a.  **Selection:** The GA Module queries the Idea Database for highly-rated ideas from the current generation.
    b.  **Evolution:** It applies crossover and mutation operations to the selected parent ideas to create new offspring ideas.
    c.  **New Idea Storage:** These new (evolved) ideas are stored in the Idea Database, typically marked as belonging to the next generation and with links to their parent ideas.
    d.  These new ideas then go through the evaluation process (steps 5-7).
9.  **Iteration:** The process (steps 2-8) can iterate, with agents potentially building on previously generated/evaluated ideas, or the GA module continually refining the pool of ideas.

## Directory Structure (Initial)

A preliminary directory structure will be:

*   `mad_spark_alternative/`
    *   `agents/`: Contains code related to agent behavior and LLM interaction.
    *   `database/`: Manages the idea database (e.g., schema, connection, initial data files).
    *   `evaluation/`: Houses the logic for evaluating ideas.
    *   `ga/`: Contains the genetic algorithm implementation for evolving ideas.
    *   `thinking_methods/`: Stores the library of thinking methods/prompts.
    *   `orchestration/`: (Could also be `core/` or `main/`) For the Multi-Agent Orchestrator logic.
    *   `main.py`: Main script to run the system.
    *   `config.py`: System-wide configurations.
    *   `README.md`
    *   `ARCHITECTURE.md` (this file)
