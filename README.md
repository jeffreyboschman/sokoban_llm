## Track 1: Tree-Based Planning with LLMs

### Objective
 
Develop a tree-based planning system that uses an LLM as a one-step action predictor to solve
Sokoban puzzles of increasing complexity.

### Task Description

Sokoban is a classic puzzle game where the player pushes boxes onto target locations. Your system should:
1. Parse Sokoban puzzle states from text format
2. Use an LLM to predict single actions given the current state
3. Implement a tree search algorithm that leverages these predictions
4. Solve puzzles with increasing difficulty

### Constraints and Requirements

- LLM Usage: The LLM can only be used as a one-step predictor. Given a board state, it should output a single action (up, down, left, right) with optional confidence scores or reasoning
- Model Selection: Smaller open-source models preferred (e.g., Llama-7B, Mistral-7B, or smaller)
- Data Source: Use puzzles from David Skinner's Microban collection:
http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban_files/Microban.txt


### Technical Specifications

- State Representation: Design how to represent the board state for the LLM (text, ASCII art, structured format, etc.)
- Action Space: Standard Sokoban actions (up, down, left, right)
- Search Algorithm: Implement a tree-based search (e.g., MCTS, A*, beam search) that uses LLM predictions to guide exploration
- [Optional] Training with RL: Train the model with RL and evaluate your new model against the search powered pipeline.
- Evaluation: Test on at least 10 puzzles of varying difficulty from Microban

### Deliverables

1. Code Implementation:
- Sokoban environment parser and simulator
- LLM integration for action prediction
- Tree search algorithm implementation
- Evaluation framework

2. Experimental Analysis:
- Compare different state representations for the LLM
- Analyze the effect of different search strategies
- Study how LLM prediction quality affects solving performance
- Success rate vs. puzzle complexity analysis

3. Discussion:
- How does your approach handle the constraint of single-step LLM usage?
- What are the computational trade-offs of your search strategy?
- How might you improve the system with more LLM calls or different architectures?
- [Optional] RL-trained model vs Tree-Search