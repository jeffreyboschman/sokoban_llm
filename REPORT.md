# Tree-Based Planning with Large Language Models for Sokoban

**Author:** Jeffrey Boschman
**Track:** Track 1 – Tree-Based Planning with LLMs
**Model(s):** Mistral-7B (primary)
**Date:** *[fill in]*

---

## 1. Introduction

Large Language Models (LLMs) have demonstrated strong reasoning and pattern-recognition capabilities across a variety of domains. However, their use in sequential decision-making problems such as planning remains constrained by computational cost, latency, and the difficulty of ensuring reliable long-horizon reasoning.

This project explores a constrained planning setting in which an LLM is restricted to **single-step action prediction**, and a **tree-based search algorithm** is responsible for composing these local predictions into global plans. We study this paradigm in the context of **Sokoban**, a classic planning benchmark with sparse rewards, irreversible actions, and combinatorial state spaces.

The primary objective is to evaluate whether a lightweight LLM-guided policy can effectively guide tree search to solve Sokoban puzzles of increasing difficulty.

---

## 2. Problem Setup

### 2.1 Sokoban Environment

Sokoban is a grid-based puzzle game in which an agent must push boxes onto designated target tiles. The player may move in four cardinal directions (up, down, left, right), subject to the following constraints:

* Boxes may only be pushed, not pulled
* Boxes cannot be pushed into walls or other boxes
* The puzzle is solved when all boxes are placed on target locations

We use puzzles from **David Skinner’s Microban collection**, a widely used benchmark consisting of small but non-trivial Sokoban levels of increasing difficulty.

---

### 2.2 Action Space

The action space is fixed and discrete:

```
A = {up, down, left, right}
```

No other actions (e.g., wait or noop) are permitted.

---

## 3. System Overview

The system consists of three main components:

1. **Sokoban Environment Simulator**
2. **LLM-based One-Step Policy**
3. **Tree-Based Search Algorithm**

A high-level overview of the pipeline is shown below:

```
Current State
     ↓
LLM One-Step Policy
     ↓
Action Probabilities
     ↓
Tree-Based Search
     ↓
Solution Plan (or failure)
```

---

## 4. State Representation

Each Sokoban state is represented as an **ASCII grid**, where characters denote walls, boxes, targets, and the player.

Example:

```
#####
#.@ #
# $ #
# . #
#####
```

This representation is:

* Human-readable
* Compact
* Directly consumable by LLMs without additional preprocessing

### Alternative Representations (Explored)

* Flattened symbolic tokens
* Coordinate-based representations
* Hybrid structured + ASCII formats

*[Results comparing these representations are reported in Section 8.]*

---

## 5. LLM as a One-Step Policy

### 5.1 Constraint Enforcement

The LLM is strictly limited to **one-step prediction**. Given a board state, it produces:

* A probability distribution over `{up, down, left, right}`
* No multi-step reasoning
* No rollouts or internal planning

This constraint is enforced by **logit inspection**, rather than free-form text generation.

---

### 5.2 Logit-Based Action Selection

Instead of sampling text, we:

1. Perform a single forward pass through the model
2. Extract the logits for the **next token**
3. Select logits corresponding to the tokens:

   ```
   " up", " down", " left", " right"
   ```
4. Normalize using softmax to obtain action probabilities

This guarantees that:

* Only valid actions are considered
* The LLM is used purely as a discriminative policy model

---

### 5.3 Prompt Design

The prompt is structured to describe the task succinctly and consistently:

```
You are playing Sokoban.
Given the board state below, predict the best next move.

Board:
<ASCII GRID>

Next move:
```

*[Prompt variants and their effects are discussed in Section 8.]*

---

## 6. Tree-Based Search

### 6.1 Search Algorithm

We implement a **Beam Search** planner guided by LLM action probabilities.

At each depth:

* Each frontier node is expanded
* Candidate successor states are scored using cumulative log-probabilities
* Only the top `K` candidates (beam width) are retained

This approach balances:

* Exploration of multiple action sequences
* Computational efficiency

---

### 6.2 Scoring Function

Each node is scored as:

[
\text{score} = \sum_{t=1}^{T} \log p(a_t \mid s_t)
]

Where:

* ( p(a_t \mid s_t) ) is provided by the LLM
* Log probabilities stabilize accumulation over long horizons

---

### 6.3 Dead-End Detection

The planner prunes:

* Previously visited states
* Invalid transitions
* Branches with no legal successors

Extensive logging is used to track:

* Dead-end pruning
* Beam truncation
* Goal detection

---

## 7. Experimental Setup

### 7.1 Models

* **Primary model:** Mistral-7B (instruct)
* **Inference:** Local GPU / Google Colab
* **Precision:** *[fill in: fp16 / bf16 / etc.]*

---

### 7.2 Puzzles

We evaluate on **10 Sokoban puzzles** from the Microban collection, selected to span:

* Easy
* Medium
* Hard difficulty levels

Puzzle IDs:

```
[fill in puzzle numbers]
```

---

### 7.3 Metrics

We report:

* **Success rate (%)**
* **Average solution length**
* **Average nodes expanded**
* **Average runtime per puzzle**

---

## 8. Results and Analysis

### 8.1 Overall Performance

| Difficulty | Success Rate | Avg Steps | Avg Nodes |
| ---------- | ------------ | --------- | --------- |
| Easy       | *[ ]*        | *[ ]*     | *[ ]*     |
| Medium     | *[ ]*        | *[ ]*     | *[ ]*     |
| Hard       | *[ ]*        | *[ ]*     | *[ ]*     |

---

### 8.2 Effect of State Representation

*[Describe differences observed between ASCII, symbolic, etc.]*

Key observations:

* *[fill in]*
* *[fill in]*

---

### 8.3 Effect of Beam Width

*[Analyze performance vs beam width trade-offs.]*

---

### 8.4 LLM Prediction Quality

We observe that:

* The LLM often assigns near-uniform probabilities in ambiguous states
* Strong preferences emerge near walls, corners, and box interactions
* Incorrect high-confidence predictions can mislead search

*[Insert qualitative examples.]*

---

## 9. Discussion

### 9.1 Single-Step LLM Usage

This approach strictly adheres to the constraint that the LLM performs **no internal planning**. All long-horizon reasoning emerges from the interaction between:

* Local action predictions
* External tree search

This separation improves interpretability and debuggability.

---

### 9.2 Computational Trade-offs

* Beam search reduces branching but risks pruning optimal paths
* LLM inference dominates runtime
* Caching LLM evaluations significantly improves performance

---

### 9.3 Limitations

* Uniform action probabilities in early states
* Sensitivity to prompt phrasing
* Difficulty scaling to large Sokoban instances

---

## 10. Future Work

Possible extensions include:

* Using value estimates in addition to policy logits
* Incorporating learned heuristics
* Fine-tuning the LLM with RL or imitation learning
* Switching to MCTS-style planners
* Allowing limited multi-token reasoning under strict constraints

---

## 11. Conclusion

This project demonstrates that a constrained LLM, when used purely as a one-step policy predictor, can meaningfully guide tree-based planning in Sokoban. While insufficient on its own, the LLM provides valuable local guidance that enables search algorithms to scale to more complex puzzles than naive search alone.