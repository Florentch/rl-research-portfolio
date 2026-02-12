# Annotated Bibliography

This document tracks the scientific literature that influences my research and implementation choices. It covers foundational algorithms, multi-agent dynamics, and generalization challenges.

## Multi-Agent Systems & Emergence (Primary Focus)

**[Emergent Tool Use from Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528)** - Baker et al. (OpenAI, 2020)
* **Key Insight:** Demonstrates that agents in a "Hide and Seek" environment naturally discover complex strategies (blocking doors, using ramps, surfing on boxes) through competition.
* **Relevance:** Proves that complexity can emerge from simple rules and auto-curricula without explicit reward shaping.

**[Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://nature.com/articles/s41586-019-1724-z)** - Vinyals et al. (DeepMind, 2019)
* **Key Insight:** Addresses long-term planning and imperfect information in a real-time strategy game. Introduces "The League" training system to prevent cyclic strategies.
* **Relevance:** Represents a milestone in scaling RL to complex, high-dimensional strategy games.

**[Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/pdf/2107.12808)** - Team Open-Ended Learning (DeepMind, 2021)
* **Key Insight:** Introduces XLand, a vast, procedurally generated universe. Shows that training on a sufficiently diverse distribution of tasks leads to zero-shot generalization on new tasks.
* **Relevance:** Considered a significant step towards general-purpose agents (AGI) via open-ended environments.

## Foundational Algorithms

**[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)** - Mnih et al. (DeepMind, 2013)
* **Key Insight:** The seminal paper introducing the Deep Q-Network (DQN). It demonstrated for the first time that an agent could learn to play diverse Atari games directly from raw pixel inputs.
* **Relevance:** The foundation of modern Deep RL for visual environments.

**[Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/pdf/1707.06347)** - Schulman et al. (OpenAI, 2017)
* **Key Insight:** Proposes a policy gradient method that alternates between sampling data and optimizing a "clipped" surrogate objective function.
* **Relevance:** Currently the standard for on-policy algorithms due to its balance of ease of implementation, sample efficiency, and stability. This is the algorithm I implemented for the Mario agent.

**[World Models](https://arxiv.org/pdf/1803.10122)** - Ha & Schmidhuber (2018)
* **Key Insight:** Proposes training an agent inside a learned "dream" (a predictive model of the environment) rather than the real environment. Tested on Doom and CarRacing.
* **Relevance:** Highlights the importance of representation learning and model-based planning for efficiency.

## Generalization & Robustness

**[Quantifying Generalization in Reinforcement Learning](https://arxiv.org/pdf/1812.02341)** - Cobbe et al. (OpenAI, 2018)
* **Key Insight:** Introduces the CoinRun/ProcGen benchmark to measure if agents are actually learning skills or just memorizing specific level layouts.
* **Relevance:** Crucial for distinguishing between overfitting and true intelligence in RL.

**[Illuminating Search Spaces by Mapping Elites](https://arxiv.org/pdf/1504.04909)** - Mouret & Clune (2015)
* **Key Insight:** Introduces "Quality-Diversity" algorithms (Map-Elites). Instead of finding one optimal solution, the goal is to find a archive of high-performing but diverse solutions.
* **Relevance:** Provides an alternative to gradient descent for creating robust populations of agents (e.g., in robotics).

## Challenges & Safety

**[The Surprising Creativity of Digital Evolution](https://arxiv.org/pdf/1803.03453)** - Lehman et al. (2018)
* **Key Insight:** A survey of "specification gaming" where algorithms exploit loopholes in the reward function to achieve high scores in unintended (and often absurd) ways.
* **Relevance:** A critical lesson on the difficulty of designing correct reward functions (Reward Shaping).

**[Adversarial Policies: Attacking Deep Reinforcement Learning Agents](https://arxiv.org/pdf/1905.10615)** - Gleave et al. (2020)
* **Key Insight:** Shows that an agent can learn to defeat a superhuman opponent not by playing better, but by performing "adversarial" movements that confuse the opponent's neural network.
* **Relevance:** Highlights the fragility of current Deep RL models in competitive settings.