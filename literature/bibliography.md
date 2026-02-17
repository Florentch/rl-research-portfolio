# Annotated Bibliography

Scientific literature and resources shaping my understanding of Deep RL. Papers marked as *read* have been studied; the rest are on my reading list as I build up the theoretical foundation alongside my projects.

---

## Online Courses & Tutorials

**[Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)** - OpenAI *(read)*
OpenAI's practical introduction to Deep RL — covers the core vocabulary (states, policies, value functions) and links directly to clean algorithm implementations.

**[A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)** - Lilian Weng (2018) *(read)*
A thorough blog post covering MDP formalism through policy gradient methods — good as a map of the field before going into individual papers.

**[RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)** - DeepMind / UCL (2015) *(in progress — lecture 3/10)*
The standard lecture series for classical RL theory: MDPs, dynamic programming, model-free control. On my list to finish before moving deeper into the MARL literature.

---

## Multi-Agent Systems & Emergence (Primary Focus)

**[Emergent Tool Use from Multi-Agent Autocurricula](https://arxiv.org/pdf/1909.07528)** - Baker et al. (OpenAI, 2020) *(read)*
Agents in a Hide & Seek environment develop strategies like door blocking and ramp surfing purely through competition — no explicit reward shaping needed. The clearest demonstration I've seen of emergence through auto-curricula.

**[Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://nature.com/articles/s41586-019-1724-z)** - Vinyals et al. (DeepMind, 2019) *(to read)*
Grandmaster-level StarCraft II via a self-play league that prevents cyclic strategies — a key reference for scaling RL to long-horizon, imperfect information games.

**[Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/pdf/2107.12808)** - Team Open-Ended Learning (DeepMind, 2021) *(to read)*
Training on a procedurally generated universe (XLand) yields agents that generalize zero-shot to new tasks — directly relevant to my interest in open-ended environments.

---

## Foundational Algorithms

**[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)** - Mnih et al. (DeepMind, 2013) *(to read)*
The DQN paper — first demonstration of learning from raw pixels across multiple Atari games. Essential starting point for understanding value-based methods.

**[Proximal Policy Optimization Algorithms (PPO)](https://arxiv.org/pdf/1707.06347)** - Schulman et al. (OpenAI, 2017) *(to read)*
The algorithm behind most of my current projects. PPO's clipped objective makes it stable enough for practical use while remaining conceptually simple.

**[World Models](https://arxiv.org/pdf/1803.10122)** - Ha & Schmidhuber (2018) *(to read)*
Trains agents inside a learned model of the environment ("dream") rather than the real one — an interesting angle on sample efficiency and model-based planning.

---

## Generalization & Robustness

**[Quantifying Generalization in Reinforcement Learning](https://arxiv.org/pdf/1812.02341)** - Cobbe et al. (OpenAI, 2018) *(to read)*
Introduces ProcGen to test whether agents actually generalize or just memorize level layouts — directly relevant to what I'm trying to avoid in my own environments.

**[Illuminating Search Spaces by Mapping Elites](https://arxiv.org/pdf/1504.04909)** - Mouret & Clune (2015) *(to read)*
Quality-Diversity algorithms that maintain a diverse archive of high-performing solutions rather than collapsing to a single optimum — an alternative worth exploring for robust multi-agent training.

---

## Challenges & Safety

**[The Surprising Creativity of Digital Evolution](https://arxiv.org/pdf/1803.03453)** - Lehman et al. (2018) *(to read)*
A collection of reward hacking examples where agents exploit unintended loopholes — a good reminder of how hard it is to specify what you actually want.

**[Adversarial Policies: Attacking Deep Reinforcement Learning Agents](https://arxiv.org/pdf/1905.10615)** - Gleave et al. (2020) *(to read)*
Agents can beat superhuman opponents not by playing better, but by performing movements that confuse the opponent's network — raises real questions about robustness in competitive settings.