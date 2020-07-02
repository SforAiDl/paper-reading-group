# Reading Notes

# Learning to Communicate with Deep MARL

[https://arxiv.org/abs/1605.06676](https://arxiv.org/abs/1605.06676)

*Jakob N. Foerster, Yannis M. Assael, Nando de Freitas, Shimon Whiteson*

## Key Learnings:

- The need for communication between agents
- Agents need to cooperate to maximise shared utility
- Two approaches:
    - Reinforced Inter-Agent Learning (RIAL): Uses Deep Recurrent Q-learning
    - Differentiable Inter-Agent Learning (DIAL): Agents back-propagate through sharing gradients directly.
- Centralised Learning but decentralised execution

## Communication:

- Very broad field in AI. Types of classification:
    - Predefined vs Learned Communication Protocols  (here, Learned)
    - Planning vs Learning Methods  (here, Learning)
    - Cooperative vs Competitive Settings  (here, Cooperative)
    - Evolution vs Reinforcement Learning  (here, RL)
- When would agents need communication?
    - Cooperative games
    - Partially observable (both states and other agents' actions)
    - Aim would be to learn a good communication protocol so both perform well and also teach each other well
- Reinforcement Learning and Differentiable Communication were probably used for the first time to successfully learn communication protocols involving sequences and raw images
- Centralised Learning and Decentralised Execution and the advantage of using Deep Learning to perform this.

## Quick Background:

![https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

### Deep Q-learning (DQN):

- Neural Networks can approximate Q-values, Value functions and Policy functions
- Action-selection is $\epsilon$-greedy
- MSE Loss function
- Target DQN function: $y^{DQN}_i = r + \gamma \max_{a'}Q(s', a'; \theta_i^-)$
- Experience Replay

### Deep Recurrent Q-learning:

- Replace FNN approximators with RNNs
- Helps model partial observability because of hidden states

### Multi-Agent RL:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled.png)

- Types of MARL:
    - Centralized vs Decentralized
    - Cooperative vs Competitive

## Problem at Hand:

**Find the optimal communication protocol**

- Each agent receives a partial observation correlated with the true Markov state
- Every timestep, each agent
    - takes an "environment action" which acts on the environment and gives the reward
    - takes a "communication action" which is the action visible to the other agent
- A communication protocol mapping from action and observation histories to sequences of messages must be developed
- Difficulties in establishing the protocol:
    - Protocol space is really large
    - Agents not only need to know what messages to communicate but also how to interpret the other agent's message
    - Positive rewards are extremely sparse
- Centralised learning (communication is not restricted) and decentralised execution (communication is restricted)

## Reinforced Inter-Agent Learning (RIAL):

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%201.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%201.png)

- Simplest implementation for the communication protocol: Add DRQN with Independent Q-learning
- Have two separate Q-networks for each agent $Q^a_u$ and $Q^a_m$for both the environment actions and the community actions
- $Q^a(o^a_t, m^{a'}_{t-1},h^a_{t-1},u^a)$ will be the Q-network for the environment action selection
- $a$ is the index of the agent
- Choose actions using $\epsilon$-greedy policy for both "environment action" and "community action"
- Maximise over $U$ and then over $M$
- Disable experience replay to avoid non-stationarity (underlying Markov states must be the same at each timestep for both o
- Communication protocol here is only the communication action $m$ being taken
- Loss function is the DQN loss of environment action and communication action respectively for $Q_u$ and $Q_m$ respectively

### Parameter Sharing:

- Since the above approach doesn't take advantage of "centralised learning", we can have the agents share common parameters by training a common Q-net. Speeds up learning because there's less parameters.
- $Q_u(o^a_t, m^{a'}_{t-1}, h^a_{t-1}, u^a_{t-1}, m^a_{t-1}, a, u^a_t)$ will be the common Q function. Likewise, another $Q_m(\cdot)$
- During decentralised execution, each agent uses a copy of the network.

## Differential Inter-Agent Learning:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%202.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%202.png)

- Motivation: If Centralised learning is allowed, why stop at just sending the communication action, instead send the gradients learnt through training directly
- Feedback is important when we consider communication. How does agent 1 know that its messages are appropriate?
- To quote the paper:

> *"Thus, while RIAL is end-to-end trainable **within** each agent, DIAL is end-to-end trainable
**across** agents"*

### How this works:

- C-Net:
    - This is a neural network that returns both the "environment action" and the "communication action"
    - In RIAL, both actions went through a common action-selector ($\epsilon$-greedy policy)
    - In DIAL, the environment action goes through the action-selector and the communication action goes through the Discretise/Regularise Unit (DRU)
- DRU:
    - During centralised learning, DRU$(m^a_t) = Sig(\mathcal{N}(m^a_t, \sigma))$ where $\sigma$ = std. dev. of noise added to communication. Performs regularisation.
    - During decentralised execution, DRU$(m^a_t) = \mathbb{1}   \{m^a_t > 0\}$. Performs discretisation
- Loss function:
    - For $Q_u$, the gradients are from the DQN loss of the C-net
    - For $Q_m$, the gradients are from the backprop of the error in message received by the other agent

## Architecture:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%203.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%203.png)

- Both architectures are the same for Q-net and C-net
- $z^a_t =$  MLP$(o^a_t)$ + MLP($m_{t-1}^a)$ + Lookup$(u^a_{t-1})$ + Lookup$(a)$
- Each function above returns 128 embeddings
- $z^a_t$ is processed through a 2-layer RNN with GRUs[128, 128]
- Final output passed through an MLP[128, 128, $|U| + |M|$] to give $(Q^a_t, m^a_t)$

## Switch Riddle:

> “One hundred prisoners have been newly ushered into prison. The warden tells them that starting tomorrow, each of them will be placed in an isolated cell, unable to communicate amongst each other. Each day, the warden will choose one of the prisoners uniformly at random with replacement, and place him in a central interrogation room containing only a light bulb with a toggle switch. The prisoner will be able to observe the current state of the light bulb. If he wishes, he can toggle the light bulb. He also has the option of announcing that he believes all prisoners have visited the interrogation room at some point in time. If this announcement is true, then all prisoners are set free, but if it is false, all prisoners are executed. The warden leaves and the prisoners huddle together to discuss their fate. Can they agree on a protocol that will guarantee their freedom?"

### Definitions:

- $u^a \in$ {"None", "Tell"}
- $m^a \in$  {"On", "Off"}  → One-bit message
- $r_t \neq 0$ only if any agent picks "Tell"
    - = +1 if all agents have been to the room atleast once
    - = -1 if not
- $o^a_t \in$ {0, 1} → Whether the agent is in the interrogation room or not

### Results:

- Compared with a no-communication, shared parameters baseline

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%204.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%204.png)

## MNIST Games:

### Multi-step MNIST:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%205.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%205.png)

- $u^a_t \in \{0, 1, 2, ..., 8, 9\}$
- $m^a_t = \{0, 1\}$ for decentralised execution and $= (0, 1)$ for centralised learning (in case of DIAL)
- Reward is 0.5 at the final timestep for each correct guess (max = 1)

### Colour-Digit MNIST:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%206.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%206.png)

- Slightly more complicated than the previous one. Agents have to play two games.
- Either learn colour or learn parity (odd/even). Parity earns 2x rewards

    ## $r(a) = 2(-1)^{u^a_2 + c^a+d^{a'}} + -1^{u^a_2 + c^{a'} + d^a}$

### Results:

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%207.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%207.png)

## Noise in Communication:

- Added noise to communication channel in DRU

![https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%208.png](https://github.com/SforAiDl/paper-reading-group/blob/master/Learning%20to%20Communicate%20with%20Deep%20Multi-Agent%20Reinforcement%20Learning/assets/Untitled%208.png)

## Further Research Areas:

- Has been made more complex and implemented on larger scale envs and more complex algos (like QMIX) have come which achieved pretty good results on the much more complicated StarCraft II
- More research in terms of what kind of communication is being done between agents. Can we try to understand how exactly the agents are communicating or can that be expressed in human terms.

# References:

- Overview of Communication in MARL: [https://arxiv.org/pdf/1911.05438v1.pdf](https://arxiv.org/pdf/1911.05438v1.pdf)
- Github repo of the project: [https://github.com/iassael/learning-to-communicate](https://github.com/iassael/learning-to-communicate)
- QMIX: [https://arxiv.org/abs/1803.11485v2](https://arxiv.org/abs/1803.11485v2)
- Cool paper on learning languages with MARL + NLP: [https://arxiv.org/pdf/1611.03218v4.pdf](https://arxiv.org/pdf/1611.03218v4.pdf)
- Reddit, Arxiv-Sanity
