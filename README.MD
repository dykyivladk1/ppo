# PPO (Proximal Policy Optimization)

This repository contains a simplified implementation of PPO, specifically adapted for testing with a Seq2Seq Transformer model.

Proximal Policy Optimization (PPO) is a state-of-the-art reinforcement learning algorithm designed to optimize a policy by interacting with an environment to maximize cumulative rewards. Unlike traditional reinforcement learning methods that directly maximize expected returns, PPO focuses on stabilizing policy updates. This is achieved by using a clipped objective function that constrains policy changes, preventing drastic updates that could potentially destabilize learning.

### Core Concept of PPO
The key idea behind PPO is to balance the trade-off between exploration and exploitation by preventing large updates to the policy through a *clipping mechanism*. This ensures that new policy parameters do not deviate significantly from the previous policy, making the optimization process more reliable. This is particularly useful in complex environments, like language generation tasks in NLP, where overly aggressive policy updates can lead to suboptimal outcomes.

### Key Components
1. **Policy Agent**: 
   The policy agent, defined in the `PPOAgent` class, represents the decision-making model in reinforcement learning. It takes as input the source sequence (`src`) and outputs a sequence of predicted tokens. This model is based on a Transformer architecture and is fine-tuned to optimize the expected reward.

2. **Reward Model**: 
   The reward model evaluates how well a generated sequence aligns with the desired goal. It is trained to predict rewards based on generated sequences, guiding the policy toward better performance.

3. **Memory Buffer**: 
   PPO uses a `PPOMemory` class to store sequences, states, actions, log-probabilities, and rewards. This data is used during the training phase to calculate advantage estimates and optimize the policy through multiple epochs.

4. **Clipped Surrogate Objective Function**: 
   PPO employs a clipped objective function to restrict policy updates. This ensures that the new policy (`πθ`) is not significantly different from the old policy (`πθ_old`), reducing the risk of sudden performance drops.

   - Objective Function:
   \[
   L^{CLIP} = \min(\text{ratio} \cdot A, \, \text{clip}(\text{ratio}, 1 - \epsilon, 1 + \epsilon) \cdot A)
   \]
   Where:
   - `ratio` = \(\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)}\) is the ratio of the new and old policy probabilities.
   - `A` is the advantage estimate.
   - `clip` function restricts the ratio to the range [1 - `eps_clip`, 1 + `eps_clip`].

5. **Training and Fine-Tuning**: 
   The fine-tuning is performed using the `finetune_llm_PPO` function. During training, sequences are generated using the current policy, rewards are estimated using the reward model, and the policy is optimized to maximize the expected reward while minimizing the change in the policy distribution.

### Code Structure
- **`src/ppo.py`**: Contains the main PPO components (`PPOAgent`, `PPOMemory`) and the fine-tuning function `finetune_llm_PPO`.
- **`src/model.py`**: Includes the implementation of the Seq2Seq Transformer used as the policy network.
- **`src/dataset.py`**: Includes the implementation `Example Dataset` for dummy testing.
- **`src/utils.py`**: Helper functions for data generation and processing.
- **`src/train.py`**: Script to pretrain the Seq2Seq Transformer, train the reward model, and fine-tune the policy using PPO.


Advantages of Using PPO
* **Stability**: The clipped objective prevents large policy updates, making PPO more stable than traditional policy gradient methods.
* **Sample Efficiency**: By reusing experiences multiple times (k_epochs), PPO achieves higher sample efficiency compared to other methods like vanilla policy gradient.
* **Flexibility**: PPO is adaptable to various tasks, including complex sequence generation tasks in natural language processing.


This repository demonstrates how PPO can be used effectively to fine-tune a Seq2Seq Transformer model for sequence-to-sequence tasks, balancing between exploration and exploitation using reward-guided optimization.

