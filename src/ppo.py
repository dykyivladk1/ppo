import torch
import torch.nn as nn
from model import TransformerSeq2Seq
from utils import generate_sequence
from tqdm import tqdm

SRC_VOCAB_SIZE = 1000  
TGT_VOCAB_SIZE = 1000 

EMBED_SIZE = 512
HIDDEN_SIZE = 512
N_LAYERS = 6
N_HEADS = 8
MAX_LEN = 512
FF_HIDDEN_MULT = 4
DROPOUT = 0.1

LR = 1e-4
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 5
REWARD_MODEL_EPOCHS = 3
PPO_UPDATES = 3
EPS_CLIP = 0.2
GAMMA = 0.99
MAX_SEQ_LEN = 50 


DEVICE = torch.device('cuda' if torch.cuda.is_available() else torch.device('mps'))





class PPOAgent:
    def __init__(self, policy_model, reward_model,
                 lr=1e-4, eps_clip=0.2,
                 gamma=0.99, k_epochs=3,
                 pad_token_id=0, voc_size=TGT_VOCAB_SIZE):
        self.policy = policy_model
        self.policy_old = TransformerSeq2Seq(
            src_vocab_size=SRC_VOCAB_SIZE,
            tgt_vocab_size=TGT_VOCAB_SIZE,
            embed_size=EMBED_SIZE,
            num_encoder_layers=N_LAYERS,
            num_decoder_layers=N_LAYERS,
            n_head=N_HEADS,
            max_len=MAX_LEN,
            ff_hidden_mult=FF_HIDDEN_MULT,
            dropout=DROPOUT,
            tokenizer_pad_token_id=pad_token_id
        ).to(DEVICE)
        self.policy_old.load_state_dict(policy_model.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.reward_model = reward_model
        self.mse_loss = nn.MSELoss()
        self.voc_size = voc_size
        self.pad_token_id = pad_token_id

    def generate(self, src, max_len=MAX_SEQ_LEN):
        self.policy.eval()
        device = DEVICE
        src = src.to(device)
        with torch.no_grad():
            generated_seq = generate_sequence(self.policy, src, max_length=max_len)
        return generated_seq  

    def update(self, memory):
        old_srcs = torch.cat(memory.srcs, dim=0).to(DEVICE)       
        old_states = torch.cat(memory.states, dim=0).to(DEVICE)     
        old_actions = torch.cat(memory.actions, dim=0).to(DEVICE)   
        old_logprobs = torch.cat(memory.logprobs, dim=0).to(DEVICE)  
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(DEVICE)  

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        advantages = rewards 

        for epoch in range(self.k_epochs):  # Iterate through epochs
            logits = self.policy(old_srcs, old_states)  
            logprobs = nn.functional.log_softmax(logits, dim=-1) 
            action_logprobs = logprobs.gather(2, old_actions.unsqueeze(-1)).squeeze(-1) 

            action_logprobs = action_logprobs.view(-1)  
            old_logprobs = old_logprobs.view(-1)        
            ratios = torch.exp(action_logprobs - old_logprobs.detach())  

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()

            print(f"Epoch {epoch + 1}/{self.k_epochs}, Loss: {loss.item()}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())



class PPOMemory:
    def __init__(self):
        self.states = []      
        self.actions = []    
        self.logprobs = []    
        self.rewards = []    
        self.srcs = []      

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.srcs = []





def finetune_llm_PPO(agent, dataloader, num_updates=1):
    memory = PPOMemory()
    for update in range(num_updates):
        for src, _ in tqdm(dataloader, desc=f"PPO Update {update+1}/{num_updates}"):
            src = src.to(DEVICE)
            generated_seq = agent.generate(src)  

            predicted_rewards = agent.reward_model(generated_seq)  

            memory.srcs.append(src.cpu())

            states = generated_seq[:, :-1] 
            actions = generated_seq[:, 1:]  
            memory.states.append(states.cpu())
            memory.actions.append(actions.cpu())

            with torch.no_grad():
                logits = agent.policy_old(src, states.to(DEVICE))  
                logprobs = nn.functional.log_softmax(logits, dim=-1) 
                action_logprobs = logprobs.gather(2, actions.unsqueeze(-1)).squeeze(-1) 
                memory.logprobs.append(action_logprobs.cpu())


            batch_rewards = predicted_rewards.detach().cpu().numpy() 
            seq_len = actions.size(1)
            for reward in batch_rewards:
                memory.rewards.extend([reward] * seq_len)

        agent.update(memory)

        memory.clear_memory()
        print(f"PPO Update {update+1}/{num_updates} completed.")




