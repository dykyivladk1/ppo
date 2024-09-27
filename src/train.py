import torch
from torch.utils.data import DataLoader


from model import RewardModel, TransformerSeq2Seq
from dataset import ExampleDataset
from utils import train_seq2seq, train_reward_model, collect_reward_model_data
from ppo import PPOAgent, finetune_llm_PPO

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



dataset = ExampleDataset(num_samples = 100)

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)




model = TransformerSeq2Seq(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    num_encoder_layers=N_LAYERS,
    num_decoder_layers=N_LAYERS,
    n_head=N_HEADS,
    max_len=MAX_LEN,
    ff_hidden_mult=FF_HIDDEN_MULT,
    dropout=DROPOUT,
    tokenizer_pad_token_id=0
).to(DEVICE)



reward_model = RewardModel(voc_size = TGT_VOCAB_SIZE,
                           emb_size = EMBED_SIZE,
                           hidden_size = HIDDEN_SIZE,
                           n_layers = N_LAYERS,
                           pad_token_id = 0).to(DEVICE)




print('TRAINING SEQ2SEQ')
train_seq2seq(model, dataloader)



print('TRAINING REWARD MODEL...')

sequences, rewards = collect_reward_model_data(model, dataloader)
print('COLLECTING DATA FOR REWARD MODEL')


train_reward_model(reward_model, sequences, rewards, num_epochs = REWARD_MODEL_EPOCHS,
                   lr = LR, b_size = BATCH_SIZE)


ppo_agent = PPOAgent(
    policy_model = model,
    reward_model = reward_model,
    lr = LR,
    eps_clip = EPS_CLIP,
    gamma = GAMMA,
    k_epochs = 4,
    pad_token_id = 0,
    voc_size = TGT_VOCAB_SIZE
)








finetune_llm_PPO(ppo_agent, dataloader, num_updates = 3)
print('DONE')