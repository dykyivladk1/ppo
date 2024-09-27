import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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




def train_seq2seq(model, dataloader, n_epochs = 1, lr = 1e-4, voc_size = TGT_VOCAB_SIZE):
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    device = DEVICE
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)


    model.train()

    model = model.to(device)


    for epoch in tqdm(range(n_epochs), total = n_epochs, desc = 'Pretraining'):
        epoch_loss = 0.0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, voc_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Pretraining Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")





def generate_sequence(model, src, max_length=MAX_SEQ_LEN):

    model.eval()
    device = DEVICE
    src = src.to(device)
    src_mask = model.make_src_mask(src)
    enc_out = model.src_embed(src)
    enc_out = model.pos_encoder(enc_out)
    for layer in model.encoder_layers:
        enc_out = layer(enc_out, src_mask)

    generated = torch.zeros(src.size(0), 1, dtype=torch.long).to(device)  
    for _ in range(max_length):
        dec_out = model.tgt_embed(generated)
        dec_out = model.pos_decoder(dec_out)
        for layer in model.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, model.make_tgt_mask(generated))
        logits = model.fc_out(dec_out) 
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  
        generated = torch.cat((generated, next_token), dim=1)
    return generated  





def simulate_human_feedback(seq):
    reward = (seq == 42).float().mean().item()
    return reward 



def collect_reward_model_data(model, dataloader):
    device = DEVICE
    model.eval()
    sequences = []
    rewards = []
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc = 'Collecting data for reward model'):
            src = src.to(device)
            generated_seq = generate_sequence(model, src, max_length = MAX_SEQ_LEN)
            for seq in generated_seq:
                reward = simulate_human_feedback(seq)
                sequences.append(seq)
                rewards.append(reward)
    return sequences, rewards



def train_reward_model(reward_model, sequences, rewards, num_epochs=2, lr=1e-4, b_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    reward_model.train()
    device = DEVICE
    dataset = torch.utils.data.TensorDataset(torch.stack(sequences), torch.tensor(rewards))
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for seq_batch, reward_batch in tqdm(dataloader, desc=f"Training Reward Model Epoch {epoch+1}/{num_epochs}"):
            
            seq_batch = seq_batch.to(device) 
            reward_batch = reward_batch.to(device, dtype = torch.float32)
            optimizer.zero_grad()
            predicted_rewards = reward_model(seq_batch).float()  # [b]
            loss = criterion(predicted_rewards, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Reward Model Training Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")




