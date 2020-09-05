import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from save_load_attn import save_checkpoint, load_checkpoint, translate_sentence, bleu

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True,
                init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de','.en'),
                                                        fields=(german, english))
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,
        bidirectional=True, dropout=p)

        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2,hidden_size)

    def forward(self,x):
        # x : input vector shape ( seq_len, bs )

        embedding = self.dropout(self.embedding(x))
        # embedding : ( seq_len, bs, embedding_size )

        encoder_states, (hidden,cell) = self.rnn(embedding)
        # encoder_states : specific to each timestamp

        hidden = self.fc_hidden(torch.cat((hidden[0:1],hidden[1:2]), dim=2))
        # hidden :(2, bs,hidden_size)
        cell = self.fc_cell(torch.cat((cell[0:1],cell[1:2]), dim=2))

        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(hidden_size*2+embedding_size,hidden_size,num_layers,dropout=p)
        self.energy = nn.Linear(hidden_size*3, 1)
        # Why hidden_size*3 ?: hidden_size*2 from encoder + hidden_size from prev timestamp(decoder)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size,output_size) # output_size: eng_vocab

    def forward(self, x, encoder_states, hidden, cell):
        # x: shape (bs) we need (1, bs)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding ; ( 1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length,1,1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states),dim=2)))
        attention= self.softmax(energy)
        # attention: (seq_len, bs, 1)

        attention = attention.permute(1,2,0)
        # attention: (bs, 1, seq_len)
        encoder_states = encoder_states.permute(1,0,2)
        # (bs , seq_len, hidden_size*2)

        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)
        # context_vector: (N, 1, hidden_size*2) -> (1,N,hidden_size*2)

        rnn_input = torch.cat((context_vector, embedding),dim=2)


        outputs, (hidden, cell) = self.rnn(rnn_input,(hidden,cell))
        # outputs :( 1, N, hidden_size)

        predictions = self.fc(outputs)
        # predictions : ( 1, N, length_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x,encoder_states,hidden,cell)

            outputs[t] = output

            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# training hyperparameters
num_epochs= 3
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model =True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1

enc_dropout = .0
dec_dropout = .0

writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data,validation_data,test_data),
                                                batch_size = batch_size, sort_within_batch=True,
                                                sort_key=lambda x: len(x.src), device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                    hidden_size,num_layers,enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                    hidden_size,output_size,num_layers,dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar",map_location=torch.device('cpu')),model,optimizer) # ,map_location=torch.device('cpu')
    print("Model Loaded..")

step = 0
train = False

if train:
    print('training start: ')
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch}/{num_epochs}]')

        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

        for batch_idx, batch in enumerate(train_iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(inp_data, target)
            # output: ( target_len, bs ,output_size)

            output = output[1:].reshape(-1, output.shape[2]) # first token is sos
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()

            writer.add_scalar('Training loss: ',loss,global_step =step)
            step+=1
sen = "Die Person ging ins Einkaufszentrum."
w = translate_sentence(model, sen, german,english, device, max_length=40)

print(w)
score = bleu(test_data,model,german,english,device)
print(score)
