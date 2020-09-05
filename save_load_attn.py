import torch
from torchtext.data.metrics import bleu_score
import spacy


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint...')
    torch.save(state, filename)

def load_checkpoint(checkpoint,model,optimizer):
    print("=> Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def translate_sentence(model, sentence, german,english, device, max_length=50):
    spacy_ger = spacy.load('de')

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_idx = [german.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_idx).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, encoder_states,hidden,cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sen = [english.vocab.itos[idx] for idx in outputs]

    return translated_sen[1:]

def bleu(data, model, german, english, device):
    targets= []
    outputs= []

    for eg in data:
        src = vars(eg)['src']
        trg = vars(eg)['trg']

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1] # eos removed

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)
