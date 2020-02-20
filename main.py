from model import BERTmodel
import torch
import math
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import pickle

import os

#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#torch.cuda.set_device(0)
#print(torch.cuda.current_device())

with open('preprocessed_advising.pkl', 'rb') as f:
    train_data, val_data = pickle.load(f)

train_data['tokens_tensors'] = torch.tensor(train_data['input_ids'])
train_data['segments_tensors'] = torch.tensor(train_data['segment_ids'])
train_data['attention_tensors'] = torch.tensor(train_data['attention_masks'])
train_data['label_tensors'] = torch.tensor(train_data['labels'])

val_data['tokens_tensors'] = torch.tensor(val_data['input_ids'])
val_data['segments_tensors'] = torch.tensor(val_data['segment_ids'])
val_data['attention_tensors'] = torch.tensor(val_data['attention_masks'])
val_data['label_tensors'] = torch.tensor(val_data['labels'])

def data_reshape(data) :
    data_shape = data.shape
    return data.reshape(data_shape[0] * data_shape[1], data_shape[2])

def data_reshape_label(data) :
    data_shape = data.shape
    return data.reshape(data_shape[0] * data_shape[1])

#train_set = data_reshape(train_data['tokens_tensors']), data_reshape(train_data['segments_tensors']), data_reshape(train_data['attention_tensors']).long(), data_reshape_label(train_data['label_tensors'])
#val_set = data_reshape(val_data['tokens_tensors']), data_reshape(val_data['segments_tensors']), data_reshape(val_data['attention_tensors']).long(), data_reshape_label(val_data['label_tensors'])
train_set = train_data['tokens_tensors'], train_data['segments_tensors'], train_data['attention_tensors'].long(), train_data['label_tensors']
val_set = val_data['tokens_tensors'], val_data['segments_tensors'], val_data['attention_tensors'].long(), val_data['label_tensors']
def batch_generater(input, seg, attn, label, batch_size, iteration) :
    if (iteration+1) * batch_size > len(input) :
        tokens = input[batch_size * iteration : ]
        segments = seg[batch_size * iteration : ]
        attentions = attn[batch_size * iteration : ]
        labels = label[batch_size * iteration : ]
    else :
        tokens = input[batch_size * iteration : batch_size * (iteration + 1)]
        segments = seg[batch_size * iteration : batch_size * (iteration + 1)]
        attentions = attn[batch_size * iteration : batch_size * (iteration + 1)]
        labels = label[batch_size * iteration : batch_size * (iteration + 1)]

    return tokens, segments, attentions, labels

model = BERTmodel()
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = 2e-5)

if torch.cuda.is_available() :
    device = 'cuda'
else:
    device = 'cpu'

from time import time

def top_recall(logits, labels, K=10) :
    labels = labels.squeeze(0)
    sorted_logits = list()
    for idx, logit in enumerate(logits) :
        sorted_logits.append((idx, logit))
    #print(sorted_logits[0])
    sorted_logits.sort(key=lambda element:element[1], reverse = True)
    for idx, logits in sorted_logits :
        if labels[idx] == 1 :
            return 1
        else :
            for l in labels :
                if l == 1 :
                    return 0
            if sorted_logits[0][1] < 0.5 :
                return 1
            else :
                return 0


def logits_accuracy(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    preds = (probs > 0.5).long()
    acc = (preds.squeeze() == labels).float().mean()
    return acc

# Defining an evaluation function for training
def evaluate(model, criterion, val_set, device):

    losses, accuracies, recall = 0, 0, 0

    # Setting model to evaluation mode
    model.eval()
    tokens, segments, attentions, labels = val_set
    num_samples = len(tokens)
    for i in range(num_samples) :
        # Move inputs and targets to device
        input, seq, attn_mask, label = batch_generater(tokens, segments, attentions, labels, batch_size = 1, iteration = i)
        input, seq, attn_mask, label = input.to(device), seq.to(device), attn_mask.to(device), label.to(device)
        #val_logits = model(input, seq, attn_mask)
        #val_loss = criterion(val_logits.squeeze(-1), label.float().view(-1))
        val_logits = model(input.squeeze(0), seq.squeeze(0), attn_mask.squeeze(0)) # [num_batch, num_candidate=100, max_len = 120]
        val_loss = criterion(val_logits.squeeze(-1).unsqueeze(0), label.float())
        losses += val_loss.item()

        # Calculate validation accuracy
        accuracies += logits_accuracy(val_logits, label)
        recall += top_recall(val_logits, label)

    return losses / num_samples, accuracies / num_samples, recall / num_samples

def train(model, criterion, optimizer, train_set, val_set, device, epochs=3, print_every = 100, batch_size = 32) :
    model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    model.to(device)

    model.train()

    tokens, segments, attentions, labels = train_set
    num_samples = len(tokens)
    print('\n========== Training start ==========')
    for epoch in range(epochs) :
        print('Epoch {}'.format(epoch))
        t1 = time()
        for i in range(int(math.ceil(num_samples/batch_size))) :
            optimizer.zero_grad()
            input, seq, attn_mask, label = batch_generater(tokens, segments, attentions, labels, batch_size = batch_size, iteration = i)
            input, seq, attn_mask, label = input.to(device), seq.to(device), attn_mask.to(device), label.to(device)
            for batch in range(len(input)) : #len(inputs) == batch_size
                #print(input[batch].unsqueeze(0).shape)
                #logits = model(input[batch].unsqueeze(0), seq[batch].unsqueeze(0), attn_mask[batch].unsqueeze(0))
                #loss = criterion(logits.squeeze(-1), label[batch].float().view(-1))
                logits = model(input[batch], seq[batch], attn_mask[batch])
                loss = criterion(logits.squeeze(-1), label[batch].float())

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 1)
            optimizer.step()

            if (i+1) % print_every ==0 :
                print("Iteration {} ==== Loss {}".format(i+1, loss.item()))

        t2 = time()
        print('Time Taken for Epoch: {}'.format(t2-t1))
        print('Time Taken for Epoch: {}'.format(t2-t1))
        print('\n========== Validating ==========')
        mean_val_loss, mean_val_acc, mean_val_recall = evaluate(model, criterion, val_set, device)
        print("Validation Loss: {}\nValidation Accuracy: {}\nValidation Recall: {}".format(mean_val_loss, mean_val_acc, mean_val_recall))

# starting training
train(model, criterion, optimizer, train_set, val_set, device, epochs=3, print_every=10)
