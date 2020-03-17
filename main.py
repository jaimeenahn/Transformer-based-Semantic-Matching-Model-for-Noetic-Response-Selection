from model import BERTmodel
import torch
import math
from torch import nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, BCELoss

import pickle

import os

#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#torch.cuda.set_device(0)
#print(torch.cuda.current_device())

with open('preprocessed_advising_both_small.pkl', 'rb') as f:
    input_data = pickle.load(f)

bert_train_data, xlnet_train_data = dict(), dict()
bert_val_data, xlnet_val_data = dict(), dict()

bert_train_data['tokens_tensors'] = torch.tensor(input_data['bert'][0]['input_ids'])
bert_train_data['segments_tensors'] = torch.tensor(input_data['bert'][0]['segment_ids'])
bert_train_data['attention_tensors'] = torch.tensor(input_data['bert'][0]['attention_masks'])
bert_train_data['label_tensors'] = torch.tensor(input_data['bert'][0]['labels'])

xlnet_train_data['tokens_tensors'] = torch.tensor(input_data['xlnet'][0]['input_ids'])
xlnet_train_data['segments_tensors'] = torch.tensor(input_data['xlnet'][0]['segment_ids'])
xlnet_train_data['attention_tensors'] = torch.tensor(input_data['xlnet'][0]['attention_masks'])
xlnet_train_data['label_tensors'] = torch.tensor(input_data['xlnet'][0]['labels'])

bert_val_data['tokens_tensors'] = torch.tensor(input_data['bert'][1]['input_ids'])
bert_val_data['segments_tensors'] = torch.tensor(input_data['bert'][1]['segment_ids'])
bert_val_data['attention_tensors'] = torch.tensor(input_data['bert'][1]['attention_masks'])
bert_val_data['label_tensors'] = torch.tensor(input_data['bert'][1]['labels'])

xlnet_val_data['tokens_tensors'] = torch.tensor(input_data['xlnet'][1]['input_ids'])
xlnet_val_data['segments_tensors'] = torch.tensor(input_data['xlnet'][1]['segment_ids'])
xlnet_val_data['attention_tensors'] = torch.tensor(input_data['xlnet'][1]['attention_masks'])
xlnet_val_data['label_tensors'] = torch.tensor(input_data['xlnet'][1]['labels'])

def data_reshape(data) :
    data_shape = data.shape
    return data.reshape(data_shape[0] * data_shape[1], data_shape[2])

def data_reshape_label(data) :
    data_shape = data.shape
    return data.reshape(data_shape[0] * data_shape[1])

#train_set = data_reshape(train_data['tokens_tensors']), data_reshape(train_data['segments_tensors']), data_reshape(train_data['attention_tensors']).long(), data_reshape_label(train_data['label_tensors'])
#val_set = data_reshape(val_data['tokens_tensors']), data_reshape(val_data['segments_tensors']), data_reshape(val_data['attention_tensors']).long(), data_reshape_label(val_data['label_tensors'])
train_set = bert_train_data['tokens_tensors'], bert_train_data['segments_tensors'], bert_train_data['attention_tensors'].long(), bert_train_data['label_tensors'], xlnet_train_data['tokens_tensors'], xlnet_train_data['segments_tensors'], xlnet_train_data['attention_tensors'].long(), xlnet_train_data['label_tensors']
val_set = bert_val_data['tokens_tensors'], bert_val_data['segments_tensors'], bert_val_data['attention_tensors'].long(), bert_val_data['label_tensors'], xlnet_val_data['tokens_tensors'], xlnet_val_data['segments_tensors'], xlnet_val_data['attention_tensors'].long(), xlnet_val_data['label_tensors']

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
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr = 5e-6)

if torch.cuda.is_available() :
    device = 'cuda'
else:
    device = 'cpu'

alpha = 0.7

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
def evaluate(model, criterion, val_set, device, alpha):

    losses, accuracies, recall = 0, 0, 0

    # Setting model to evaluation mode
    model.eval()
    bert_tokens, bert_segments, bert_attentions, bert_labels, xlnet_tokens, xlnet_segments, xlnet_attentions, xlnet_labels = val_set
    num_samples = len(bert_tokens)
    for i in range(num_samples) :
        # Move inputs and targets to device
        bert_ins, bert_seq, bert_attn_mask, bert_label = batch_generater(bert_tokens, bert_segments, bert_attentions, bert_labels, batch_size = 1, iteration = i)
        bert_ins, bert_seq, bert_attn_mask = bert_ins.to(device), bert_seq.to(device), bert_attn_mask.to(device)
        bert_inputs = bert_ins.squeeze(0), bert_seq.squeeze(0), bert_attn_mask.squeeze(0)
        xlnet_ins, xlnet_seq, xlnet_attn_mask, xlnet_label = batch_generater(xlnet_tokens, xlnet_segments, xlnet_attentions, xlnet_labels, batch_size = 1, iteration = i)
        xlnet_inputs = xlnet_ins, xlnet_seq, xlnet_attn_mask = xlnet_ins.to(device), xlnet_seq.to(device), xlnet_attn_mask.to(device)
        xlnet_inputs = xlnet_ins.squeeze(0), xlnet_seq.squeeze(0), xlnet_attn_mask.squeeze(0)

        bert_label, xlnet_label = bert_label.to(device), xlnet_label.to(device)
        #val_logits = model(input, seq, attn_mask)
        #val_loss = criterion(val_logits.squeeze(-1), label.float().view(-1))
        val_logits = model(bert_inputs, xlnet_inputs, alpha) # [num_batch, num_candidate=100, max_len = 120]
        #print(val_logits.shape, bert_labels.shape)
        val_loss = criterion(val_logits.squeeze(-1).unsqueeze(0), bert_label.float())
        losses += val_loss.item()
        val_loss.detach()

        # Calculate validation accuracy
        #accuracies += logits_accuracy(val_logits.data, bert_label)
        recall += top_recall(val_logits.data, bert_label)

    return losses / num_samples, accuracies / num_samples, recall / num_samples

def train(model, criterion, optimizer, train_set, device, epoch, print_every = 100, batch_size = 1, alpha = 0.7) :

    bert_tokens, bert_segments, bert_attentions, bert_labels, xlnet_tokens, xlnet_segments, xlnet_attentions, xlnet_labels = train_set
    num_samples = len(bert_tokens)
    print('\n========== Training start ==========')

    t1 = time()
    for i in range(int(math.ceil(num_samples/batch_size))) :
        optimizer.zero_grad()

        bert_ins, bert_seq, bert_attn_mask, bert_label = batch_generater(bert_tokens, bert_segments, bert_attentions, bert_labels, batch_size = batch_size, iteration = i)
        bert_ins, bert_seq, bert_attn_mask = bert_ins.to(device), bert_seq.to(device), bert_attn_mask.to(device)
        xlnet_ins, xlnet_seq, xlnet_attn_mask, xlnet_label = batch_generater(xlnet_tokens, xlnet_segments, xlnet_attentions, xlnet_labels, batch_size = batch_size, iteration = i)
        xlnet_ins, xlnet_seq, xlnet_attn_mask = xlnet_ins.to(device), xlnet_seq.to(device), xlnet_attn_mask.to(device)

        bert_label, xlnet_label = bert_labels.to(device), xlnet_labels.to(device)
        #loss = 0
        for batch in range(len(bert_ins)) : #len(inputs) == batch_size
            bert_inputs = bert_ins[batch], bert_seq[batch], bert_attn_mask[batch]
            xlnet_inputs = xlnet_ins[batch], xlnet_seq[batch], xlnet_attn_mask[batch]

            logits = model(bert_input = bert_inputs, xlnet_input = xlnet_inputs, alpha = alpha)
            #loss += criterion(logits.squeeze(1), bert_label[batch].float())
            #print(logits.squeeze(1), bert_label[batch].float())
            loss = criterion(logits.squeeze(1), bert_label[batch].float())
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()

        if (i+1) % print_every ==0 :
            print("Iteration {} ==== Loss {}".format(i+1, loss.item()))
        loss.detach()

    t2 = time()
    print('Time Taken for Epoch: {}'.format(t2-t1))


# starting training
#model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3])
model = torch.nn.DataParallel(model)
model.to(device)
epochs = 3

print('\n========== Validating ==========')
mean_val_loss, mean_val_acc, mean_val_recall = evaluate(model, criterion, val_set, device, alpha)
print("Validation Loss: {}\nValidation Accuracy: {}\nValidation Recall: {}".format(mean_val_loss, mean_val_acc, mean_val_recall))

model.train()
for epoch in range(epochs) :
    print('Epoch {}'.format(epoch))
    train(model, criterion, optimizer, train_set, device, epoch = epoch, print_every=10, alpha = alpha)

    print('\n========== Validating ==========')
    mean_val_loss, mean_val_acc, mean_val_recall = evaluate(model, criterion, val_set, device, alpha)
    print("Validation Loss: {}\nValidation Accuracy: {}\nValidation Recall: {}".format(mean_val_loss, mean_val_acc, mean_val_recall))
