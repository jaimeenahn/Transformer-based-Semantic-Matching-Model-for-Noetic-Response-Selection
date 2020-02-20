from torch import nn
from transformers import BertModel

class BERTmodel(nn.Module) :
    def __init__(self, freeze_bert = True) :

        super(BERTmodel, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.bert_classifier = nn.Linear(768, 1)


    def forward(self, input, seq, attn_masks) :

        seq_repr, _ = self.bert_layer(input, token_type_ids = seq, attention_mask = attn_masks)

        cls_repr = seq_repr[:, 0]

        bert_logits = self.bert_classifier(cls_repr)

        return bert_logits
