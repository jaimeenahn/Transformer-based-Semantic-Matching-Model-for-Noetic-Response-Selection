from torch import nn
from transformers import BertModel, XLNetModel

class BERTmodel(nn.Module) :
    def __init__(self, freeze_bert = True) :

        super(BERTmodel, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.bert_classifier = nn.Linear(768, 1)
        self.xlnet_layer = XLNetModel.from_pretrained('xlnet-base-cased')
        self.xlnet_classifier = nn.Linear(768, 1)

        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, bert_input, xlnet_input, alpha) :


        bert_tokens, bert_seq, bert_attn_masks = bert_input
        xlnet_tokens, xlnet_seq, xlnet_attn_masks = xlnet_input

        seq_repr, _ = self.bert_layer(bert_tokens, token_type_ids = bert_seq, attention_mask = bert_attn_masks)
        cls_repr = seq_repr[:, 0]
        cls_repr = self.dropout(cls_repr)
        bert_logits = self.bert_classifier(cls_repr)

        xlnet_seq_repr = self.xlnet_layer(xlnet_tokens, token_type_ids = xlnet_seq, attention_mask = xlnet_attn_masks)

        xlnet_cls_repr = xlnet_seq_repr[0][:, -1]
        xlnet_cls_repr = self.dropout(xlnet_cls_repr)
        xlnet_logits = self.xlnet_classifier(xlnet_cls_repr)

        total_logits = alpha * self.softmax_layer(bert_logits) + (1-alpha) * self.softmax_layer(xlnet_logits)

        return total_logits
