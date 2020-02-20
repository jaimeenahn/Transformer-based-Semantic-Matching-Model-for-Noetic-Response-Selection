from torch import nn
from transformers import BertModel, XLNetModel

class BERTmodel(nn.Module) :
    def __init__(self, freeze_bert = True) :

        super(BERTmodel, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.xlnet_layer = XLNetModel.from_pretrained('xlnet-base-cased')
        self.dropout = nn.Dropout(0.1)
        self.bert_classifier = nn.Linear(768, 1)
        self.xlnet_classifier = nn.Linear(768, 1)
        self.softmax_layer = nn.Sotfmax(dim=1)

    def forward(self, bert_input, xlnet_input, seq, attn_masks, alpha) :

        seq_repr, _ = self.bert_layer(input, token_type_ids = seq, attention_mask = attn_masks)

        cls_repr = seq_repr[:, 0]

        bert_logits = self.bert_classifier(cls_repr)

        xlnet_seq_repr, _ = self.xlnet_layer(input, token_type_ids = seq, attention_mask = attn_masks)

        xlnet_cls_repr = xlnet_seq_repr[:, -1]

        xlnet_logits = self.xlnet_classifier(xlnet_cls_repr)

        total_logits = alpha * softmax_layer(bert_logits) + (1-alpha) * softmax_layer(xlnet_logits)

        return total_logits
