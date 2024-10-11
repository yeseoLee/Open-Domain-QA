import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel

class CNN_block(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(CNN_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.dropout(output)  # Dropout 추가
        output = x + self.relu(output)
        output = output.transpose(1, 2)
        output = self.layer_norm(output)
        output = output.transpose(1, 2)
        return output

class CNN_RobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # CNN 블록 수를 3개로 줄임
        self.cnn_blocks = nn.ModuleList([CNN_block(config.hidden_size) for _ in range(3)])
        
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = sequence_output.transpose(1, 2)

        for cnn_block in self.cnn_blocks:
            sequence_output = cnn_block(sequence_output)

        sequence_output = self.dropout(sequence_output.transpose(1, 2))
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
