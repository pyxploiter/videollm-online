import torch
from torch import nn
from transformers import MistralForCausalLM, Cache
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_mistral import LiveMistralConfig
from ..modeling_live import build_live, LiveMixin


logger = logging.get_logger(__name__)


class LiveMistralForCausalLM(MistralForCausalLM, LiveMixin):
    config_class = LiveMistralConfig

    def __init__(self, config: LiveMistralConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs

        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)


def build_live_mistral(**kwargs):
    return build_live(config_class=LiveMistralConfig, model_class=LiveMistralForCausalLM, **kwargs)
