from transformers import MistralConfig

from ..configuration_live import LiveConfigMixin


class LiveMistralConfig(MistralConfig, LiveConfigMixin):
    pass
