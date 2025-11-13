from transformers import Qwen2Config

from ..configuration_live import LiveConfigMixin


class LiveQwenConfig(Qwen2Config, LiveConfigMixin):
    pass
