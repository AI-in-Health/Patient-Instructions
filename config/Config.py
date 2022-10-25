from transformers.models.t5.configuration_t5 import T5Config


class DSConfig(T5Config):
    model_type = "Discharge_Summary_Model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
