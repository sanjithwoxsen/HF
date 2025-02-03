from transformers import PretrainedConfig

class FeelWiseConfig(PretrainedConfig):
    model_type = "FeelWiseEmotion"

    def __init__(self, d_model=256, max_len=500, input_vocab_size=50000, n_layers=1, n_head=8, d_ff=1024, num_classes=6, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.input_vocab_size = input_vocab_size
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.dropout = dropout