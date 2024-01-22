import torch
from torch import nn

torch.set_printoptions(linewidth=1000, precision=3, sci_mode=False)


def debug_model():

    bsz, seq_len, embed_dim = 2, 8, 12
    hidden_dim = 8
    num_layers = 3

    rnn = nn.LSTM(input_size=embed_dim,
                  hidden_size=hidden_dim,
                  num_layers=3,
                  batch_first=True,
                  dropout=0.0,
                  bidirectional=True)

    
    model_input = torch.randn(bsz, seq_len, embed_dim)
    h0 = torch.randn(num_layers * 2 if rnn.bidirectional else 1, bsz, hidden_dim)
    c0 = torch.randn(num_layers * 2 if rnn.bidirectional else 1, bsz, hidden_dim)
    output, (h_n, c_n) = rnn.forward(model_input, (h0, c0))

    print(f"output {list(output.shape)} =", output, sep='\n', end='\n\n') # [bsz, seq_len, embe]
    print(f"h_n {list(h_n.shape)} =", output, sep='\n', end='\n\n')
    print(f"c_n {list(c_n.shape)} =", output, sep='\n', end='\n\n')


if __name__ == "__main__":
    debug_model()
    ...
