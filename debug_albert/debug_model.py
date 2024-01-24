import os
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel, AlbertForMaskedLM

albert_xxlarge_v2_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "albert", "albert-xxlarge-v2")


def debug_tokenizer():
    print(albert_xxlarge_v2_path)
    tokenizer = AlbertTokenizer.from_pretrained(albert_xxlarge_v2_path)
    ...

def debug_fill_mask():
    """
    可以在本地电脑跑起来
    """
    from transformers import pipeline
    unmasker = pipeline('fill-mask', model=albert_xxlarge_v2_path)
    output = unmasker("Hello I'm a [MASK] model.")
    print("output =", end='\n\n')
    for o in output:
        print(o, end='\n\n')
    ...

if __name__ == "__main__":
    debug_tokenizer()
    # debug_fill_mask()
    ...