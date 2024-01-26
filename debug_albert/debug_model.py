import os
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel, AlbertForMaskedLM

albert_xxlarge_v2_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "albert", "albert-xxlarge-v2")


def debug_tokenizer():
    """
    分词
    """
    tokenizer:AlbertTokenizer = AlbertTokenizer.from_pretrained(albert_xxlarge_v2_path)
    text = "Replace me by any text you'd like."
    tokens = tokenizer.tokenize(text)
    print("tokenized tokens :", tokens, sep='\n', end='\n\n')

    # 方式1
    token_ids = tokenizer.encode(tokens, add_special_tokens=False)
    print("token_ids :", token_ids, sep='\n', end='\n\n')

    # 方式2
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("token_ids :", token_ids, sep='\n', end='\n\n')

    


def debug_special_token():
    """
    调试 特殊 token
    """

    tokenizer:AlbertTokenizer = AlbertTokenizer.from_pretrained(albert_xxlarge_v2_path)
    text = "Replace me by any text you'd like."
    tokens = tokenizer.tokenize(text)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 返回一个加好特殊token的token序列。
    token_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=token_ids)
    print("token_ids :", token_ids, sep='\n', end='\n\n')

    # 获取特殊token数量，pair=False 表示单个句子
    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    print("num_special :", num_special, sep='\n', end='\n\n')


    ...

def debug_model_inputs():
    """
    模型输入
    """
    tokenizer:AlbertTokenizer = AlbertTokenizer.from_pretrained(albert_xxlarge_v2_path)
    text = "Replace me by any text you'd like."
    model_inputs = tokenizer(text, return_tensors='pt')

    print("model_inputs :", end='\n\n')
    for k, v in model_inputs.items():
        print(k, v, sep='\n', end='\n\n')

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
    # debug_model_inputs()
    ...