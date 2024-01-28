import os
from transformers import InputExample
from transformers import AlbertTokenizer

albert_xxlarge_v2_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "albert", "albert-xxlarge-v2")
tokenizer:AlbertTokenizer = AlbertTokenizer.from_pretrained(albert_xxlarge_v2_path)

def rte_data():
    """
    rte 任务描述：自然语言推断任务
    给模型一个前提（premise），然后问模型一个假设（hypothesis）是否成立。label 成立（entailment）不成立（not_entailment）
    """
    
    # \u2014 为中文横线—
    data = {
        "premise": "Even the most draconian proposal \u2014 to reinstate flight limits to ease the bottlenecks at Kennedy \u2014 might backfire, industry analysts say, because airlines would end up shifting flights to Newark Airport, which is already strained. \"Kennedy is the perfect example of putting 10 pounds in a 5-pound bag,\" said Darryl Jenkins, a longtime airline consultant. \"J.F.K. was never set up to be a hub for anybody; its been a gateway,\" he said. But in recent years it became a hub for JetBlue and Delta. These days, delays at Kennedy are so bad, he said, that \"it's backing up the whole country.\".",
        "hypothesis": "JFK airport is in New York.",
        "label": "not_entailment",
        "idx": 2363
    }

    # 方便调试跟踪
    debug_data = {
        "premise": "Even the most draconian proposal .",
        "hypothesis": "JFK airport is in New York.",
        "label": "not_entailment",
        "idx": 2363
    }

    return debug_data


def data_process():
    """
    
    """
    data = rte_data()

    examples = []
    example = InputExample(guid=f'train-{data["idx"]}', text_a=data["premise"], text_b=data["hypothesis"], label=data["label"])
    
    # print(example.to_json_string())

    # get_input_features
    # True:表示上下文，非模板中的提示词。False:表示模板中提示词。
    a_txt = [(example.text_a, True), ('Question:', False), (example.text_b, True), ("?", False), ("the", False), ("Answer:", False), ("[MASK]", False), (".", False)]

    a_token_ids= [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in a_txt if x] # 如果 x 不是空串


    tokens_a = [token_id for part, _ in a_token_ids for token_id in part]
    ...


if __name__ == "__main__":
    data_process()
    ...
