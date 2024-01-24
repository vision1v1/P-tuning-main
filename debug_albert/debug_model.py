import os

albert_xxlarge_v2_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "albert", "albert-xxlarge-v2")

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
    debug_fill_mask()
    ...