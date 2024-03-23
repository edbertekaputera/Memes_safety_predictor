import fasttext
from transformers import M2M100Tokenizer

def identify_language(text:str, model) -> str:
    """
    Detect which langauge is the text
    Base model:
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    - en : english
    - ta : tamil
    - zh : mandarin
    - ms : malay
    - id : bahasa
    Example:
    import fasttext
    model = fasttext.load_model("lid.176.bin")
    identify_language("ni hao", model)
    """
    try:
        predictions = model.predict(text)
        predicted_language = predictions[0][0].split("__label__")[1]
        return predicted_language
    except ValueError as e:
        print(e)
        return "NA"
    
def translate(model, text:str, source_lang:str, device:str="mps"):
    """
    Translate a given text
    Example:
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to("mps")
    translate(model, "你好，你好吗？", "zh")
    - en : english
    - ta : tamil
    - zh : mandarin
    - ms : malay
    - id : bahasa
    """
    # Load Model
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    tokenizer.src_lang = source_lang

    # Translating
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en"))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]