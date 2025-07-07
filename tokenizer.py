from transformers import AutoTokenizer, BertModel

model_dir = "/Users/prestonrank/LSTM_multi_perspective/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
bert_model = BertModel.from_pretrained(
    model_dir,
    from_tf=False,
    from_flax=False,
    trust_remote_code=False,
    use_safetensors=False  # this is the critical flag
)

# had an issue with circular imports, so i could put everything into a separate file 
