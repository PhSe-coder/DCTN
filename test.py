from transformers import BertTokenizer

tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
result = tokenizer.convert_ids_to_tokens(1037)
print(result)