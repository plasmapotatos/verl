from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(tok.tokenize("<DATE>"))
print(tok.tokenize("</DATE>"))
print(tok.tokenize("[DATE]"))
print(tok.tokenize("[/DATE]"))