from transformers import LlamaTokenizer
from transformers import AutoTokenizer
llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct-HF')

print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)
print(llama_tokenizer.vocab_size)
print(llama_tokenizer.pad_token_id)
print(llama_tokenizer.pad_token)

new_tokens = llama_tokenizer.add_tokens(['<unk>'], special_tokens=True)
llama_tokenizer.pad_token = '<unk>'
llama_tokenizer.pad_token_id = llama_tokenizer.encode('<unk>')[0]
image_tokens = ['<img_{:05d}>'.format(i) for i in range(100)]

new_tokens = llama_tokenizer.add_tokens(image_tokens)
print(llama_tokenizer.vocab_size)
print(new_tokens)
new_tokens = llama_tokenizer.add_tokens(['<img>', '</img>', '<patch>', '</patch>'], special_tokens=True)
print(llama_tokenizer.vocab_size)
print(new_tokens)

location_tokens = ['<loc-{:d}>'.format(i) for i in range(224)]
new_tokens = llama_tokenizer.add_tokens(location_tokens)
print(new_tokens)
new_tokens = llama_tokenizer.add_tokens(['<box_start>', '<box_end>'])
print(new_tokens)
print(llama_tokenizer.encode('<box_end>'))

llama_tokenizer.save_pretrained(
    'pretrained/cvlm_llama3_tokenizer_100img_and_224loc_addpatch')

res = llama_tokenizer('<img><img_00001></img>')
print(res)
print(llama_tokenizer.decode(res['input_ids']))
print(llama_tokenizer.decode([128585]))