# Usage
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')
model = LlamaForCausalLM.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base', torch_dtype=torch.bfloat16)
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")


prompt = "देवनागरी एक भारतीय लिपि है जिसमें अनेक भारतीय भाषाएँ तथा कई विदेशी भाषाएं लिखीं जाती हैं। यह बायें से दायें लिखी जाती है। इसकी पहचान एक क्षैतिज रेखा से है जिसे 'शिरिरेखा' कहते हैं। इस लिपि का प्रयोग वैदिक युग के पूर्व से ही होता आ रहा है । </s><2hi>" 
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_new_tokens = 512)
x = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(x)
