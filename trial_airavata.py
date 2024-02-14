import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


def create_prompt_with_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                    message["role"]
                )
            )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def inference(input_prompts, model, tokenizer):
    input_prompts = [
        create_prompt_with_chat_format([{"role": "user", "content": input_prompt}], add_bos=False)
        for input_prompt in input_prompts
    ]

    encodings = tokenizer(input_prompts, padding=True, return_tensors="pt")
    encodings = encodings.to(device)

    with torch.inference_mode():
        outputs = model.generate(encodings.input_ids, do_sample=False, max_new_tokens=250)

    output_texts = tokenizer.batch_decode(outputs.detach(), skip_special_tokens=True)

    input_prompts = [
        tokenizer.decode(tokenizer.encode(input_prompt), skip_special_tokens=True) for input_prompt in input_prompts
    ]
    output_texts = [output_text[len(input_prompt) :] for input_prompt, output_text in zip(input_prompts, output_texts)]
    return output_texts


model_name = "ai4bharat/Airavata"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

input_prompts = [
    "Transalte below sentences to english: ",
    "मैं अपने समय प्रबंधन कौशल को कैसे सुधार सकता हूँ? मुझे पांच बिंदु बताएं।",
    "मैं अपने समय प्रबंधन कौशल को कैसे सुधार सकता हूँ? मुझे पांच बिंदु बताएं और उनका वर्णन करें।",
]
outputs = inference(input_prompts, model, tokenizer)
print(outputs)

