from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART-XXEN", do_lower_case=False, use_fast=False, keep_accents=True)

# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART-XXEN", do_lower_case=False, use_fast=False, keep_accents=True)

model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART-XXEN")

# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART-XXEN")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']

# First tokenize the input and outputs. The format below is how IndicBART-XXEN was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>". 

para = "भारत के स्वतंत्रता सेनानी और बापू के तौर पर अपनी पहचान बनाने वाले मोहनदास करमचंद गांधी का जन्म 2 अक्टूबर 1869 को गुजरात के पोरबंदर में हुआ था। उन्होंने अंग्रेज़ों की गुलामी से भारत को आज़ाद कराने के लिए अपना पूरा जीवन दे दिया था। आज़ादी के लिए उन्होंने चंपारण, खेड़ा, आंदोलन, आंदोलन और भारत छोड़ो आदि आंदोलन किए। </s><2hi>"



inp = tokenizer(para, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

out = tokenizer("<2en> I am a boy </s>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids

model_outputs=model(input_ids=inp, decoder_input_ids=out[:,0:-1], labels=out[:,1:])

# For loss
model_outputs.loss ## This is not label smoothed.

# For logits
model_outputs.logits

# For generation. Pardon the messiness. Note the decoder_start_token_id.

model.eval() # Set dropouts to zero

model_output=model.generate(inp, use_cache=True, num_beams=4, max_new_tokens = 1000, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))


# Decode to get output strings

decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(decoded_output) # I am a boy

