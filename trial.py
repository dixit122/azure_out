from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import AlbertTokenizer, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("sarvamai/OpenHathi-7B-Hi-v0.1-Base", do_lower_case=False, use_fast=False, keep_accents=True)
# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("sarvamai/OpenHathi-7B-Hi-v0.1-Base")
# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']
# First tokenize the input. The format below is how IndicBART was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. Similarly, the output should be "<2yy> Sentence </s>".

para = "can you translate this to english </s><2en> :  देवनागरी एक भारतीय लिपि है जिसमें अनेक भारतीय भाषाएँ तथा कई विदेशी भाषाएं लिखीं जाती हैं। यह बायें से दायें लिखी जाती है। इसकी पहचान एक क्षैतिज रेखा से है जिसे 'शिरिरेखा' कहते हैं। इस लिपि का प्रयोग वैदिक युग के पूर्व से ही होता आ रहा है । </s><2hi>"



inp = tokenizer(para,id_special_tokens=False, return_tensors="pt", padding=True).input_ids 

# For generation. Pardon the messiness. Note the decoder_start_token_id.

model_output=model.generate(inp,max_new_tokens = 512, use_cache=True,no_repeat_ngram_size=3,encoder_no_repeat_ngram_size=3, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2en>"))
# Decode to get output strings
decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded_output) #दिल्ली यूनिवर्सिटी भारत की सबसे बड़ी यूनिवर्सिटी है।

