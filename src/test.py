from multilingual_clip import pt_multilingual_clip
import transformers

texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'

# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

model.to(0)


embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)