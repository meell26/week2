from transformers import pipeline
generator = pipeline("text-generation",model="distilgpt2")
result=generator(
    "what is the steps to make a tuna sandwich",
    max_length=80,
    num_return_sequences=1,
    temperature=0.5,
    top_k=100,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.2    
)
print(result[0]["generated_text"])