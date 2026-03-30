from transformers import pipeline
import re
import streamlit as st

generator = pipeline("text-generation",model="distilgpt2")
st.title("AI Bio Generator")
name=st.text_input("enter your Name ")
role=st.text_input("enter your Role ")
hobbies=st.text_input("enter your Hobbies ")
if st.button("Generator Bio"):
    prompt=f"write a professional short bio about {name}, who is a {role},who enjoys{hobbies} start with the name"
    result=generator(
        prompt,
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True
    )
    bio=result[0]["generated_text"].replace(prompt,"").strip()
    st.subheader("Generated Bio:")
    st.write(bio)
zero_shot_prompt="How to make a tuna sandwich:\n1."
few_shot_prompt=""" 
How to make a cheese sandwich:
1.Take bread
2.Add cheese
3.Close the sandwich
How to make a chicken sandwich:
1.Take bread
2.Add cooked chicken
3.Close the sandwich
How to make a tuna sandwich:
1.

   """
zero_result=generator(
       zero_shot_prompt,
       
       num_return_sequences=1,
       temperature=0.5,
       top_k=100,
       top_p=0.9,
       do_sample=True,
       repetition_penalty=1.2 ,  
       no_repeat_ngram_size=2,
       early_stopping=True
)
few_result=generator(
       few_shot_prompt,
       max_new_tokens=60,
       num_return_sequences=1,
       temperature=0.7,
       top_k=50,
       top_p=0.9,
       do_sample=True,
       repetition_penalty=1.3  , 
       eos_token_id=generator.tokenizer.eos_token_id
 )
def clean_and_format(text,max_sentences=3):
    sentences=re.split(r'(?<=[.!?])\s+',text)
    seen=set()
    unique=[]
    for s in sentences:
        s=s.strip()
        if s and s not in seen:
            unique.append(s)
            seen.add(s)
    limited=unique[:max_sentences]
    formatted =""
    for i ,s in enumerate(limited,1):

        formatted +=f"{i}.{s}\n"
    return formatted.strip()    
    
print("==zero_shot Output===")
print(clean_and_format(zero_result[0]["generated_text"]))
print("\n==few_shot Output===")
print(clean_and_format(few_result[0]["generated_text"])) 