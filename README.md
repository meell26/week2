# week2
In this project we implemented a simple text generation system using the hugging face transformers library and the pre trained model gpt2
techniques sauch as zero shots and few shots 
Tools:
python
hugging face transformers 
pre trained model:gpt2
Methodology:
we created a text generation pipline using hugging face :
Input prompt :"How to make a tuna sandwich "
zero shot propmt 
few shots prompt
The model generates a continuation of the given prompt based on learned patterns from training data
Parameters:
Tempreture
top_k
Top_p
Do_sample
Max_length
max new tokens 
Repetiton_penalty
Conclusion: This project demonstrates how pre trained language models can be used for text generation and how parameter tuning plays a key role in improving output quality
even with a lightweight model like gpt2 we can achive reasonable results by adjusting the right parameters this project demonstrates that prompt engineering plays a critical role in text generation