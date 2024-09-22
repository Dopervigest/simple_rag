import warnings
from dataset import *
from model import get_model, get_pipeline

def infer(user_input, doc_embeddings, pipeline, encoder_model, corpus_of_documents=None, print_results=False):
    
    relevant_documents = get_recomendation(user_input, encoder_model, doc_embeddings, corpus_of_documents, print_results)

    prompt = f"""
    You are an assistant that makes recommendations for activities.
    You answer in plain text in short sentences to help your users.
    The user says this: '{user_input}'
    These are potential activities for them: '{relevant_documents}'. You must make your answer based on the provided options.
    Provide the user with 3 recommended activities based on their query.
    Respond in very casual manner using simple words and no slang and do not include special symbols. 
    
    Answer: 
    """
    
    sequences = pipeline(
        prompt, 
        do_sample=True,
        top_k=20,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        max_length=400,
        truncation=True
    )
    return sequences


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    corpus_of_documents = load_documents()
    encoder_model = get_encoder()
    
    doc_embeddings = encode_docs(corpus_of_documents, encoder_model)
    
    model, tokenizer = get_model()
    pipeline = get_pipeline(model, tokenizer)
    
    user_input = 'start'
    print('The model is ready! Now you can enter your input.')
    print("To exit the program enter 'stop'.")
    while True:
        user_input = str(input('Input: '))
        if user_input == 'stop':
            break
        
        sequences = infer(user_input, doc_embeddings, pipeline, encoder_model, corpus_of_documents)
        print(f"{sequences[0]['generated_text'][sequences[0]['generated_text'].index('Answer:') :]}")
        