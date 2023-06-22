from transformers import PegasusForConditionalGeneration, PegasusTokenizer
def abstracttive_summary(meeting_note):
    # Load tokenizer 
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    # Load model 
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    # Create tokens - number representation of our text
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    #tokens
    # Summarize 
    summary = model.generate(**tokens)
    # Output summary tokens         
    #summary[0]
    # Decode summary
    output = tokenizer.decode(summary[0])
    return output
## test branch
##done with the changes 


