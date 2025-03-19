import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load model and tokenizer
model_name = "facebook/opt-350m"  # Change this to any other model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define function for text generation
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=50,    # Restrict length
            do_sample=True,   # Make the output less robotic
            top_p=0.9         # Controls diversity
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Gradio UI
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
iface.launch()
