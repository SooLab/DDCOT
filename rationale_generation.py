from openai import OpenAI
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import re

COMPLETIONS_MODEL = "gpt-3.5-turbo"
temperature = 0.1
max_tokens = 512

client = OpenAI(api_key=YOUR_KEY)

context = "N/A"
has_image = "yes"
question = "Which nutrients are mainly provided by the foods in the image?"
option = [
    "Vitamins",
    "Protein",
    "Fats"
]
image_root = "image.png"

# # stage 1

negative_space_prompting = [
    {"role": "system", "content": "You are a helpful, highly intelligent guided assistant. You will do your best to guide humans in choosing the right answer to the question. Note that insufficient information to answer questions is common. The final answer should be one of the options. "},
    {"role": "user", "content": f"Given the context, questions and options, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions based on context, questions and options. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:\nSub-questions:\n 1. <sub-question 1>\n2. <sub-question 2>\n...\nSub-answers:\n1. <sub-answer 1> or 'Uncertain'\n2. <sub-answer 2> or 'Uncertain'\n...\nAnswer: <One of the options> or 'Uncertain'\n\nFor a question, assume that you do not have any information about the picture, but try to answer the sub-questions and prioritize whether your general knowledge can answer it, and then consider whether the context can help. If sub-questions can be answered, then answer in as short a sentence as possible. If sub-questions cannot be determined without information in images, please formulate corresponding sub-answer into \"Uncertain\". \nOnly use \"Uncertain\" as an answer if it appears in the sub-answers. All answers are expected as concise as possible. \nHere is an attempt:\nContext: {context} \nHas An Image: {has_image}\nQuestion: {question}\nOptions: {option}"},
    ]

response = client.chat.completions.create(
    model=COMPLETIONS_MODEL,
    messages=negative_space_prompting,
    max_tokens=max_tokens,
    temperature=temperature,
)
answer = response.choices[0].message
print(answer.content)

sub_question_pattern = re.compile(r"Sub-questions:(.*?)Sub-answers:", re.DOTALL)
sub_answer_pattern = re.compile(r"Sub-answers:(.*?)Answer:", re.DOTALL)

sub_questions_match = sub_question_pattern.search(answer.content)
sub_answers_match = sub_answer_pattern.search(answer.content)

sub_questions = [re.sub(r'^\d+\.\s*', '', sub.strip()) for sub in sub_questions_match.group(1).split('\n') if sub.strip()]
sub_answers = [re.sub(r'^\d+\.\s*', '', sub.strip()) for sub in sub_answers_match.group(1).split('\n') if sub.strip()]

print("Sub-questions:", sub_questions)
print("Sub-answers:", sub_answers)

# sub_questions = ['What foods are shown in the image?', 'What nutrients are typically found in those foods?']
# sub_answers = ['Uncertain', 'Uncertain']

# # stage 2
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", device_map=device, load_in_8bit=True) 

keywords = ["uncertain", "Uncertain", "insufficient", "Insufficient", "cannot be determined", "not provide", "not possible"]

preliminary_knowledge = ""
for sub_q, sub_a in zip(sub_questions, sub_answers):
    reform_sub_q = f"Question: {sub_q} Answer:"
    if any(keyword in sub_a for keyword in keywords):
        image = Image.open(image_root).convert('RGB')
        inputs = processor(image, text=reform_sub_q, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        preliminary_knowledge += reform_sub_q + generated_text + "\n"
    else:
        preliminary_knowledge += reform_sub_q + sub_a + "\n"

print("preliminary_knowledge: ", preliminary_knowledge)
# preliminary_knowledge = "Question: What foods are shown in the image? Answer:Oranges\nQuestion:what nutrients are commonly found in the foods in the image? Answer:The oranges are a good source of vitamin C"

integration = [
    {"role": "system", "content": "You are a helpful, highly intelligent teacher. You will not only do your best to guide humans to the correct answer, but you will also give the rationales as a reference. "},
    {"role": "user", "content": f"Given the context, questions, options, preliminary knowledge, think step by step and answer the questions. Please note that we need not only the answer, but more importantly the rationales of getting the answer. The expected answering form is as follows:\nRationale: <rationale>\nAnswers: <one of the options>\n\nPlease note that the preliminary knowledge given may not always be valid. Please select valid information to form the rationale and choose the relatively correct option as your answer. \nHere is an attempt:\nContext: {context} \nHas An Image: {has_image}\nQuestion: {question}\nOptions: {option}\nPreliminary knowledge: \n{preliminary_knowledge}"},
]

response = client.chat.completions.create(
    model=COMPLETIONS_MODEL,
    messages=integration,
    max_tokens=max_tokens,
    temperature=temperature,
)
answer = response.choices[0].message
print("final answer: ", answer.content)