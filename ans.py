from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载保存的模型和 tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("base_model")
tokenizer = AutoTokenizer.from_pretrained("base_model")

# 定义回答函数
def answer_question(question, context):
    inputs = f"answer: {question} context: {context}"
    inputs = tokenizer(inputs, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # 生成答案
    outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# 使用模型回答问题
for item in data:
    question = item["问题"]
    context = item["条款"]
    answer = answer_question(question, context)
    print(f"问题: {question}\n答案: {answer}\n")
