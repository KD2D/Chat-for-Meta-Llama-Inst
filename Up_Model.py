# Этот модуль выполняет все главные штуки
# Например вывод текста или обновление кода во время работы программы


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import module


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_path = "D:/AIR" # Поменяйте на путь к вашему репозиторию с моделью
fine_tuned_path = "D:/AIR/fine-training"


# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    load_in_4bit=True)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)

# Для fine-traning
# model_training = PeftModel.from_pretrained(model, fine_tuned_path)

# перевести в eval режим (ускоряет и снижает память или нет)
# model.eval()



module.Start(model, tokenizer)


# Старый код можно удалить потом
# # Функция общения
# def chat(prompt, history=None, max_new_tokens=256):
#     #Что мы вводим
#     system_prompt = "You are a helpful, respectful, and honest assistant.\n"
#
#     full_prompt = f"<|system|>\n{system_prompt}</s>\n"
#
#     if history:
#         for user, assistant in history:
#             full_prompt += f"<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>\n"
#
#     full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
#
#     # Оброботка и вывод
#     inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         top_p=0.95,
#         temperature=0.7,
#         repetition_penalty=1.1
#     )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
#
#     # Сокращение вывода нейросети и отправка. Получим только (последний) ответ
#     answer = decoded.split("<|assistant|>\n")[-1].split("</s>")[0].strip()
#
#     return answer
#
#
# history = []
#
# while True:
#     user_input = input("Ты: ")
#     if user_input.lower() in ["exit", "quit", "выход"]:
#         break
#
#     response = chat(user_input, history)
#     print("ИИ:", response)
#     history.append((user_input, response))
