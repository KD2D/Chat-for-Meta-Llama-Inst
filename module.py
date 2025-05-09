# Этот модуль выполняет все остальные штуки
# Например вывод текста или обновление кода во время работы программы
# When this code is useful to someone, I will probably translate it into English

import torch
import importlib
import sys
from transformers import TextStreamer

# Класс мы добавили класс для стриминга
class SavingTextStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.generated_text = ""  # сюда мы копим весь текст

    def on_text(self, text: str, **kwargs):
        print(text, end="", flush=True)  # печатаем сразу
        self.generated_text += text       # и параллельно копим текст



# Функция общения со стримингом
def chat_stream(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""):
    try:

        #Тут нам нужен system_prompt
        full_prompt = f"<|system|>\n{system_prompt}</s>\n"

        if history:
            for user, assistant in history:
                full_prompt += f"<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>\n"

        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        # Оброботка и вывод
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

        print("ИИ: ", end="")

        streamer = SavingTextStreamer(tokenizer) # отсылка на класс

        #with torch.no_grad():  # Генерация ответа без отслеживания градиентов (Tab)
        outputs = model.generate( # ТЫ стример Гарри (Зачем тут outputs?)
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.90, # Выбирает токины у которых вероятность записана тут
            temperature=0.7, # Чем выше тем ответы будут хаотичнее
            repetition_penalty=1.1,
            # pad_token_id=tokenizer.eos_token_id, Добавь что бы сломать GPT
            # skip_special_tokens=True # На случай если мне нужны спец-токены <|system|>, <|user|>, <|assistant|>
            streamer=streamer
        )
        # После генерации:
        answer = streamer.generated_text.strip()

        # Очистка неиспользуемой памяти
        torch.cuda.empty_cache()
        del inputs
        del outputs
        print("Вывод стримера", answer)
        return answer

    except Exception as e:
        print(f"Ошибка чата 1 {e}")



# Функция общения нормальная
def chat_normal(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""):
    try:

        #Промт дня "Ты говоришь как пират, с "Arrr!" в каждом предложении"

        #Что мы вводим в system_prompt будет тут
        full_prompt = f"<|system|>\n{system_prompt}</s>\n"

        if history:
            for user, assistant in history:
                full_prompt += f"<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>\n"

        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        # Оброботка и вывод
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

        #with torch.no_grad():  # Генерация ответа без отслеживания градиентов (Tab)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.90, # Выбирает токины у которых вероятность записана тут
            temperature=0.7,
            repetition_penalty=1.1,
            # pad_token_id=tokenizer.eos_token_id, Добавь что бы сломать GPT
            # skip_special_tokens=True # На случай если мне не нужны спец-токены <|system|>, <|user|>, <|assistant|>

        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Сокращение вывода нейросети и отправка. Получим только (последний) ответ
        answer = decoded.split("<|assistant|>\n")[-1].split("</s>")[0].strip()

        # Очистка неиспользуемой памяти
        torch.cuda.empty_cache()
        # Удаление ненужных переменных
        del inputs
        del outputs
        del decoded

        return answer

    except Exception as e:
        print(f"Ошибка чата 0 {e}")


# Перезагружаем не забудь нажать ctrl + s
def RestartKey(model, tokenizer, history):
    try:
        # Модуль_re = sys.modules[__name__] # Вроде бы не ннада

        print("Перезапуск!")
        importlib.reload(sys.modules[__name__])  # <-- файл (модуль.py) перезагружает себя
        return Start(model, tokenizer, history)# Востанавливаем ход
    except Exception as e:
        print(f"Ошибка перезагрузки {e}")



def Start(model, tokenizer, history=None): # Запускаем
    try:
        # Это будет вводится как основа для модели
        system_prompt = """Ты — Тим, добрый ИИ помошник.\n"""


        stream = input("Стриминг, нет? 1 или 0 (0 defolt): ")

        if history is None: # история есть?
            history = []

        while True:
            print(f"Ранее: {history}")  # можно потом убрать

            user_input = input("Ты: ") # Ввод пользователя
            if user_input.lower() in ["re", "restart", "reload"]: # Перезагружаем модуль? Не?
                return RestartKey(model, tokenizer, history)
                # continue # Это пока побудет тут

            elif user_input.lower() in ["exit", "quit", "выход", "покинуть"]: # Alt + F4 (как это этот код)
                break

            else: # Ну значит это нейронке + узнаём какой метод общения используем

                if (stream == "1"):
                    response = chat_stream(model, tokenizer, prompt=user_input, history=history, system_prompt=system_prompt)
                    # -
                else:
                    response = chat_normal(model, tokenizer, prompt=user_input, history=history, system_prompt=system_prompt)
                    print("ИИ:", response)
                history.append((user_input, response))

    except Exception as e:
        print(f"Ошибка главного цикал 0 {e}")

