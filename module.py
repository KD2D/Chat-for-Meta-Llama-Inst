# Этот модуль выполняет все остальные штуки
# Например вывод текста или обновление кода во время работы программы
# When this code is useful to someone, I will probably translate it into English

import re
import torch
import traceback
import json
import importlib
import sys
from threading import Thread
from transformers import TextIteratorStreamer


# Функция общения со стримингом
def chat_stream(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""): # Проблема с непрерывным выводом от нейросети

    """Эта функция выполняет последовательный вывод текста с помощью
    TextIteratorStreamer в transformers перед этим добавляя специальные символы для модели"""

    try:
        #Тут нам нужен system_prompt
        full_prompt = f"<|system|>\n{system_prompt}</s>\n"

        if history:
            for user, assistant in history:
                full_prompt += f"<|user|>\n{user}</s>\n<|assistant|>\n{assistant}</s>\n"

        full_prompt += f"<|user|>\n{prompt}</s>\n<|assistant|>\n"

        # Обработка и вывод
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        print("ИИ: ", end="")

        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|system|>", "<|user|>", "</s>", "<|assistant|>"]) # Добавь сюда <|eot_id|> если хочешь посмотреть на баги :)

        streamer = TextIteratorStreamer(tokenizer) # отсылка на класс

        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        thread.join()
        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text

        # answer = generated_text.strip()
        # После генерации:
        print()

        # Удалить сдужебный токен

        # Сокращение вывода нейросети и отправка. Получим только (последний) ответ

        # match = re.search(r"<\|assistant\|\>\n(.*?)</s>", generated_text, re.DOTALL)
        answer = generated_text.split("<|assistant|>\n")[-1].split("</s>")[0].strip()

        # Очистка неиспользуемой памяти
        torch.cuda.empty_cache()
        del generated_text
        del inputs
        #del outputs
        # print("Вывод стримера", answer) # это проверка. Уже исправленно
        return answer

    except Exception as e:
        print(f"Ошибка чата 1 {e}")
        handle_error(e)



# Функция общения нормальная
def chat_normal(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""):

    """Эта функция выполняет полный вывод текста перед этим добавляя специальные символы для модели"""

    try:

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
        handle_error(e)


def handle_error(e): # Просто выводит полную ошибку
    choice = input("Вывести полный traceback? (y/n): ")
    if choice.lower() == 'y':
        traceback.print_exc()



# Перезагружаем не забудь нажать ctrl + s заранее
def RestartKey(model, tokenizer, history):
    try:
        # Модуль_re = sys.modules[__name__] # Вроде бы не нннада

        print("Перезапуск!")
        importlib.reload(sys.modules[__name__])  # <-- файл (модуль.py) перезагружает себя
        return start(model, tokenizer, history)# Восстанавливаем ход
    except Exception as e:
        print(f"Ошибка перезагрузки {e}")
        handle_error(e)



def save_in_file(file_path, text, response): # Сохраняет
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            data = [text, response]
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")  # перенос строки

    except Exception as e:
        print(f"Ошибка сохранения файла {e}")
        handle_error(e)


def read_file(file_path): #Читает
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        print(f"Ошибка загрузки файла {e}")
        handle_error(e)
        return ""


def start(model, tokenizer, history=None):  # Запускаем
    """
    Стартуем!

    >>> (CTRL + S) потом напиши re для перезагрузки кода без перезагрузки модели

    >>> exit/выход

    >>> del/стереть отчищает память
    """
    try:
        # Промт дня "Ты говоришь как пират, с "Arrr!" в каждом предложении"

        # Это будет вводится как основа для модели
        system_prompt = """Ты — Тим, добрый ИИ помошник. Луна - это не сыр\n"""

        file = "memory.json"
        question = input("Стриминг, нет? 1 или 0 (0 defolt): ")

        if history is None: # история есть?
            history = []
            # history.append(read_file(file))

        while True:
            print(f"Ранее: {history}")  # можно потом убрать
            user_input = input("Ты: ") # Ввод пользователя

            if user_input.lower() in ["re", "restart", "reload"]: # Перезагружаем модуль? Не?
                return RestartKey(model, tokenizer, history)
                # continue # Это пока побудет тут
            elif user_input.lower() in ["exit", "quit", "выход", "покинуть"]: # Alt + F4 и Ctrl + c(как это этот код)
                break
            elif user_input.lower() in ["стереть", "del"]: # Делай только в крайнем случае
                # model.reset_cache()  # зависит от еслинужно отчищать всё
                history.clear()
                continue

            else: # Ну значит это нейронке + узнаём какой метод общения используем
                if (question == "1"):
                    response = chat_stream(model, tokenizer, prompt=user_input, history=history, system_prompt=system_prompt)
                    # -
                else:
                    response = chat_normal(model, tokenizer, prompt=user_input, history=history, system_prompt=system_prompt)
                    print("ИИ:", response)
                history.append((user_input, response))
                save_in_file(file, user_input, response)


    except Exception as e:
        print(f"Ошибка главного цикла 0 {e}")
        handle_error(e)
        RestartKey(model, tokenizer, history)
