# Этот модуль выполняет все остальные штуки
# • This module handles all other stuff
# Например вывод текста или обновление кода во время работы программы
# • For example: text output or live code updates


import re
import torch
import traceback
import json
import importlib
import sys
from threading import Thread
from transformers import TextIteratorStreamer


# Функция общения со стримингом • Streaming chat function
def chat_stream(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""): # Есть проблема с непрерывным выводом от нейросети • There's an issue with continuous output from the neural network

    """Эта функция выполняет последовательный вывод текста с помощью
    TextIteratorStreamer в transformers перед этим добавляя специальные символы для модели"""

    try:
        # Тут мы используем system_prompt • Here we use the system_prompt
        full_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        full_prompt += f"{system_prompt}<|eot_id|>"

        if history:
            for user, assistant in history:
                full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                full_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"


        full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Оброботка и вывод • Processing and output
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]
        print("ИИ: ", end="")

        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|system|>", "<|user|>", "</s>", "<|assistant|>"]) # Добавь сюда <|eot_id|> если хочешь посмотреть на баги :) • Add <|eot_id|> here if you want to debug some weird behavior :)

        streamer = TextIteratorStreamer(tokenizer) # отсылка на класс • Reference to the class

        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        thread.join()

        generated_text = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text

        # answer = generated_text.strip()
        # После генерации: • After generation:
        print()

        # Удалить служебный токен • Remove special tokens
        # -
        # Сокращение вывода нейросети и отправка. Получим только (последний) ответ • Shorten the model output and send only the (last) response

        # match = re.search(r"<\|assistant\|\>\n(.*?)</s>", generated_text, re.DOTALL)
        new_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Очистка неиспользуемой памяти • Clear unused memory
        torch.cuda.empty_cache()
        del generated_text
        del inputs
        return answer

    except Exception as e:
        print(f"Ошибка чата 1 {e}")
        handle_error(e)



# Функция общения нормальная • Normal communication function
def chat_normal(model, tokenizer, prompt="", history=None, max_new_tokens=512, system_prompt=""):

    """Эта функция выполняет полный вывод текста перед этим добавляя специальные символы для модели"""

    try:

        #Что мы вводим в system_prompt будет тут • This is where the system_prompt content goes
        full_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        full_prompt += f"{system_prompt}<|eot_id|>"

        if history:
            for user, assistant in history:
                full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                full_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"

        # Текущий вопрос пользователя
        full_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        # Метка для начала ответа ассистента
        full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Оброботка и вывод • Processing and output
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():  # Генерация ответа без отслеживания градиентов (Tab) • Generate response without tracking gradients
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.90, # Выбирает токены с этой вероятностью • Picks tokens based on this probability
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=128001
                # pad_token_id=tokenizer.eos_token_id, # Добавь что бы сломать GPT • Add this if you want to break GPT
                # skip_special_tokens=True # На случай если тебе не нужны спец-токены <|system|> и прочие • For cases when you don’t need special tokens like <|system|> etc.

        )

        # decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Сокращение вывода нейросети и отправка. Получим только (последний) ответ • Shorten the model output and send only the (last) response
        new_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Очистка неиспользуемой памяти • Clear unused memory
        torch.cuda.empty_cache()
        # Удаление ненужных переменных • Delete unnecessary variables
        del inputs, outputs

        return answer

    except Exception as e:
        print(f"Ошибка чата 0 {e}")
        handle_error(e)


def handle_error(e): # Просто выводит полную ошибку • Just prints the full error trace
    choice = input("Вывести полный traceback? (y/n): ")
    if choice.lower() == 'y':
        traceback.print_exc()



# Перезагружаем. Не забудь нажать ctrl + s заранее • Reload — don’t forget to press Ctrl + S first!
def RestartKey(model, tokenizer, history):
    try:
        # Модуль_re = sys.modules[__name__] # Вроде бы не нннада • This is probably not necessary

        print("Перезапуск!")
        importlib.reload(sys.modules[__name__])  # <-- файл (модуль.py) перезагружает себя • This file (module.py) reloads itself
        return start(model, tokenizer, history) # Восстанавливаем ход • Resume execution flow
    except Exception as e:
        print(f"Ошибка перезагрузки {e}")
        handle_error(e)



def save_in_file(file_path, text, response): # Сохраняет память • Save memory to file
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            data = [text, response]
            json.dump(data, file, ensure_ascii=False, indent=2)
            # file.write("\n")  # перенос строки • Newline

    except Exception as e:
        print(f"Ошибка сохранения файла {e}")
        handle_error(e)


def read_file(file_path): #Читает файл • Read from file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as e:
        print(f"Ошибка загрузки файла {e}")
        handle_error(e)
        return 


def start(model, tokenizer, history=None):  # Запускаем • Launch
    """
    Стартуем! • Let’s start!

    >>> (CTRL + S) 🠒 re
    для перезагрузки кода без перезагрузки модели • reload the code without reloading the model

    >>> exit/выход

    >>> del/стереть
    отчищает память • clears memory
    """
    try:
        # Промт дня "Ты говоришь как пират, с "Arrr!" в каждом предложении" • Prompt of the day: "You speak like a pirate, saying 'Arrr!' in every sentence"

        # Это будет вводится как основа для модели • This will be used as the base prompt for the model
        system_prompt = """Ты — Тим, добрый ИИ помошник. Луна - это не сыр\n"""
        file = "memory.json"

        question = input("Стриминг, нет? 1 или 0 (0 defolt): ")

        if history is None or "": # история есть? • Is there a history?
            history = []
            try:
                with open(file, "w", encoding="utf-8") as file:
                    json.dump("", file, ensure_ascii=False)
                history.append(read_file(file))
            except:
                print("Память не найдена")

        while True:
            print(f"Ранее: {history}")  # можно потом убрать • Can be removed later
            user_input = input("Ты: ") # Ввод пользователя • User input

            if user_input.lower() in ["re", "restart", "reload"]: # Перезагружаем модуль? Не? • Reload module? No?
                return RestartKey(model, tokenizer, history)
                # continue # Это пока побудет тут • This will stay here for now
            elif user_input.lower() in ["exit", "quit", "выход", "покинуть"]: # Как Alt + F4 • Like Alt + F4
                break
            elif user_input.lower() in ["стереть", "del"]: # Делай только в крайнем случае (отчищает память) • Only do this as a last resort (clears memory)
                # model.reset_cache()  # если нужно отчищать всё • if you want to clear everything
                history.clear()
                continue

            #Тут будет поиск • Search goes here

            else: # Ну значит это нейронке + узнаём какой метод общения используем • Okay, so this is for the model + figuring out which chat mode is used
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
