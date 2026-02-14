# Этот код запускает модель
# • This code launches the model


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import module
import gc
import torch


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # иногда помогает при залипании CUDA кеша • Sometimes helps when CUDA cache gets stuck
    torch.cuda.synchronize()  # синхронизация операций • Synchronize operations

# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Meta-Llama-3-8B-Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_path = "D:/AIR" # Поменяйте на путь к вашему репозиторию с моделью • Replace with the path to your model repository
fine_tuned_path = "D:/AIR/fine-training"


# Конфигурация 4-bit квантизации
# • 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # или "fp4", но nf4 чуть лучше • Or use "fp4" but nf4 performs slightly better
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


# Загрузка модели с 4-bit конфигом
# • Load the model with 4-bit config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_path,
    quantization_config=bnb_config,
    offload_buffers=True,
    device_map="auto" # Автоматически на GPU • Automatically on GPU
    torch_dtype=torch.float16,  # Загрузка быстрее чем с float32
    offload_folder="D:/offload",  # отгружает часть данных на CPU
    # attn_implementation="flash_attention_2", # Кротое внимание. Нужна сборка
    attn_implementation="sdpa" # Быстрое внимание, но чуть хуже
)
#
# Загрузка токенизатора • Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)


# Используйте для fine-traning
# • Use for fine-tuning
# model_training = PeftModel.from_pretrained(model, fine_tuned_path)

# перевести в eval режим (ускоряет и снижает память или нет)
# • Switch to eval mode (may improve speed and reduce memory usage — or not)
model.eval()

# Запускаем дальше остальную работу
# • Now we run the rest of the process
module.start(model, tokenizer)
