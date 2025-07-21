# Этот код запускает модель

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import module
import gc
import torch


gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # иногда помогает при залипании кеша
    torch.cuda.synchronize()  # синхронизация операций

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
cache_path = "D:/AIR" # Поменяйте на путь к вашему репозиторию с моделью
fine_tuned_path = "D:/AIR/fine-training"


# Конфигурация 4-bit квантизации
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # или "fp4" — nf4 чуть лучше
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Загрузка модели с 4-bit конфигом
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_path,
    quantization_config=bnb_config,
    device_map="auto" # Автоматически на GPU
)
# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)

# Для fine-traning
# model_training = PeftModel.from_pretrained(model, fine_tuned_path)

# перевести в eval режим (ускоряет и снижает память или нет)
model.eval()

# Запускаем дельше остальную работу
module.start(model, tokenizer)
