"""A0: first-load probe of Qwen3.6-27B on the VM. Prints module paths, load time, one generation."""
import time, torch, sys
from transformers import AutoTokenizer, AutoConfig, Qwen3_5ForConditionalGeneration
P = "/data/models/Qwen3.6-27B"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(P)
model = Qwen3_5ForConditionalGeneration.from_pretrained(P, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print("load s", round(time.time() - t0, 1), "mem GB", round(torch.cuda.memory_allocated() / 1e9, 1))
print(type(model.model), [n for n, _ in model.model.named_children()])
lm = model.model.language_model
print("n layers", len(lm.layers))
for n, m in lm.layers[0].named_modules():
    if n.count(".") <= 1: print("L0:", n, type(m).__name__)
for n, m in lm.layers[3].named_modules():
    if n.count(".") == 0 and n: print("L3:", n, type(m).__name__)
print("down_proj weight", lm.layers[0].mlp.down_proj.weight.shape, lm.layers[0].mlp.down_proj.weight.dtype)
msgs = [{"role": "user", "content": "Who are you?"}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(s))
ids = tok(s, return_tensors="pt", add_special_tokens=False).to("cuda")
print("prompt tokens:", [tok.decode([i]) for i in ids.input_ids[0].tolist()])
t0 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=64, do_sample=True, temperature=1.0, top_p=0.95, top_k=20)
dt = time.time() - t0
new = out[0, ids.input_ids.shape[1]:]
print("gen", repr(tok.decode(new, skip_special_tokens=True)), "| tok/s", round(len(new) / dt, 1))
print("pad", tok.pad_token, tok.pad_token_id, "eos", tok.eos_token, tok.eos_token_id, "padding_side", tok.padding_side)
