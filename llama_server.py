import json
import os
import psutil
import gc
from typing import List, Union, Dict
from threading import Thread
import random
import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import torch
from transformers import set_seed
from transformers import LlamaTokenizer
from transformers import TextIteratorStreamer

import qlinear
from utils import Utils


def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained("./llama-2-wts-hf/7B_chat")
    ckpt = "pytorch_llama27b_w_bit_{}_awq{}_{}amd.pt".format(args.w_bit, "_fa" if args.flash_attention else "",
                                                             "lm_" if args.lm_head else "")
    print(f"Loading from ckpt: {ckpt}")
    if not os.path.exists(ckpt):
        print(
            f"\n\n *** Run run_awq.py --task quantize first to save quantized model! *** \n\n")
        raise SystemExit
    model = torch.load(ckpt)
    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer


def create_app(model, tokenizer):
    app = FastAPI()

    class Options(BaseModel):
        seed: int = 0
        num_predict: int = 100

    class GenerateRequest(BaseModel):
        model: str
        prompt: Union[List[str], str]
        format: str = None
        options: Options = Options()
        stream: bool = False

    class ChatRequest(BaseModel):
        model: str
        messages: List[Dict[str, str]]
        format: str = None
        options: Options = Options()
        stream: bool = False

    @app.post("/api/generate")
    def generate(gen_args: GenerateRequest):
        print("Generate request")
        inputs = tokenizer(gen_args.prompt, return_tensors="pt")
        input_ids_ = inputs.input_ids
        input_length = input_ids_.shape[1]
        attention_mask = inputs.attention_mask
        if gen_args.options.seed == 0:
            set_seed(random.randint(1, 1000000))
        else:
            set_seed(gen_args.options.seed)
        generate_ids = model.generate(input_ids_,
                                      attention_mask=attention_mask,
                                      max_new_tokens=gen_args.options.num_predict)
        response = tokenizer.batch_decode(generate_ids[:, input_length:],
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return {"response": response}

    @app.post("/api/chat")
    def chat(chat_args: ChatRequest):
        print("Chat request")
        if chat_args.options.seed == 0:
            set_seed(random.randint(1, 1000000))
        else:
            set_seed(chat_args.options.seed)
        if chat_args.stream is True:
            streamer = TextIteratorStreamer(
                tokenizer=tokenizer,
                timeout=60.0,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            inputs = tokenizer.apply_chat_template(chat_args.messages, return_tensors="pt")
            kwargs = dict(
                inputs=inputs,
                max_new_tokens=chat_args.options.num_predict,
                streamer=streamer
            )
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()

            def data_stream():
                for new_text in streamer:
                    yield json.dumps({"message": {"role": "assistant", "content": new_text}}) + '\n'

            return StreamingResponse(data_stream(), media_type="application/x-ndjson")
        else:
            inputs = tokenizer.apply_chat_template(chat_args.messages, return_tensors="pt")
            input_length = inputs.shape[1]
            generate_ids = model.generate(
                inputs,
                max_new_tokens=chat_args.options.num_predict
            )
            response = tokenizer.batch_decode(
                generate_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            return {"message": {"role": "assistant", "content": response}}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--host", help="IP address of the server. default=127.0.0.1", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="PORT number of the server. default=3000", type=int, default=3000)

    parser.add_argument('--w_bit', help="Quantized bit size. default=3", type=int, default=3, choices=[3, 4])
    parser.add_argument('--flash_attention', help="Enable flash attention. default=store_true", action='store_true')
    parser.add_argument('--lm_head', help="Enable quantization of lm_head layer. default=store_true", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads. default=8", type=int, default=8,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    print(f"Arguments: {args}")

    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)

    model, tokenizer = load_model(args)
    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer: {n}")
            m.device = "aie"
            m.quantize_weights()
    print(model)
    Utils.print_model_size(model)

    app = create_app(model, tokenizer)
    uvicorn.run(app, host=args.host, port=args.port)
