import gc
import torch
import json
import time
import os
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
from typing import Dict
import psutil

print('Welcome to RWKVFF! Version: 1.0.0')

RWKV_HEAD_QK_DIM = 0
DEBUG_TIME = False

@torch.jit.ignore
def sample(probs, temperature: float = 1.0, top_p_usual: float = 0.8) -> int:
    sorted_probs = torch.sort(probs, descending=True)[0]
    cumulative_probs = torch.cumsum(
        sorted_probs.float(), dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(
        cumulative_probs > top_p_usual)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    out = torch.multinomial(probs.float(), 1, True)[0]
    return out

def isIn(a, b):
    for i in a:
        if i["ctx"] == b["ctx"]:
            return True
    return False

class RWKV_RNN(nn.Module):
    def __init__(self, args, argsnumns, layers: list, device='cuda', verbose:bool=False):
        super().__init__()
        self.args = args
        self.argsnumns = argsnumns
        if device == 'cuda':
            self.FLOAT_MODE = 'fp16'
        else:
            self.FLOAT_MODE = 'fp32'
        self.RUN_DEVICE = device
        self.layerdist = layers
        self.n_layer = 0
        self.verbose = verbose
        self.procDevice = list(filter(
            lambda x: x != "proc", self.layerdist)).pop()
        if (self.procDevice == "proc"):
            self.procDevice = self.layerdist[-1]
        with torch.no_grad():
            w: Dict[str, torch.Tensor] = torch.load(args["MODEL_NAME"], map_location='cpu')
            self.n_emb = len(w['blocks.0.ln1.weight'])
            keys = list(w.keys())
            if 'pos_emb_x' in keys:
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(argsnumns["ctx_len"]+1, -1)[:-1, :]
            keys = list(w.keys())
            print_need_newline = False
            for x in keys:
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                if self.FLOAT_MODE == "fp32":
                    w[x] = w[x].float()

                elif self.FLOAT_MODE == "bf16":
                    w[x] = w[x].bfloat16()
                elif self.FLOAT_MODE == "fp16":
                    if ('weight' in x or 'bias' in x) and 'ln' in x:
                        w[x] = w[x].half()
                    else:
                        w[x] = w[x].half()
                w[x].requires_grad = False
                try:
                    if (int(x.split('.')[1])+1 > self.n_layer):
                        self.n_layer = int(x.split('.')[1])+1
                except:
                    pass
                if self.RUN_DEVICE == "cuda" and x != 'emb.weight':
                    piece = x.split('.')[1]
                    if (piece in ["weight", "bias"]):
                        processedLayer = self.layerdist[self.n_layer-1]
                        if (processedLayer == "proc"):
                            processedLayer = self.procDevice
                    else:
                        processedLayer = self.layerdist[int(x.split('.')[1])]
                        if (processedLayer == "proc"):
                            processedLayer = "cpu"
                    w[x] = w[x].to(device=processedLayer, non_blocking=True)
                if self.RUN_DEVICE == 'cuda':
                    if (w[x].device.type == "cpu"):
                        w[x] = w[x].pin_memory()
                if ('blocks.' not in x) or ('blocks.0.' in x):
                    if self.verbose:
                        print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)
                else:
                    if self.verbose:
                        print('x' if "cpu" in f'{w[x].device}' else "x", end='', flush=True)
        keys = list(w.keys())
        self.w = w

        self.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def FF(self, sx, ln2w, ln2b, statex, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        state = statex
        x = torch.layer_norm(sx, (self.n_emb,), weight=ln2w, bias=ln2b)
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid((rw @ xr))
        dx = (kw @ xk)
        clamped = torch.relu(dx)
        k = torch.square(clamped)
        kv = (vw @ k)
        return sx+(r * kv), state

    def SA(self, sx: torch.Tensor, ln1w, ln1b, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw: torch.Tensor, vw, rw, ow):

        x = torch.layer_norm(
            sx, (self.n_emb,), weight=ln1w, bias=ln1b)

        xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)

        state[5*i+1] = x

        r = torch.sigmoid((rw @ xr))
        k = (kw @ xk)
        v = (vw @ xv)

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)

        a = e1 * aa + e2 * v
        b = e1 * bb + e2

        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p

        rwkv = (r * a) / b
        return sx+(ow @ rwkv), state

    def forward(self, ctx: list, state: torch.Tensor, preprocess_only: bool = False):
        with torch.no_grad():
            if (self.layerdist[0] != self.layerdist[self.n_layer-1]):
                state = state.to(device=self.layerdist[0])

            w = self.w

            x: torch.Tensor = w["emb.weight"][ctx[-1]]

            if ("pos_emb" in w.keys()):
                pos_emb = w["pos_emb"][(len(ctx) % 1024)-1]
                x = x + pos_emb

            if self.RUN_DEVICE == 'cuda':
                x = x.to(device="cuda:0", non_blocking=True)

            for o in range(self.n_layer):
                i = o

                d: dict[str, torch.Tensor] = w

                d = {}
                for rr in w.keys():
                    if ("blocks."+str(i)+"." in rr):
                        if (self.RUN_DEVICE == "cuda" and self.layerdist[i] == "proc"):
                            d[rr] = w[rr].to(
                                self.procDevice, non_blocking=True)
                        else:
                            d[rr] = w[rr]

                ln1w = d["blocks."+str(i)+".ln1.weight"]
                ln1b = d["blocks."+str(i)+".ln1.bias"]
                tmk = d["blocks."+str(i)+".ffn.time_mix_k"]
                tmr = d["blocks."+str(i)+".ffn.time_mix_r"]
                tmkw = d["blocks."+str(i)+".ffn.key.weight"]
                tmvw = d["blocks."+str(i)+".ffn.value.weight"]
                tmrw = d["blocks."+str(i)+".ffn.receptance.weight"]
                ln2w = d["blocks."+str(i)+".ln2.weight"]
                ln2b = d["blocks."+str(i)+".ln2.bias"]
                atmk = d["blocks."+str(i)+".att.time_mix_k"]
                atmv = d["blocks."+str(i)+".att.time_mix_v"]
                atmr = d["blocks."+str(i)+".att.time_mix_r"]
                atf = d["blocks."+str(i)+".att.time_first"]
                atc = d["blocks."+str(i)+".att.time_decay"]
                atd = d["blocks."+str(i)+".att.key.weight"]
                avw = d["blocks."+str(i)+".att.value.weight"]
                arw = d["blocks."+str(i)+".att.receptance.weight"]
                aow = d["blocks."+str(i)+".att.output.weight"]

                if (i == 0):
                    x = torch.layer_norm(
                        x, (self.n_emb,), weight=d["blocks.0.ln0.weight"], bias=d["blocks.0.ln0.bias"])
                else:
                    if (self.layerdist[i] != self.layerdist[i-1]):
                        if (self.layerdist[i] == "proc"):
                            x = x.to(device=self.procDevice)
                            state = state.to(device=self.procDevice)
                        else:
                            x = x.to(device=self.layerdist[i])
                            state = state.to(device=self.layerdist[i])

                sx, state = self.SA(x, ln1w, ln1b, state, i,
                                    atmk, atmv, atmr, atf, atc, atd, avw, arw, aow
                                    )

                rx, state = self.FF(sx, ln2w, ln2b, state, i,
                                    tmk, tmr, tmkw, tmvw, tmrw)

                x = rx

                if ((self.layerdist[i] == "proc")):

                    for rr in w.keys():
                        if ("blocks."+str(i)+"." in rr):

                            del d[rr]

            if preprocess_only:
                return x, state

            return (w["head.weight"] @ torch.layer_norm(
                x, (self.n_emb,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])), state

    @ torch.jit.ignore
    def empty_state(self):
        device = self.layerdist[0]
        if (device == "proc"):
            if (self.RUN_DEVICE == "cuda"):
                device = self.procDevice
            else:
                device = "cpu"
        state = torch.zeros(
            self.n_layer * 5, self.n_emb, device=device, dtype=torch.float32 if self.FLOAT_MODE == "fp32" else torch.bfloat16 if self.FLOAT_MODE == "bf16" else torch.float16)
        for i in range(self.n_layer):
            state[5*i+4] -= 1e30
        return state

    @ torch.jit.ignore
    def loadContext(self, ctx: list = [], newctx: list = [187], statex=None, silent=False):
        if statex is None:
            statex = self.empty_state()
        m = lambda x: x
        for i in m(range(len(newctx)-1)):
            x = ctx+newctx[:i+1]
            o, statex = self.forward(
                [x[-1]], statex, preprocess_only=True)
        return ctx+newctx, statex

    @ torch.jit.ignore
    def sample_logits(self, ozut: torch.Tensor, x: list, temperature: float = 1.0, top_p_usual: float = 0.8):
        out = ozut
        if out.dtype == torch.half and out.device == torch.device('cpu'):
            out = out.float()
        probs = F.softmax(out, dim=-1)
        return sample(probs, temperature, top_p_usual)

    @ torch.jit.ignore
    def run(self, currstate: list({"score": float, "ctx": list, "state": torch.Tensor}), temp: float = 1.5, top_p: float = 0.9, nla: float = 0, endChars=[[187, 187], [535]]):
        ctx = currstate[0]["ctx"]
        state = currstate[0]["state"]
        out1, state = self.forward(ctx, state.clone())
        ttt = self.sample_logits(out1, ctx, temperature=0.8, top_p_usual=0.9)
        return [{"score": 1, "ctx": ctx+[ttt], "state": state}]

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083', verbose:bool=False):
        self.verbose = verbose
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] != None:
                from transformers import PreTrainedTokenizerFast
                if self.verbose:
                    print(WORD_NAME[0])
                self.tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

def loadModel(file, device='cuda', multi=True, float_mode='fp16', verbose:bool=False):
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    args = {}
    argsnums = {}
    vocab_size = 50277
    numdevices = 1
    if multi:
        numdevices = int(torch.cuda.device_count())
    layerdist = []
    if device == "cuda":
        for devic in range(numdevices):
            layerdist += [f"cuda:{devic}"] * 28

    if device == "cuda":
        if (numdevices == 1):
            layerdist += ["proc"] * 100 + ["cuda:0"]
        else:
            layerdist += ["proc"] * 16 + [1, 0]
    else:
        layerdist = ["cpu"]*100
    
    # fp32 // bf16 (saves VRAM, slightly less accurate) // fp16 (saves VRAM, slightly less accurate, can only be used with cuda, sometimes faster)
    if device == 'cuda':
        args["FLOAT_MODE"] = float_mode
    else:
        args["FLOAT_MODE"] = 'fp32'

    threads_count = psutil.cpu_count(logical=False)
    threads_count = int(threads_count/2)+threads_count
    if verbose:
        print('Threads:', threads_count)
    torch.set_num_threads(int(threads_count))
    opt = "jit"

    if (device == "cpu" and args["FLOAT_MODE"] == "fp16"):
        raise (Warning("fp16 is only supported on cuda"))

    args["MODEL_NAME"] = file
    argsnums["ctx_len"] = 4068
    argsnums["vocab_size"] = vocab_size
    argsnums["head_qk"] = 0
    argsnums["pre_ffn"] = 0
    argsnums["grad_cp"] = 0
    argsnums["my_pos_emb"] = 0
    os.environ["RWKV_RUN_DEVICE"] = device

    model = RWKV_RNN(args, argsnums, layerdist, device=device, verbose=verbose)
    return model


class RWKVFromFile:
    def __init__(self, file:str, tokenizer_json:str, verbose:bool=False, device='cuda'):
        self.verbose = verbose
        self.model = loadModel(file, verbose=self.verbose, device=device)
        if self.verbose:
            print(self.model.n_layer)
        self.state1 = self.model.empty_state()
        self.temp = 1.0
        self.top_p = 0.9
        self.init_state = self.state1
        if self.verbose:
            print('Optimizing speed...')
        self.model.forward([187, 187], self.state1)
        gc.collect()
        torch.cuda.empty_cache()
        self.TOKEN_MODE = "pile"
        self.WORD_NAME = [tokenizer_json]
        self.model_tokens = None
        self.UNKNOWN_CHAR = None
        if self.verbose:
            print(f'Loading tokenizer {self.WORD_NAME}...')
        self.tokenizer = TOKENIZER(self.WORD_NAME, UNKNOWN_CHAR=self.UNKNOWN_CHAR, verbose=self.verbose)
        if self.TOKEN_MODE == "pile":
            assert self.tokenizer.tokenizer.decode([187]) == '\n'

    def initialize(self, context):
        if self.verbose:
            print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        self.model_tokens = self.tokenizer.tokenizer.encode(context)
        state = self.model.loadContext(newctx=self.model_tokens)
        return state

    def save(self, filename, state):
        if self.verbose:
            print("Saved state...")
        savestates = {
            "init": (state[0], state[1].clone())
        }
        torch.save(savestates, filename)

    def load(self, filename):
        if os.path.isfile(filename):
            if self.verbose:
                print("Loading save state...")
            savestates = torch.load(filename, map_location=torch.device('cpu'))
            state = savestates["init"]
        return state

    def generate(self, context, stopping_criteria=[], max_new_tokens=25, state=None):
        state = self.model.loadContext(ctx=state[0], statex=state[1], newctx=self.tokenizer.tokenizer.encode(context), silent=True)
        state = [{"score": 1, "state": state[1], "ctx": state[0]}]
        compiled = ''
        with torch.no_grad():
            for i in range(max_new_tokens):
                state = self.model.run(state, temp=self.temp, top_p=self.top_p)
                outchar = self.tokenizer.tokenizer.decode(state[0]["ctx"][-1])
                if outchar in stopping_criteria:
                    break
                compiled += outchar
        state = (state[0]["ctx"], state[0]["state"])
        return compiled, state
