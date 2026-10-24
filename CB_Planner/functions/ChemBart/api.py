import sys
from .ChemBart import ChemBart, CB_mul_END, CB_MCTS
import torch

class CBTempYield():
    def __init__(self, ty_path, name = "temp_yield_bart", dev = "cuda:0"):
        self.model = CB_mul_END(ty_path, name, dev)
        self.model.eval()

    def pred(self, reactant, reagant, product):
        smi = reactant + ">" + reagant + ">" + product + "<n01><end>"
        inp = self.model.tokenizer.encoder(smi)
        with torch.no_grad():
            out = self.model(inp)
        return out.tolist()

class CBRetro():
    """Single-device, single-reaction prediction using the pretraining beam search."""

    def __init__(self, path, dev = "cuda:0"):
        self.model = ChemBart(path,dev)
        self.model.BartNN.eval()

    def share_memory(self):
        self.model.BartNN.share_memory()

    def _predict(self, encoder_input, decoder_input, lock, num_samples,
                 max_len, sampling_method='beam', top_k=10, top_p=0.9):
        if lock is not None:
            lock.acquire()
        try:
            # A supplied decoder prefix also occupies model positions.
            prefix = self.model.tokenizer.encoder(decoder_input)
            if len(prefix) == 0:
                raise ValueError("Decoder prefix could not be tokenized")
            prefix_length = len(prefix[0]) - 1  # Exclude tokenizer-added <end>.
            max_len = min(
                max_len, self.model.config.max_position_embeddings - prefix_length
            )
            if max_len < 1:
                raise ValueError("No model positions remain for generation")
            if sampling_method == 'beam':
                return self.model._beam_predictions(
                    [encoder_input], decoder_input=decoder_input,
                    beam_width=num_samples, max_len=max_len, stop_with_sep=True,
                )[0]
            return self.model.predict(
                encoder_input, decoder_input=decoder_input,
                sampling_method=sampling_method, max_len=max_len,
                stop_with_sep=True, top_k=top_k, top_p=top_p,
                num_samples=num_samples,
            )
        finally:
            if lock is not None:
                lock.release()

    def precursor(self, product, lock = None, top_k=10, top_p=0.9, sampling_method='beam', num_samples=50, max_len=768):
        """Predict precursors; num_samples sets beam width in beam mode."""
        out = self._predict(
            "<msk>>>"+product, "<cls>", lock, num_samples, max_len,
            sampling_method=sampling_method, top_k=top_k, top_p=top_p,
        )
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-5:] == "<end>":
                out[j][0] = ans[5:-5]
            elif ans[-1] == ">":
                out[j][0] = ans[5:-1]
            else:
                out[j][0] = ans[5:]
        return out

    def reagent(self, reactant, product, k=50, lock = None, max_len=768):
        """Predict reagents using a width-k beam (default 50)."""
        out = self._predict(
            reactant+"><msk>>"+product, "<cls>"+reactant+">", lock, k, max_len,
        )
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
        return out

    def product(self, reactant, reagent, k=50, lock = None, max_len=768):
        """Predict products using a width-k beam (default 50)."""
        # 确保 reactant 和 reagent 都是字符串

        if reagent is None:
            reagent = ""

        out = self._predict(
            reactant+">"+reagent+"><msk>",
            "<cls>"+reactant+">"+reagent+">", lock, k, max_len,
        )
        for j in range(len(out)):
            ans = out[j][0]
            if ans[-1] != ">":
                out[j][0] = ans.split(">")[-1]
            else:
                ans = ans.split(">")[-2]
                if ans[-4:] == "<end":
                    ans = ans[:-4]
                out[j][0] = ans
        return out

class CBRL():
    def __init__(self, path, dev):
        self.model = CB_MCTS(path, dev)
        self.tokenizer = self.model.core.tokenizer
        self.model.core.eval()
    
    def share_memory(self):
        self.model.core.share_memory()

    def policy(self, product, precursorlist, lock = None):
        inputlist = [self.tokenizer.encoder(precursor + ">>" + product) for precursor in precursorlist]
        if lock is not None:
            lock.acquire()
        with torch.no_grad():
            ret = (self.model.policy(inputlist)).tolist()
        if lock is not None:
            lock.release()
        return ret

    def value(self, product, lock = None):
        smi = self.tokenizer.encoder(product)
        if lock is not None:
            lock.acquire()
        with torch.no_grad():
            ret = (self.model.value(smi)).item()
        if lock is not None:
            lock.release()
        return ret
