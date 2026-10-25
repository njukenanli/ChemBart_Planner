import sys
from pathlib import Path
from .ChemBart import ChemBart, CB_mul_END, CB_MCTS
import torch

def _require_checkpoint(path):
    """Use the supplied inference checkpoint; never fall back to another model."""
    if not Path(path).is_file():
        raise FileNotFoundError("Configured model checkpoint not found: {}".format(path))
    return path

class CBTempYield():
    def __init__(self, ty_path, name = "temp_yield_bart", dev = "cuda:0"):
        self.model = CB_mul_END(_require_checkpoint(ty_path), name, dev)
        self.model.to(dev)
        self.model.eval()

    def pred(self, reactant, reagant, product):
        if reagant is None:
            reagant = ""
        smi = reactant + ">" + reagant + ">" + product + "<n01><end>"
        inp = self.model.tokenizer.encoder(smi).to(self.model.device)
        with torch.no_grad():
            out = self.model(inp)
        return out.tolist()

class CBRetro():
    """Single-device, single-reaction prediction using the pretraining beam search."""

    def __init__(self, path, dev = "cuda:0"):
        self.model = ChemBart(_require_checkpoint(path), dev)
        self.model.BartNN.eval()

    def share_memory(self):
        self.model.BartNN.share_memory()

    def _token_length(self, smiles):
        tokens = self.model.tokenizer.encoder("<cls>" + smiles)
        if len(tokens) == 0:
            raise ValueError("Molecule could not be tokenized within the model input limit")
        return len(tokens[0]) - 2  # Exclude <cls> and <end>; empty SMILES has length 0.

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
                max_len, 1018,
                self.model.config.max_position_embeddings - prefix_length,
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

    def precursor(self, product, lock = None, top_k=10, top_p=0.9, sampling_method='beam', num_samples=50, max_len=1018):
        """Predict precursors; num_samples sets beam width in beam mode."""
        # Reserve <cls>, ><msk>>, the product, <end>, and one extra <n01> position.
        max_len = min(max_len, self.model.config.max_position_embeddings
                      - 1 - 3 - self._token_length(product) - 1 - 1)
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

    def reagent(self, reactant, product, k=50, lock = None, max_len=1018):
        """Predict reagents using a width-k beam (default 50)."""
        # Reserve <cls>, reactant, both separators, product, <end>, and <n01>.
        max_len = min(max_len, self.model.config.max_position_embeddings
                      - 1 - self._token_length(reactant) - 2 - self._token_length(product) - 1 - 1)
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

    def product(self, reactant, reagent, k=50, lock = None, max_len=1018):
        """Predict products using a width-k beam (default 50)."""
        # 确保 reactant 和 reagent 都是字符串

        if reagent is None:
            reagent = ""

        # Reserve <cls>, reactant, reagent, both separators, <end>, and <n01>.
        max_len = min(max_len, self.model.config.max_position_embeddings
                      - 1 - self._token_length(reactant) - 1 - self._token_length(reagent) - 1 - 1 - 1)
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
        self.model = CB_MCTS(_require_checkpoint(path), dev)
        self.tokenizer = self.model.core.tokenizer
        self.model.core.eval()
    
    def share_memory(self):
        self.model.core.share_memory()

    def policy(self, product, precursorlist, lock = None):
        inputlist = [self.tokenizer.encoder(precursor + ">>" + product) for precursor in precursorlist]
        if lock is not None:
            lock.acquire()
        try:
            with torch.no_grad():
                return (self.model.policy(inputlist)).tolist()
        finally:
            if lock is not None:
                lock.release()

    def value(self, product, lock = None):
        smi = self.tokenizer.encoder(product)
        if lock is not None:
            lock.acquire()
        try:
            with torch.no_grad():
                return (self.model.value(smi)).item()
        finally:
            if lock is not None:
                lock.release()
