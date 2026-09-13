import os, sys
absdir = os.path.dirname(os.path.abspath(__file__))+"/"
sys.path.insert(0, absdir)
from CBTokenizer import CBTokenizer
from transformers import BartForConditionalGeneration
from transformers import BartConfig
import gc
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import *
import random
import math


class ChemBart():
    tokenizer=None
    BartNN=None
    config=None

    class _ThreeDirectionReactionDataset(Dataset):
        """Build precursor, reagent, and product prediction examples."""

        def __init__(self, stringlist, tokenizer, max_length, report_invalid=True):
            self.reactions = []
            self.mask_token_id = int(tokenizer.vocab["<msk>"])
            separator_id = int(tokenizer.vocab[">"])
            for reaction_index, reaction in enumerate(stringlist):
                if not isinstance(reaction, str):
                    raise TypeError(
                        "reaction at index {} must be a string".format(
                            reaction_index
                        )
                    )
                encoded = tokenizer.encoder(reaction, alllen=max_length)
                if len(encoded) == 0:
                    if report_invalid:
                        print(
                            "Skipping reaction {}: tokenization failed".format(
                                reaction_index
                            ),
                            flush=True,
                        )
                    continue
                target_ids = encoded[0].long()
                separator_positions = (
                    target_ids.eq(separator_id)
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                if len(separator_positions) != 2:
                    if report_invalid:
                        print(
                            "Skipping reaction {}: expected exactly two '>' "
                            "separators, found {}".format(
                                reaction_index, len(separator_positions)
                            ),
                            flush=True,
                        )
                    continue
                first_separator, second_separator = separator_positions
                target_length = target_ids.numel()
                source_lengths = (
                    3 + target_length - second_separator,
                    first_separator + 2 + target_length - second_separator,
                    second_separator + 3,
                )
                if max(source_lengths) > max_length:
                    if report_invalid:
                        print(
                            "Skipping reaction {}: a masked input exceeds the "
                            "model's maximum length of {} tokens".format(
                                reaction_index, max_length
                            ),
                            flush=True,
                        )
                    continue
                self.reactions.append(
                    (target_ids, first_separator, second_separator)
                )

        def __len__(self):
            return 3 * len(self.reactions)

        def __getitem__(self, index):
            full_target_ids, first_separator, second_separator = self.reactions[
                index // 3
            ]
            direction = index % 3
            mask = full_target_ids.new_tensor([self.mask_token_id])

            if direction == 0:
                # Encoder: <cls><msk>>>product<end>
                source_ids = torch.cat(
                    (
                        full_target_ids[:1],
                        mask,
                        full_target_ids[first_separator : first_separator + 1],
                        full_target_ids[second_separator:],
                    )
                )
                # Target: <cls>reactant>>product<end>
                target_ids = torch.cat(
                    (
                        full_target_ids[: first_separator + 1],
                        full_target_ids[second_separator:],
                    )
                )
            elif direction == 1:
                # Encoder: <cls>reactant><msk>>product<end>
                source_ids = torch.cat(
                    (
                        full_target_ids[: first_separator + 1],
                        mask,
                        full_target_ids[second_separator:],
                    )
                )
                target_ids = full_target_ids
            else:
                # Encoder: <cls>reactant>reagent><msk><end>
                source_ids = torch.cat(
                    (
                        full_target_ids[: second_separator + 1],
                        mask,
                        full_target_ids[-1:],
                    )
                )
                target_ids = full_target_ids
            return source_ids, target_ids

    class _ThreeDirectionCollator:
        """Pad and shift each task-specific target for teacher forcing."""

        def __init__(self, pad_token_id):
            self.pad_token_id = int(pad_token_id)

        def __call__(self, samples):
            batch_size = len(samples)
            source_length = max(source.numel() for source, _ in samples)
            decoder_length = max(target.numel() - 1 for _, target in samples)
            source_ids = torch.full(
                (batch_size, source_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            decoder_input_ids = torch.full(
                (batch_size, decoder_length),
                self.pad_token_id,
                dtype=torch.long,
            )
            labels = torch.full(
                (batch_size, decoder_length), -100, dtype=torch.long
            )
            for row, (source, target) in enumerate(samples):
                source_ids[row, : source.numel()] = source
                shifted_length = target.numel() - 1
                decoder_input_ids[row, :shifted_length] = target[:-1]
                labels[row, :shifted_length] = target[1:]

            # Each target is shifted once: decoder input excludes <end>, while
            # labels exclude <cls>. Every non-padding label token contributes
            # to the loss in the same parallel forward pass.
            return {
                "input_ids": source_ids,
                "attention_mask": source_ids.ne(self.pad_token_id).long(),
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_input_ids.ne(
                    self.pad_token_id
                ).long(),
                "labels": labels,
            }

    @staticmethod
    def _find_pretrain_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _paral_train_worker(
        local_rank,
        world_size,
        master_port,
        model_path,
        stringlist,
        train_kwargs,
    ):
        """Entry point used by ``torch.multiprocessing.spawn``."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        trainer = ChemBart(model_path, dev="cpu")
        trainer._paral_train_impl(
            stringlist=stringlist,
            device=torch.device("cuda", local_rank),
            rank=local_rank,
            world_size=world_size,
            initialize_process_group=True,
            **train_kwargs
        )

    @staticmethod
    def _pretrain_eval_worker(
        local_rank,
        world_size,
        master_port,
        checkpoint_path,
        trainset,
        testset,
        max_new_tokens_precursor,
        epoch_number,
        log_file,
    ):
        """Evaluate one data shard per GPU and aggregate precursor accuracy."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        evaluator = ChemBart(
            checkpoint_path,
            dev="cuda:{}".format(local_rank)
        )
        evaluator._evaluate_pretrain_epoch(
            trainset=trainset,
            testset=testset,
            max_new_tokens_precursor=max_new_tokens_precursor,
            epoch_number=epoch_number,
            log_file=log_file,
            device=torch.device("cuda", local_rank),
            rank=local_rank,
            world_size=world_size,
            initialize_process_group=True,
        )

    def __init__(self, path: Optional[str], dev: str = "cpu"):
        self.tokenizer=CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.load_model(path)
        self.dev = torch.device(dev)
        self.BartNN.to(self.dev)
        self.model_path = path
    
    def load_model(self, path: Optional[str]):
        self.BartNN=BartForConditionalGeneration(self.config)
        if path is not None:
            self.BartNN.load_state_dict(torch.load(path, map_location='cpu'))
            print("load previous model: "+path)
        else:
            print("new model")

    def trans_to_list(self,l):#<cls> is not included
        out=[]
        for i in range(1,len(l)):
            temp=[0.0]*len(self.tokenizer.vocab)
            temp[l[i]]=1.0
            out.append(temp)
        return out

    def compare(self, prediction: str, label: str) -> bool:
        def canonize(smi: str) -> str:
            from rdkit import Chem
            smi_list = [part for part in smi.split(".") if part.strip()]
            result: List[str] = []
            for part in smi_list:
                molecule = Chem.MolFromSmiles(part)
                if molecule is None:
                    continue
                result.append(
                    Chem.MolToSmiles(
                        molecule,
                        canonical=True,
                        isomericSmiles=True,
                        kekuleSmiles=False,
                    )
                )
            return ".".join(result)
        prediction = canonize(prediction)
        label = canonize(label)
        pred_set = {part for part in prediction.split(".") if part.strip()}
        label_set = {part for part in label.split(".") if part.strip()}
        return label_set.issubset(pred_set) or len(label_set&pred_set)>0

    def pretrain(
        self,
        trainset: List[str], # list['reactant>reagent>product']
        testset: List[str],
        mini_batch_size: int = 4,
        accumulative_steps: int = 4,
        lr: float = 1e-5,
        epochs: int = 100,
        max_new_tokens_precursor: int = 768,
        log_file: str = "log.txt",
        ckpt_path: str = "./pretrainckpt",
    ) -> None:
        """Train, checkpoint, and evaluate the three-direction pretraining task.

        ``self.model_path`` is an optional source of initial weights.  Epoch
        checkpoints are always written under ``ckpt_path`` and never overwrite
        the initial model file.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if mini_batch_size < 1:
            raise ValueError("mini_batch_size must be at least 1")
        if accumulative_steps < 1:
            raise ValueError("accumulative_steps must be at least 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if max_new_tokens_precursor < 1:
            raise ValueError("max_new_tokens_precursor must be at least 1")
        if max_new_tokens_precursor > self.config.max_position_embeddings - 1:
            raise ValueError(
                "max_new_tokens_precursor cannot exceed {}".format(
                    self.config.max_position_embeddings - 1
                )
            )
        trainset = list(trainset)
        testset = list(testset)
        if log_file is not None:
            log_file = str(Path(log_file).expanduser().resolve())
        if ckpt_path is None:
            raise ValueError("ckpt_path must be a directory path")
        checkpoint_directory = Path(ckpt_path).expanduser().resolve()
        if checkpoint_directory.exists() and not checkpoint_directory.is_dir():
            raise ValueError(
                "ckpt_path is not a directory: {}".format(
                    checkpoint_directory
                )
            )
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        starting_checkpoint = (
            None
            if self.model_path is None
            else str(Path(self.model_path).expanduser().resolve())
        )
        for epoch_number in range(1, epochs + 1):
            epoch_checkpoint = self._pretrain_checkpoint_path(
                epoch_number, checkpoint_directory
            )
            if epoch_checkpoint.is_file():
                print(
                    "epoch {}/{}: loading existing checkpoint {}; skipping "
                    "training and evaluation".format(
                        epoch_number, epochs, epoch_checkpoint
                    ),
                    flush=True,
                )
                state = torch.load(str(epoch_checkpoint), map_location="cpu")
                self.BartNN.load_state_dict(state)
                del state
                self.BartNN.to(self.dev)
                starting_checkpoint = str(epoch_checkpoint)
                continue

            # Keep paral_train's public API and one-epoch optimizer lifetime.
            # These private values only assign the correct checkpoint number
            # and starting weights to automatically spawned DDP workers.
            self._paral_train_checkpoint_start_epoch = epoch_number
            self._paral_train_starting_checkpoint = starting_checkpoint
            self._paral_train_ckpt_path = str(checkpoint_directory)
            try:
                self.paral_train(
                    trainset,
                    epoch=1,
                    mini_batch_size=mini_batch_size,
                    accumulative_steps=accumulative_steps,
                    lr=lr,
                    log_file=log_file,
                )
            finally:
                del self._paral_train_checkpoint_start_epoch
                del self._paral_train_starting_checkpoint
                del self._paral_train_ckpt_path

            starting_checkpoint = str(epoch_checkpoint)
            self._run_pretrain_evaluation(
                trainset=trainset,
                testset=testset,
                checkpoint_path=epoch_checkpoint,
                max_new_tokens_precursor=max_new_tokens_precursor,
                epoch_number=epoch_number,
                log_file=log_file,
            )

    @staticmethod
    def _reaction_to_precursor_example(reaction):
        """Return the precursor-evaluation input and reactant label."""
        if not isinstance(reaction, str):
            return None
        reaction = reaction.strip()
        if reaction.startswith("<cls>"):
            reaction = reaction[5:]
        if reaction.endswith("<end>"):
            reaction = reaction[:-5]
        parts = reaction.split(">")
        if len(parts) != 3 or not parts[0] or not parts[2]:
            return None
        reactant, _reagent, product = parts
        encoder_input = "<cls><msk>>>" + product + "<end>"
        return encoder_input, reactant

    @staticmethod
    def _decoded_precursor(decoded):
        """Remove decoder control tokens from a precursor prediction."""
        if decoded.startswith("<cls>"):
            decoded = decoded[5:]
        decoded = decoded.split(">", 1)[0]
        decoded = decoded.split("<end>", 1)[0]
        return decoded

    def _run_pretrain_evaluation(
        self,
        trainset,
        testset,
        checkpoint_path,
        max_new_tokens_precursor,
        epoch_number,
        log_file,
    ):
        """Use every visible GPU for train/test precursor evaluation."""
        environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if environment_world_size > 1 or dist.is_initialized():
            world_size = (
                dist.get_world_size()
                if dist.is_initialized()
                else environment_world_size
            )
            rank = (
                dist.get_rank()
                if dist.is_initialized()
                else int(os.environ["RANK"])
            )
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            original_device = self.dev
            try:
                self._evaluate_pretrain_epoch(
                    trainset=trainset,
                    testset=testset,
                    max_new_tokens_precursor=max_new_tokens_precursor,
                    epoch_number=epoch_number,
                    log_file=log_file,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    initialize_process_group=not dist.is_initialized(),
                )
            finally:
                self.BartNN.to(original_device)
                self.dev = original_device
            return

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            original_device = self.dev
            del self.BartNN
            gc.collect()
            torch.cuda.empty_cache()
            try:
                mp.spawn(
                    ChemBart._pretrain_eval_worker,
                    args=(
                        gpu_count,
                        self._find_pretrain_port(),
                        str(checkpoint_path),
                        trainset,
                        testset,
                        max_new_tokens_precursor,
                        epoch_number,
                        log_file,
                    ),
                    nprocs=gpu_count,
                    join=True,
                )
            finally:
                self.load_model(checkpoint_path)
                self.BartNN.to(original_device)
            return

        device = torch.device("cuda", 0) if gpu_count == 1 else torch.device("cpu")
        original_device = self.dev
        try:
            self._evaluate_pretrain_epoch(
                trainset=trainset,
                testset=testset,
                max_new_tokens_precursor=max_new_tokens_precursor,
                epoch_number=epoch_number,
                log_file=log_file,
                device=device,
                rank=0,
                world_size=1,
                initialize_process_group=False,
            )
        finally:
            self.BartNN.to(original_device)
            self.dev = original_device

    def _evaluate_pretrain_epoch(
        self,
        trainset,
        testset,
        max_new_tokens_precursor,
        epoch_number,
        log_file,
        device,
        rank,
        world_size,
        initialize_process_group,
    ):
        """Evaluate precursor top-1 on train and top-10 on test."""
        owns_process_group = False
        if initialize_process_group:
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            owns_process_group = True

        try:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            self.dev = device
            self.BartNN.to(device)
            self.BartNN.eval()

            train_correct = 0
            train_evaluated = 0
            for sample_index in range(rank, len(trainset), world_size):
                example = self._reaction_to_precursor_example(
                    trainset[sample_index]
                )
                if example is None:
                    continue
                encoder_input, label = example
                try:
                    candidates = self.predict(
                        encoder_input,
                        decoder_input="<cls>",
                        sampling_method="beam",
                        top_k=1,
                        max_len=max_new_tokens_precursor,
                        stop_with_sep=True,
                        num_samples=1,
                    )
                except Exception as error:
                    print(
                        "Skipping train evaluation sample {} on rank {}: {}".format(
                            sample_index, rank, error
                        ),
                        flush=True,
                    )
                    continue
                train_evaluated += 1
                if candidates and self.compare(
                    self._decoded_precursor(candidates[0][0]), label
                ):
                    train_correct += 1

            test_correct_at_10 = 0
            test_evaluated = 0
            for sample_index in range(rank, len(testset), world_size):
                example = self._reaction_to_precursor_example(
                    testset[sample_index]
                )
                if example is None:
                    continue
                encoder_input, label = example
                try:
                    candidates = self.predict(
                        encoder_input,
                        decoder_input="<cls>",
                        sampling_method="beam",
                        top_k=50,
                        max_len=max_new_tokens_precursor,
                        stop_with_sep=True,
                        num_samples=50,
                    )
                except Exception as error:
                    print(
                        "Skipping test evaluation sample {} on rank {}: {}".format(
                            sample_index, rank, error
                        ),
                        flush=True,
                    )
                    continue

                test_evaluated += 1
                for candidate in candidates[:10]:
                    prediction = self._decoded_precursor(candidate[0])
                    if self.compare(prediction, label):
                        test_correct_at_10 += 1
                        break

            metrics = torch.tensor(
                [train_correct, train_evaluated]
                + [test_correct_at_10, test_evaluated],
                dtype=torch.long,
                device=device,
            )
            if world_size > 1:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            if rank == 0:
                train_accuracy = metrics[0].item() / max(1, metrics[1].item())
                test_accuracy = metrics[2].item() / max(1, metrics[3].item())
                message = (
                    "Epoch {}: train precursor accuracy: top-1={:.6f}; "
                    "test precursor accuracy: top-10={:.6f}"
                ).format(
                    epoch_number,
                    train_accuracy,
                    test_accuracy,
                )
                if log_file is None:
                    print(message, flush=True)
                else:
                    log_path = Path(log_file)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        print(message, file=handle, flush=True)
        finally:
            if owns_process_group and dist.is_initialized():
                dist.destroy_process_group()

    def _pretrain_checkpoint_path(self, epoch_number, ckpt_path):
        return Path(ckpt_path).expanduser().resolve() / (
            "ChemBart_pretrain_{}.pth".format(epoch_number)
        )

    def _save_pretrain_checkpoint(self, model, epoch_number, ckpt_path):
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        checkpoint_path = self._pretrain_checkpoint_path(
            epoch_number, ckpt_path
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        torch.save(raw_model.state_dict(), str(temporary_path))
        os.replace(str(temporary_path), str(checkpoint_path))
        return checkpoint_path

    def paral_train(
        self,
        stringlist,
        epoch=100,
        mini_batch_size=4,
        accumulative_steps=4,
        lr=1e-5,
        log_file=None,
    ):
        """Train all three masked reaction directions with teacher forcing.

        All visible GPUs are used. Multiple GPUs run one DDP process per GPU;
        one GPU or a CPU-only host runs directly. ``self.model_path`` is only
        the initialization checkpoint and is never written by this method.
        """
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        if mini_batch_size < 1:
            raise ValueError("mini_batch_size must be at least 1")
        if accumulative_steps < 1:
            raise ValueError("accumulative_steps must be at least 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if log_file is not None:
            try:
                log_file = str(Path(log_file).expanduser().resolve())
            except TypeError:
                raise TypeError("log_file must be a path or None")

        reactions = list(stringlist)
        checkpoint_start_epoch = int(
            getattr(self, "_paral_train_checkpoint_start_epoch", 1)
        )
        starting_checkpoint_value = getattr(
            self,
            "_paral_train_starting_checkpoint",
            self.model_path,
        )
        starting_checkpoint = (
            None
            if starting_checkpoint_value is None
            else str(Path(starting_checkpoint_value).expanduser().resolve())
        )
        checkpoint_directory = Path(
            getattr(
                self,
                "_paral_train_ckpt_path",
                "./pretrainckpt",
            )
        ).expanduser().resolve()
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        train_kwargs = {
            "epochs": epoch,
            "mini_batch_size": mini_batch_size,
            "accumulative_steps": accumulative_steps,
            "lr": lr,
            "log_file": log_file,
            "checkpoint_start_epoch": checkpoint_start_epoch,
            "ckpt_path": str(checkpoint_directory),
        }

        environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if environment_world_size > 1 or dist.is_initialized():
            original_device = self.dev
            world_size = (
                dist.get_world_size()
                if dist.is_initialized()
                else environment_world_size
            )
            rank = (
                dist.get_rank()
                if dist.is_initialized()
                else int(os.environ["RANK"])
            )
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            try:
                self._paral_train_impl(
                    stringlist=reactions,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    initialize_process_group=not dist.is_initialized(),
                    **train_kwargs
                )
            finally:
                self.BartNN.to(original_device)
            return

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            original_device = self.dev
            del self.BartNN
            gc.collect()
            torch.cuda.empty_cache()
            completed = False
            try:
                mp.spawn(
                    ChemBart._paral_train_worker,
                    args=(
                        gpu_count,
                        self._find_pretrain_port(),
                        starting_checkpoint,
                        reactions,
                        train_kwargs,
                    ),
                    nprocs=gpu_count,
                    join=True,
                )
                completed = True
            finally:
                # Keep this parent ChemBart instance usable after spawned
                # workers exit. On failure, restore the starting checkpoint.
                reload_path = (
                    self._pretrain_checkpoint_path(
                        checkpoint_start_epoch + epoch - 1,
                        checkpoint_directory,
                    )
                    if completed
                    else (
                        None
                        if starting_checkpoint is None
                        else Path(starting_checkpoint)
                    )
                )
                if reload_path is not None:
                    self.load_model(str(reload_path))
                self.BartNN.to(original_device)
            return

        device = torch.device("cuda", 0) if gpu_count == 1 else torch.device("cpu")
        original_device = self.dev
        try:
            self._paral_train_impl(
                stringlist=reactions,
                device=device,
                rank=0,
                world_size=1,
                initialize_process_group=False,
                **train_kwargs
            )
        finally:
            self.BartNN.to(original_device)

    def _paral_train_impl(
        self,
        stringlist,
        epochs,
        mini_batch_size,
        accumulative_steps,
        lr,
        log_file,
        checkpoint_start_epoch,
        ckpt_path,
        device,
        rank,
        world_size,
        initialize_process_group,
    ):
        owns_process_group = False
        if initialize_process_group:
            backend = "nccl" if device.type == "cuda" else "gloo"
            dist.init_process_group(
                backend=backend, rank=rank, world_size=world_size
            )
            owns_process_group = True

        try:
            if device.type == "cuda":
                torch.cuda.set_device(device)
            raw_model = self.BartNN.to(device)
            distributed = world_size > 1
            if distributed:
                model = DistributedDataParallel(
                    raw_model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    output_device=device.index if device.type == "cuda" else None,
                )
            else:
                model = raw_model

            dataset = self._ThreeDirectionReactionDataset(
                stringlist,
                self.tokenizer,
                int(self.config.max_position_embeddings),
                report_invalid=rank == 0,
            )
            if len(dataset) == 0:
                raise ValueError("stringlist contains no valid reaction strings")
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                )
                if distributed
                else None
            )
            pad_token_id = int(self.tokenizer.vocab["<pad>"])
            loader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=0,
                collate_fn=self._ThreeDirectionCollator(pad_token_id),
                pin_memory=device.type == "cuda",
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-5
            )
            optimizer.zero_grad(set_to_none=True)

            for current_epoch in range(epochs):
                checkpoint_epoch = checkpoint_start_epoch + current_epoch
                if sampler is not None:
                    # ``pretrain`` calls this method for one epoch at a time,
                    # so ``current_epoch`` would otherwise always be zero.
                    # Seed the sampler with the absolute checkpoint epoch to
                    # obtain a different deterministic shuffle every epoch.
                    sampler.set_epoch(checkpoint_epoch)
                model.train()
                accumulated_token_count = torch.zeros((), device=device)
                epoch_loss_sum = torch.zeros((), device=device)
                epoch_token_count = torch.zeros((), device=device)

                for batch_index, batch in enumerate(loader):
                    should_update = (
                        (batch_index + 1) % accumulative_steps == 0
                        or batch_index + 1 == len(loader)
                    )
                    tensor_batch = {
                        key: value.to(
                            device, non_blocking=device.type == "cuda"
                        )
                        for key, value in batch.items()
                    }
                    synchronization_context = (
                        model.no_sync()
                        if distributed and not should_update
                        else nullcontext()
                    )
                    with synchronization_context:
                        outputs = model(
                            input_ids=tensor_batch["input_ids"],
                            attention_mask=tensor_batch["attention_mask"],
                            decoder_input_ids=tensor_batch["decoder_input_ids"],
                            decoder_attention_mask=tensor_batch[
                                "decoder_attention_mask"
                            ],
                            use_cache=False,
                            return_dict=True,
                        )
                        labels = tensor_batch["labels"]
                        loss_sum = F.cross_entropy(
                            outputs.logits.reshape(-1, outputs.logits.size(-1)),
                            labels.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        token_count = labels.ne(-100).sum()
                        loss_sum.backward()

                    accumulated_token_count += token_count.detach()
                    epoch_loss_sum += loss_sum.detach()
                    epoch_token_count += token_count.detach()
                    if not should_update:
                        continue

                    normalization_tokens = accumulated_token_count.clone()
                    if distributed:
                        dist.all_reduce(
                            normalization_tokens, op=dist.ReduceOp.SUM
                        )
                    token_denominator = max(1.0, normalization_tokens.item())
                    gradient_scale = (
                        world_size / token_denominator
                        if distributed
                        else 1.0 / token_denominator
                    )
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_token_count.zero_()

                epoch_metrics = torch.stack(
                    (epoch_loss_sum, epoch_token_count)
                ).to(dtype=torch.float64)
                if distributed:
                    dist.all_reduce(epoch_metrics, op=dist.ReduceOp.SUM)
                if rank == 0:
                    average_loss = epoch_metrics[0].item() / max(
                        1.0, epoch_metrics[1].item()
                    )
                    self._save_pretrain_checkpoint(
                        model, checkpoint_epoch, ckpt_path
                    )
                    message = "Epoch {}: loss per decoder token: {:.6f}".format(
                        checkpoint_epoch,
                        average_loss,
                    )
                    if log_file is None:
                        print(message, flush=True)
                    else:
                        log_path = Path(log_file)
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as handle:
                            print(message, file=handle, flush=True)
                if distributed:
                    dist.barrier()
        finally:
            if owns_process_group and dist.is_initialized():
                dist.destroy_process_group()

    def transform(self, smiles_list):
        """Return the final decoder hidden state for each valid SMILES string."""
        self.BartNN.to(self.dev)
        self.BartNN.eval()
        outputs = []
        with torch.no_grad():
            for smile in smiles_list:
                input_ids = self.tokenizer.encoder(smile)
                if len(input_ids) == 0:
                    continue
                hidden = self.BartNN(
                    input_ids=input_ids.to(self.dev),
                    decoder_input_ids=input_ids.to(self.dev),
                    return_dict=True,
                    output_hidden_states=True,
                ).decoder_hidden_states[-1][0][-1]
                outputs.append(hidden.cpu().numpy())
        return outputs

    '''
    def predict(self, s, decoder_input="<cls>",
                top_k=10, max_len=60, stop_with_sep = True):
        with torch.no_grad():
            inputvector=self.tokenizer.encoder(s)[0].tolist()
            self.BartNN=self.BartNN.eval()
            decodervector = self.tokenizer.encoder(decoder_input)[0][:-1].tolist()
            #outputdict=self.BartNN.generate(inputvector, min_length=0, max_length=60 ,return_dict_in_generate=True ,output_scores=True, 
            #    num_beams=top_k , num_return_sequences=top_k,bos_token_id=1,decoder_start_token_id=1,eos_token_id=3,pad_token_id=0 )
            outputprob=self._beam_search(inputvector,decodervector,
                                        top_k,max_len,stop_with_sep, self.dev)
            #print(outputprob)
            outl=[]
            for i in range(top_k):
                outl.append([self.tokenizer.decoder(outputprob[i][0]),outputprob[i][1]])
            return outl
    '''
    def predict(self, s, decoder_input="<cls>",
                sampling_method='beam', top_k=10, top_p=0.9,
                max_len=60, stop_with_sep=True, num_samples=5):
        '''
        for generation.

        Args:
            s: input string
            decoder_input: initial decoder input token (e.g., <cls>)
            sampling_method: 'beam', 'top_k', or 'top_p'
            top_k: number of top tokens to consider in top_k sampling
            top_p: cumulative probability threshold in top_p sampling
            max_len: maximum length of generated sequence
            stop_with_sep: whether to stop when encountering a special token (like <end> or >)
            num_samples: number of sequences to sample
        '''
        with torch.no_grad():
            inputvector = self.tokenizer.encoder(s)[0].tolist()
            self.BartNN = self.BartNN.eval()
            decodervector = self.tokenizer.encoder(decoder_input)[0][:-1].tolist()

            if sampling_method == 'beam':
                outputprob = self._beam_search(inputvector, decodervector, num_samples, max_len, stop_with_sep, self.dev)
            elif sampling_method == 'top_k':
                outputprob = self._top_k_sampling(inputvector, decodervector, top_k, max_len, stop_with_sep, self.dev, num_samples=num_samples)
            elif sampling_method == 'top_p':
                outputprob = self._top_p_sampling(inputvector, decodervector, top_p, max_len, stop_with_sep, self.dev, num_samples=num_samples)
            else:
                raise ValueError("Invalid sampling method. Choose from 'beam', 'top_k', or 'top_p'.")

            outl = []
            for i in range(min(num_samples, len(outputprob))):
                decoded_text = self.tokenizer.decoder(outputprob[i][0])
                outl.append([decoded_text, outputprob[i][1]])
            return outl
        
    @torch.no_grad()
    def _beam_search(self, s, decodervector, k, maxlen, stop_with_sep, dev):
        """Generate a shrinking beam ranked by mean token log probability.

        The first decoder position creates ``k`` beams.  Thereafter a live
        pool of size ``P`` creates ``P * k`` candidates and retains its best
        ``P`` candidates.  Any retained candidate that terminates is moved to
        the finished pool, so the live pool can only shrink.  The finished
        pool is independently capped at ``k`` sequences.

        Scores are accumulated in log space and compared as mean log
        probability, which is equivalent to the geometric mean probability
        without suffering from probability-product underflow.  Token IDs and
        accumulated scores are ordinary CPU Python values between steps.
        """
        if k < 1:
            raise ValueError("beam width must be at least 1")
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        if len(decodervector) == 0:
            raise ValueError("decodervector must contain at least one token")

        device = torch.device(dev)
        model = (
            self.BartNN.module
            if isinstance(self.BartNN, DistributedDataParallel)
            else self.BartNN
        )
        model.eval()
        source_ids = torch.tensor([s], dtype=torch.long, device=device)
        source_attention_mask = source_ids.ne(
            int(self.tokenizer.vocab["<pad>"])
        ).long()
        encoder_outputs = model.get_encoder()(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            return_dict=True,
        )
        encoder_hidden_state = encoder_outputs.last_hidden_state

        end_token_id = int(self.tokenizer.vocab["<end>"])
        separator_token_id = int(self.tokenizer.vocab[">"])
        prefix_ids = list(decodervector)

        # A beam entry is (token IDs, sum of generated-token log probabilities,
        # generated-token count).  The supplied decoder prefix is not scored.
        active = [(prefix_ids, 0.0, 0)]
        finished = []
        live_pool_size = k
        past_key_values = None

        def mean_log_probability(candidate):
            return candidate[1] / max(1, candidate[2])

        def select_cache_rows(cache, parent_indices):
            """Gather retained parents, including duplicate cache rows."""
            if cache is None or not parent_indices:
                return None
            indices = torch.tensor(
                parent_indices, dtype=torch.long, device=device
            )

            # Current Transformers cache classes mutate themselves when rows
            # are selected.  ``batch_select_indices`` supports duplicated
            # parent rows, which are required when siblings survive pruning.
            if hasattr(cache, "batch_select_indices"):
                cache.batch_select_indices(indices)
                return cache
            if hasattr(cache, "reorder_cache"):
                cache.reorder_cache(indices)
                return cache

            # Compatibility with Transformers versions that return the legacy
            # tuple[layer][self/cross-attention state] cache representation.
            return tuple(
                tuple(
                    None
                    if state is None
                    else state.index_select(0, indices.to(state.device))
                    for state in layer_cache
                )
                for layer_cache in cache
            )

        for generation_step in range(maxlen):
            if not active or live_pool_size == 0:
                break

            current_batch_size = len(active)
            if past_key_values is None:
                decoder_input_ids = torch.tensor(
                    [candidate[0] for candidate in active],
                    dtype=torch.long,
                    device=device,
                )
            else:
                decoder_input_ids = torch.tensor(
                    [[candidate[0][-1]] for candidate in active],
                    dtype=torch.long,
                    device=device,
                )

            outputs = model(
                encoder_outputs=(
                    encoder_hidden_state.expand(current_batch_size, -1, -1),
                ),
                attention_mask=source_attention_mask.expand(
                    current_batch_size, -1
                ),
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            log_probabilities = F.log_softmax(
                outputs.logits[:, -1, :], dim=-1
            )
            branch_count = min(k, log_probabilities.size(-1))
            branch_log_probabilities, branch_token_ids = torch.topk(
                log_probabilities, k=branch_count, dim=-1
            )
            token_rows = branch_token_ids.cpu().tolist()
            log_probability_rows = branch_log_probabilities.cpu().tolist()
            output_cache = getattr(outputs, "past_key_values", None)

            candidates = []
            for parent_index, (tokens, log_sum, token_count) in enumerate(active):
                for branch_index in range(branch_count):
                    token_id = int(token_rows[parent_index][branch_index])
                    candidates.append(
                        (
                            tokens + [token_id],
                            log_sum
                            + float(
                                log_probability_rows[parent_index][branch_index]
                            ),
                            token_count + 1,
                            parent_index,
                        )
                    )

            # At position zero, one decoder prefix expands into the initial
            # width-k pool.  Later positions retain at most the current live
            # pool size, which makes beam termination permanently shrink it.
            retained = sorted(
                candidates,
                key=lambda candidate: mean_log_probability(candidate[:3]),
                reverse=True,
            )[:live_pool_size]

            next_active = []
            retained_parent_indices = []
            newly_finished = []
            for tokens, log_sum, token_count, parent_index in retained:
                last_token = tokens[-1]
                has_finished = last_token == end_token_id or (
                    stop_with_sep and last_token == separator_token_id
                )
                candidate = (tokens, log_sum, token_count)
                if has_finished:
                    newly_finished.append(candidate)
                else:
                    next_active.append(candidate)
                    retained_parent_indices.append(parent_index)

            finished.extend(newly_finished)
            finished = sorted(
                finished, key=mean_log_probability, reverse=True
            )[:k]
            live_pool_size = len(next_active)
            past_key_values = select_cache_rows(
                output_cache, retained_parent_indices
            )
            active = next_active

            del (
                outputs,
                log_probabilities,
                branch_log_probabilities,
                branch_token_ids,
                decoder_input_ids,
                candidates,
                retained,
                output_cache,
            )

        # If the length limit is reached, incomplete live sequences remain
        # valid beam results and compete with completed sequences by the same
        # geometric-mean score.
        finished.extend(active)
        finished = sorted(
            finished, key=mean_log_probability, reverse=True
        )[:k]
        return [
            [tokens, math.exp(mean_log_probability((tokens, log_sum, count)))]
            for tokens, log_sum, count in finished
        ]
    


    def _top_k_sampling(self, s, decodervector, k, maxlen, stop_with_sep, dev, num_samples=5, temperature=1.0):
        """
        Generate sequences using top-k sampling with temperature
        Returns: list of [sequence, probability] sorted by probability
        """
        results = []
        end_token_id = self.tokenizer.vocab["<end>"]
        sep_token_id = self.tokenizer.vocab[">"]

        for _ in range(num_samples):
            current_ids = decodervector[:]  # Start with decoder input
            log_prob = 0.0
            step_count = 0

            for step in range(maxlen):
                # Prepare input tensors
                input_tensor = torch.tensor([s]).to(dev)
                decoder_tensor = torch.tensor([current_ids]).to(dev)

                # Forward pass
                with torch.no_grad():
                    outputs = self.BartNN(
                        input_ids=input_tensor,
                        decoder_input_ids=decoder_tensor,
                        return_dict=True
                    )
                    logits = outputs.logits[0, -1, :]  # Last token logits

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Get top-k tokens
                topk_probs, topk_indices = torch.topk(F.softmax(logits, dim=-1), k)

                # Renormalize top-k probabilities
                topk_probs = topk_probs / topk_probs.sum()

                # Sample from top-k
                next_token_idx = torch.multinomial(topk_probs, 1).item()
                next_token = topk_indices[next_token_idx].item()
                token_prob = topk_probs[next_token_idx].item()

                # Update log probability
                log_prob += math.log(token_prob)
                step_count += 1

                # Add token to sequence
                current_ids.append(next_token)

                # Check stopping conditions
                if next_token == end_token_id:
                    break
                if stop_with_sep and next_token == sep_token_id:
                    break

            # Calculate normalized probability (geometric mean)
            normalized_prob = math.exp(log_prob / step_count) if step_count > 0 else 0.0
            results.append([current_ids, normalized_prob])

        # Sort results by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _top_p_sampling(self, s, decodervector, p, maxlen, stop_with_sep, dev, num_samples=5, temperature=1.0):
        """
        Generate sequences using top-p (nucleus) sampling with temperature
        Returns: list of [sequence, probability] sorted by probability
        """
        results = []
        end_token_id = self.tokenizer.vocab["<end>"]
        sep_token_id = self.tokenizer.vocab[">"]

        for _ in range(num_samples):
            current_ids = decodervector[:]  # Start with decoder input
            log_prob = 0.0
            step_count = 0

            for step in range(maxlen):
                # Prepare input tensors
                input_tensor = torch.tensor([s]).to(dev)
                decoder_tensor = torch.tensor([current_ids]).to(dev)

                # Forward pass
                with torch.no_grad():
                    outputs = self.BartNN(
                        input_ids=input_tensor,
                        decoder_input_ids=decoder_tensor,
                        return_dict=True
                    )
                    logits = outputs.logits[0, -1, :]  # Last token logits

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above p
                remove_mask = cumulative_probs > p
                # Always keep at least one token
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False

                # Apply mask to sorted indices
                remove_indices = sorted_indices[remove_mask]
                probs[remove_indices] = 0

                # Renormalize probabilities
                if probs.sum() > 0:
                    probs /= probs.sum()
                else:
                    # Fallback: use the top token
                    probs = torch.zeros_like(probs)
                    probs[sorted_indices[0]] = 1.0

                # Sample next token
                next_token = torch.multinomial(probs, 1).item()
                token_prob = probs[next_token].item()

                # Update log probability
                log_prob += math.log(token_prob)
                step_count += 1

                # Add token to sequence
                current_ids.append(next_token)

                # Check stopping conditions
                if next_token == end_token_id:
                    break
                if stop_with_sep and next_token == sep_token_id:
                    break

            # Calculate normalized probability (geometric mean)
            normalized_prob = math.exp(log_prob / step_count) if step_count > 0 else 0.0
            results.append([current_ids, normalized_prob])

        # Sort results by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results

class CB_END(nn.Module):
    '''
    this api uses the output of end token
    '''
    def __init__(self, path: str, out_type: int, 
                 name: str, device: str = "cuda:0",
                 ran: int = 0, epoch_stop: int = 20):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        #self.name = absdir + "model/"+name+'.pth'
        self.name = path
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.ran = ran
        self.epoch_stop = epoch_stop
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(1024, 1)
        elif self.type > 2:
            self.linear = nn.Linear(1024, self.type)
        else:
            raise("invalid type!")
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
        self.to(self.device)
            
    def forward(self, x):
        last_hidden = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][0][-1]
        linear_out = self.linear(F.relu(last_hidden))
        if self.type == 1:
            if self.ran == 0:
                return linear_out[0]
            elif self.ran < 0:
                return torch.tanh(linear_out[0])*(-1)*self.ran
            else:
                return torch.sigmoid(linear_out[0])*self.ran
                #ref: (0,10)
        elif self.type == 2:
            return torch.sigmoid(linear_out[0])
        else:
            return torch.softmax(linear_out, dim = 0)
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.type == 1:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()
        bestval = None
        no_improvement_count = 0
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device))
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device))
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if self.epoch_stop is not None and no_improvement_count >= self.epoch_stop:
                print("No improvement in the last {} epochs. Training stopped."
                      .format(self.epoch_stop), flush=True)
                break
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out,i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)

    def predict(self, smiles_list):
        """Predict labels or values for valid SMILES strings."""
        self.eval()
        self.to(self.device)
        outputs = []
        with torch.no_grad():
            for smile in smiles_list:
                input_ids = self.tokenizer.encoder(smile)
                if len(input_ids) == 0:
                    continue
                outputs.append(self(input_ids.to(self.device)).cpu())
        if not outputs:
            return torch.empty(0)
        return torch.stack(outputs)

    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            out_tensor = out if isinstance(out, torch.Tensor) else torch.tensor(out)
            label_tensor = label if isinstance(label, torch.Tensor) else torch.tensor(label)
            return float(torch.argmax(out_tensor) == torch.argmax(label_tensor))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])
        return ans

    def transform(self, smiles_list):
        """Return the final decoder hidden state for each valid SMILES string."""
        self.eval()
        self.to(self.device)
        outputs = []
        with torch.no_grad():
            for smile in smiles_list:
                input_ids = self.tokenizer.encoder(smile)
                if len(input_ids) == 0:
                    continue
                hidden = self.BartNN(
                    input_ids=input_ids.to(self.device),
                    decoder_input_ids=input_ids.to(self.device),
                    return_dict=True,
                    output_hidden_states=True,
                ).decoder_hidden_states[-1][0][-1]
                outputs.append(hidden.cpu().numpy())
        return outputs

class CB_mul_END(nn.Module):
    '''
    this api uses the output of multi tokens at the end
    '''
    def __init__(self, path: str, name: str, device: str = "cuda:0"):
        super().__init__()
        #self.name = absdir + "model/"+name+'.pth'
        self.name = path
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.linear1 = nn.Linear(1024, 1)
        self.linear2 = nn.Linear(1024, 1)
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
            print(self.name)
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
            print(absdir + 'model/ChemBart.pth')
        else:
            print("new model")
            
    def forward(self, x):
        last_hidden1, last_hidden2 = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True,
                                                output_hidden_states=True).decoder_hidden_states[-1][0][-2:]
        linear_out1 = self.linear1(F.relu(last_hidden1)) #temperature
        linear_out2 = self.linear2(F.relu(last_hidden2)) #yield
        return torch.cat((linear_out1, linear_out2))
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestval_temp = None
        bestval_yiel = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                    continue
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                #print(out,i[1][0],flush = True)
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                    label = torch.tensor(i[1]).to(self.device)
                    loss = criterion(out,label)
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                    label = torch.tensor(i[1][1]).to(self.device)
                    loss = criterion(out[1], label)
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                    label = torch.tensor(i[1][0]).to(self.device)
                    loss = criterion(out[0], label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n train_temp_acc:{},train_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("train_yiel_acc:{},train_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            self.eval()
            cor_temp = 0.0
            cor_yiel = 0.0
            count_temp = 0
            count_yiel = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                        continue
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    out = self(inp.to(self.device))
                    out_no_grad = out.tolist()
                    if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_temp += 1
                        count_yiel += 1
                    elif type(i[1][0]) != type(1.1):
                        cor_yiel += (i[1][1] - out_no_grad[1])**2
                        count_yiel += 1
                    elif type(i[1][1]) != type(1.1):
                        cor_temp += (i[1][0] - out_no_grad[0])**2
                        count_temp += 1
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("epoch loss:{}\n val_temp_acc:{},val_temp_count:{}"
                  .format(ep_loss,cor_temp,count_temp))
            print("val_yiel_acc:{},val_yiel_count:{}"
                  .format(cor_yiel,count_yiel))
            if (bestval_temp is None) or\
                (cor_temp < bestval_temp and cor_yiel < bestval_yiel):
                bestval_temp = cor_temp
                bestval_yiel = cor_yiel
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data):
        cor_temp = 0.0
        cor_yiel = 0.0
        count_temp = 0
        count_yiel = 0
        self.eval()
        self.to(self.device)
        ans_temp = []
        ans_yiel = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                if type(i[1][0]) != type(1.1) and type(i[1][1]) != type(1.1):
                    continue
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                out_no_grad = out.tolist()
                #print(out_no_grad,flush = True)
                if type(i[1][0]) == type(1.1) and type(i[1][1]) == type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_temp += 1
                    count_yiel += 1
                    ans_temp.append([i[1][0],out_no_grad[0]])
                    ans_yiel.append([i[1][1],out_no_grad[1]])
                elif type(i[1][0]) != type(1.1):
                    cor_yiel += (i[1][1] - out_no_grad[1])**2
                    count_yiel += 1
                    ans_yiel.append([i[1][1],out_no_grad[1]])
                elif type(i[1][1]) != type(1.1):
                    cor_temp += (i[1][0] - out_no_grad[0])**2
                    count_temp += 1
                    ans_temp.append([i[1][0],out_no_grad[0]])
            cor_temp = (cor_temp/count_temp)**0.5
            cor_yiel = (cor_yiel/count_yiel)**0.5
            print("test_temp_acc:", cor_temp, "test_yiel_acc:", cor_yiel, flush=True)
        return ((cor_temp, cor_yiel), (ans_temp, ans_yiel))


class CB_mul_END_cls(nn.Module):
    """Two-head classifier using the final two decoder token representations."""

    def __init__(self, name: str, pre_model: str, device: str = "cuda:0",
                 class_counts: Tuple[int, int] = (6, 4), epoch_stop: int = 20):
        super().__init__()
        if len(class_counts) != 2 or any(count < 2 for count in class_counts):
            raise ValueError("class_counts must contain two class counts of at least 2")

        self.class_counts = tuple(class_counts)
        self.epoch_stop = epoch_stop
        self.name = name if name.endswith(".pth") else name + ".pth"
        self.save_path = self.name if os.path.dirname(self.name) else os.path.join("checkpoints", self.name)
        if pre_model.endswith(".pth") or os.path.dirname(pre_model):
            self.pre_model = pre_model
        else:
            self.pre_model = os.path.join("ChemBart_model", pre_model + ".pth")

        self.tokenizer = CBTokenizer()
        self.config = BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN = BartForConditionalGeneration(self.config)
        self.linear1 = nn.Linear(self.config.d_model, self.class_counts[0])
        self.linear2 = nn.Linear(self.config.d_model, self.class_counts[1])
        self.device = torch.device(device)

        if os.path.exists(self.save_path):
            self.load_state_dict(torch.load(self.save_path, map_location="cpu"))
            print("fine-tuned model", self.save_path)
        elif os.path.exists(self.pre_model):
            self.BartNN.load_state_dict(torch.load(self.pre_model, map_location="cpu"))
            print("pre-trained model", self.pre_model)
        else:
            print("new model")

    def _logits(self, x):
        last_hidden1, last_hidden2 = self.BartNN(
            input_ids=x,
            decoder_input_ids=x,
            return_dict=True,
            output_hidden_states=True,
        ).decoder_hidden_states[-1][0][-2:]
        return (
            self.linear1(F.relu(last_hidden1)),
            self.linear2(F.relu(last_hidden2)),
        )

    def forward(self, x):
        """Return softmax probabilities for the two mutually exclusive tasks."""
        logits1, logits2 = self._logits(x)
        return (
            torch.softmax(logits1, dim=-1),
            torch.softmax(logits2, dim=-1),
        )

    @staticmethod
    def _target_index(label, device):
        label_tensor = torch.as_tensor(label, device=device)
        if label_tensor.ndim == 0:
            return label_tensor.long()
        return torch.argmax(label_tensor).long()

    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.CrossEntropyLoss()
        bestval = None
        no_improvement_count = 0

        for epoch_index in range(epoch):
            print("epoch", epoch_index, flush=True)
            self.train()
            train_losses = [0.0, 0.0]
            train_counts = [0, 0]
            train_correct = [0, 0]

            for sample in data[0:tr]:
                labels = sample[1]
                if labels[0] is None and labels[1] is None:
                    continue
                input_ids = self.tokenizer.encoder(sample[0])
                if len(input_ids) == 0:
                    continue

                optimizer.zero_grad()
                logits = self._logits(input_ids.to(self.device))
                losses = []
                for head_index, label in enumerate(labels):
                    if label is None:
                        continue
                    target = self._target_index(label, self.device)
                    head_loss = criterion(logits[head_index], target)
                    losses.append(head_loss)
                    train_losses[head_index] += head_loss.item()
                    train_counts[head_index] += 1
                    train_correct[head_index] += int(
                        torch.argmax(logits[head_index]).item() == target.item()
                    )
                if not losses:
                    continue
                loss = sum(losses)
                loss.backward()
                optimizer.step()

            for head_index in range(2):
                count = train_counts[head_index]
                mean_loss = train_losses[head_index] / count if count else 0.0
                accuracy = train_correct[head_index] / count if count else 0.0
                print("train head {} loss: {}, accuracy: {}, count: {}"
                      .format(head_index + 1, mean_loss, accuracy, count))

            self.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for sample in data[tr:tr+val]:
                    labels = sample[1]
                    if labels[0] is None and labels[1] is None:
                        continue
                    input_ids = self.tokenizer.encoder(sample[0])
                    if len(input_ids) == 0:
                        continue
                    logits = self._logits(input_ids.to(self.device))
                    for head_index, label in enumerate(labels):
                        if label is None:
                            continue
                        target = self._target_index(label, self.device)
                        val_loss += criterion(logits[head_index], target).item()
                        val_count += 1

            mean_val_loss = val_loss / val_count if val_count else float("inf")
            print("validation loss: {}, count: {}".format(mean_val_loss, val_count), flush=True)
            if bestval is None or mean_val_loss < bestval:
                bestval = mean_val_loss
                parent = os.path.dirname(self.save_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                torch.save(self.state_dict(), self.save_path)
                print("model refreshed!", flush=True)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if self.epoch_stop is not None and no_improvement_count >= self.epoch_stop:
                print("No improvement in the last {} epochs. Training stopped."
                      .format(self.epoch_stop), flush=True)
                break

    def test(self, test_data, label_range=1):
        self.eval()
        self.to(self.device)
        labels_by_head = [[], []]
        predictions_by_head = [[], []]

        with torch.no_grad():
            for sample in test_data:
                labels = sample[1]
                if labels[0] is None and labels[1] is None:
                    continue
                input_ids = self.tokenizer.encoder(sample[0])
                if len(input_ids) == 0:
                    continue
                probabilities = self(input_ids.to(self.device))
                for head_index, label in enumerate(labels):
                    if label is None:
                        continue
                    target = self._target_index(label, torch.device("cpu")).item()
                    prediction = torch.argmax(probabilities[head_index]).item()
                    labels_by_head[head_index].append(target)
                    predictions_by_head[head_index].append(prediction)

        accuracies = []
        f1_scores = []
        for labels, predictions in zip(labels_by_head, predictions_by_head):
            if not labels:
                accuracies.append(0.0)
                f1_scores.append(0.0)
                continue
            matches = [abs(label - prediction) <= label_range
                       for label, prediction in zip(labels, predictions)]
            true_positives = sum(matches)
            errors = len(matches) - true_positives
            accuracies.append(true_positives / len(matches))
            denominator = 2 * true_positives + 2 * errors
            f1_scores.append((2 * true_positives / denominator) if denominator else 0.0)

        return {"accuracy": accuracies, "f1_score": f1_scores}


class CB_LSTM(nn.Module):
    '''
    this api uses the output of end token
    '''
    def __init__(self, out_type: int,
                 name: str, device: str = "cuda:0",
                 ran: int = 0):
        '''
        out_type:
        1: regression
            ran: if ran<0, range in [-ran,ran]
                    if ran>0, range in [0,ran]
                    if ran = 0, range in R
        2: binary classification
        n>=3: ont-hot-encoding classification with n classes
        '''
        super().__init__()
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.type = out_type
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        self.ran = ran
        self.lstm = torch.nn.LSTM(1024, 1024, num_layers=1,
                                  bias=True, bidirectional=True)
        if self.type == 1 or self.type == 2:
            self.linear = nn.Linear(2048, 1)
        elif self.type > 2:
            self.linear = nn.Linear(2048, self.type)
        else:
            raise("invalid type!")
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x):
        hidden_seq = self.BartNN(input_ids=x, decoder_input_ids=x, return_dict=True, output_hidden_states=True).decoder_hidden_states[-1][0]
        #print(hidden_seq.shape)
        _, (_, cell) = self.lstm(F.relu(hidden_seq))
        linear_out = self.linear(F.relu(cell.reshape(2048)))
        if self.type == 1:
            if self.ran == 0:
                return linear_out[0]
            elif self.ran < 0:
                return torch.tanh(linear_out[0])*(-1)*self.ran
            else:
                return torch.sigmoid(linear_out[0])*self.ran
                #ref: (0,10)
        elif self.type == 2:
            return torch.sigmoid(linear_out[0])
        else:
             return torch.softmax(linear_out, dim = 0)
    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        '''
        data: (one piece of input as smiles string, label)
        label: for regression/ bi-classification, float; for multi-classification, one-hot
        '''
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6, weight_decay=1e-6)
        if self.type == 1:
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.BCELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            ep_loss = 0.0
            cor = 0.0
            count = 0
            self.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count += 1
                out = self(inp.to(self.device))
                #print(out,i[1][0],flush = True)
                cor += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                label = torch.tensor(i[1]).to(self.device)
                loss = criterion(out,label)
                ep_loss = ep_loss + loss.item()
                loss.backward()
                optimizer.step()
            cor = self._post_proc(cor,count)
            print("epoch loss:{}, train_acc:{},train_count:{}".format(ep_loss,cor,count))
            self.eval()
            corval = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    inp = self.tokenizer.encoder(i[0])
                    if len(inp) == 0:
                        continue
                    count += 1
                    out = self(inp.to(self.device))
                    corval += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
            corval = self._post_proc(corval,count)
            print("validation_acc:",corval,",val_count:",count,flush=True)
            if (bestval is None) or\
                    (self.type == 1 and corval < bestval) or\
                    (self.type > 1 and corval > bestval):
                bestval = corval
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
    def test(self, test_data, return_detail = False):
        acc = 0.0
        self.eval()
        self.to(self.device)
        ans = []
        count = 0
        with torch.no_grad():
            for i in test_data:
                inp = self.tokenizer.encoder(i[0])
                if len(inp) == 0:
                    continue
                count+=1
                out = self(inp.to(self.device))
                acc += self._get_acc(out.item() if type(i[1]) == type(1.1) else out.tolist(),i[1])
                if return_detail:
                    ans.append([i[1],out])
            acc = self._post_proc(acc, count)
            print("test_acc:", acc, flush=True)
        return (acc, ans)
    def _get_acc(self,out,label) -> float:
        if self.type == 2:
            if (out<0.5 and label<0.5) or (out>=0.5 and label>=0.5):
                return 1.0
            else:
                return 0.0
        elif self.type == 1:
            return (out - label)**2
        else:
            return float(torch.argmax(out) == torch.argmax(label))
    def _post_proc(self,acc:float,num:int) -> float:
        acc = acc/num
        if self.type == 1:
            acc = acc**0.5
            #rmse
        return acc
    def ret_x_y_list(self,data):
        assert self.type == 1, "only for regression use"
        ans = []
        with torch.no_grad():
            for i in data:
                out = self(self.tokenizer.encoder(i[0]).to(self.device))
                ans.append([i[1],out])

class CB_MCTS():
    def __init__(self, path, dev = "cpu"):
        self.core = CB_END(path, out_type = 1, name = "CB_MCTS" , device = dev, ran = 0)
      
    def policy(self, input_list):
        outlist = torch.stack([self.core(i.to(self.core.device)) for i in input_list])
        p = torch.softmax(outlist, dim = 0)
        return p

    def value(self, smi):
        v = torch.tanh(self.core(smi.to(self.core.device)))
        return v
      
    def forward(self, smi, input_list):
        v = torch.tanh(self.core(smi.to(self.core.device)))
        outlist = torch.stack([self.core(i.to(self.core.device)) for i in input_list])
        p = torch.softmax(outlist, dim = 0)
        return v,p

    def single_train(self, data: list, epoch: int, tr: int, val: int, te: int):
        self.core.to(self.core.device)
        optimizer = torch.optim.AdamW(self.core.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = torch.nn.MSELoss()
        bestval = None
        for i in range(epoch):
            print("epoch", i, flush = True)
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            self.core.train()
            for i in data[0:tr]:
                optimizer.zero_grad()
                smi = self.tokenizer.encoder(i[0][0])
                inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                if len(smi) == 0:
                    continue
                status = 1
                for k in inputlist:
                    if len(k) == 0:
                        status = 0
                        break
                if status == 0:
                    continue
                count += 1
                v, p = self(smi = smi, input_list = inputlist)
                #print(out,i[1][0],flush = True)
                label_v = torch.tensor(i[1][0]).to(self.device)
                label_p = torch.tensor(i[1][1]).to(self.device)
                lv = criterion(v,label_v)
                lp = criterion(p,label_p)
                loss_v += lv.item()
                loss_p += lp.item()
                loss = lv + lp
                loss.backward()
                optimizer.step()
            print("loss v:{}, loss p:{},train_count:{}".format(loss_v,loss_p,count))
            self.core.eval()
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            with torch.no_grad():
                for i in data[tr:tr+val]:
                    smi = self.tokenizer.encoder(i[0][0])
                    inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                    if len(smi) == 0:
                        continue
                    status = 1
                    for k in inputlist:
                        if len(k) == 0:
                            status = 0
                            break
                    if status == 0:
                        continue
                    count += 1
                    v, p = self(smi = smi, input_list = inputlist)
                    label_v = torch.tensor(i[1][0]).to(self.device)
                    label_p = torch.tensor(i[1][1]).to(self.device)
                    lv = criterion(v,label_v)
                    lp = criterion(p,label_p)
                    loss_v += lv.item()
                    loss_p += lp.item()
            print("val loss v", loss_v, "val loss p", loss_p,",val_count:",count,flush=True)
            val_loss = loss_v + loss_p
            if (bestval is None) or val_loss<bestval:
                bestval = val_loss
                torch.save(self.core.state_dict(), self.core.name)
                print("model refreshed!", flush = True)
    def test(self, test_data):
            self.core.to(self.device)
            self.core.eval()
            loss_v = 0.0
            loss_p = 0.0
            count = 0
            with torch.no_grad():
                for i in test_data:
                    smi = self.tokenizer.encoder(i[0][0])
                    inputlist = [self.tokenizer.encoder(k) for k in i[0][1]]
                    if len(smi) == 0:
                        continue
                    status = 1
                    for k in inputlist:
                        if len(k) == 0:
                            status = 0
                            break
                    if status == 0:
                        continue
                    count += 1
                    v, p = self(smi = inp, input_list = inputlist)
                    label_v = torch.tensor(i[1][0]).to(self.device)
                    label_p = torch.tensor(i[1][1]).to(self.device)
                    lv = criterion(v,label_v)
                    lp = criterion(p,label_p)
                    loss_v += lv.item()
                    loss_p += lp.item()
            print("loss v", loss_v, "loss p", loss_p,",count:",count,flush=True)

class CB_Regression(nn.Module):
    class RegData():
        def __init__(self, data, tokenizer, maxlen = 1024):
            self.data = data
            self.tokenizer = tokenizer
            self.maxlen = maxlen
        def __getitem__(self, index):
            return self.tokenizer.encoder(self.data[index][0], alllen = self.maxlen, no0mode = False), self.data[index][1]
        def __len__(self):
            return len(self.data)
        def shuffle(self):
            random.shuffle(self.data)
    class DataLoader():
        def __init__(self, data, batch_size, shuffle = True):
            if shuffle:
                data.shuffle()
            self.data = data
            self.batch_size = batch_size
            self.id = 0
            self.len = len(data)
        def __iter__(self):
            return self
        def __next__(self):
            if self.id < self.len:
                count = 0
                x = []
                msk = []
                lab = []
                while (count < self.batch_size and self.id < self.len):
                    inp, label = self.data[self.id]
                    if len(inp) == 0:
                        continue
                    x.append(inp["input_ids"])
                    msk.append(inp["attention_mask"])
                    lab.append(label)
                    self.id += 1
                    count += 1
                return ((torch.stack(x),torch.stack(msk)),torch.tensor(lab))
            else:
                raise StopIteration
    def __init__(self, name: str, label_num: int, device: str,
                 bart_grad: bool = True, epoch_stop: int = 20):
        super().__init__()
        self.label_num = label_num
        self.bart_grad = bart_grad
        self.epoch_stop = epoch_stop
        self.name = absdir + "model/"+name+'.pth'
        self.tokenizer = CBTokenizer()
        self.config=BartConfig.from_pretrained(absdir + "config.json")
        self.BartNN=BartForConditionalGeneration(self.config)
        if not self.bart_grad:
            for parameter in self.BartNN.parameters():
                parameter.requires_grad = False
        self.linear_heads = nn.ModuleList([nn.Linear(1024, 1) for i in range(label_num)])
        self.device = torch.device(device)
        if os.path.exists(self.name):
            self.load_state_dict(torch.load(self.name,map_location='cpu'))
            print("fine-tuned model")
        elif os.path.exists(absdir + 'model/ChemBart.pth'):
            self.BartNN.load_state_dict(torch.load(absdir + 'model/ChemBart.pth',map_location='cpu'))
            print("pre-trained model")
        else:
            print("new model")
            
    def forward(self, x, attention_mask = None):
        hidden_list = torch.stack([i[-self.label_num:] for i in \
                self.BartNN(input_ids=x, decoder_input_ids=x,
                attention_mask = attention_mask, decoder_attention_mask = attention_mask,
                return_dict=True, output_hidden_states=True).decoder_hidden_states[-1]\
                ])
        linear_out = torch.stack([torch.cat([self.linear_heads[j](hidden_list[i][j])\
                        for j in range(self.label_num)])\
                        for i in range(len(hidden_list))])
        return linear_out

    def fit(self, data: list, epoch: int, batch_size:int, tr: int, val: int, te: int,
            id_maxlen: int = 1024, *, lr: float = 1e-6, weight_decay: float = 1e-6):
        self.to(self.device)
        parameters = self.parameters() if self.bart_grad else self.linear_heads.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()
        bestloss = None
        no_improvement_count = 0
        tr_dataset = self.RegData(data[0:tr], self.tokenizer, maxlen = id_maxlen)
        val_dataset = self.RegData(data[tr:tr+val], self.tokenizer, maxlen = id_maxlen)
        for e in range(epoch):
            dataloader = self.DataLoader(tr_dataset, batch_size = batch_size)
            print("epoch", e, flush = True)
            ep_loss = 0.0
            reslist = [[] for k in range(self.label_num)]
            self.train()
            print("train")
            for inp, label in dataloader:
                optimizer.zero_grad()
                out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                out_no_grad = out.tolist()
                for outcome in range(len(out_no_grad)):
                    for item in range(self.label_num):
                        if type(label[outcome][item]) != type(torch.tensor(1.1)):
                            label[outcome][item] = 0.0
                            out[outcome][item] = 0.0
                            out_no_grad[outcome][item] = 0.0
                            continue
                        reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
                loss = criterion(out, label.to(self.device))
                ep_loss += loss.item()
                loss.backward()
                optimizer.step()
            for idx in range(len(reslist)):
                print("regression task", idx, ": rmse =", self.RMSE(reslist[idx]))
            print("train loss:", ep_loss, flush = True)
            self.eval()
            print("validation")
            reslist = [[] for k in range(self.label_num)]
            dataloader = self.DataLoader(val_dataset, batch_size = batch_size)
            with torch.no_grad():
                for inp, label in dataloader:
                    out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                    out_no_grad = out.tolist()
                    for outcome in range(len(out_no_grad)):
                        for item in range(self.label_num):
                            if type(label[outcome][item]) != type(torch.tensor(1.1)):
                                continue
                            reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
            RMSE_sum = 0.0
            for idx in range(len(reslist)):
                item_RMSE = self.RMSE(reslist[idx])
                RMSE_sum += item_RMSE
                print("regression task", idx, ": rmse =", item_RMSE)
            if (bestloss is None) or\
                (RMSE_sum < bestloss):
                bestloss = RMSE_sum
                torch.save(self.state_dict(), self.name)
                print("model refreshed!", flush = True)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if self.epoch_stop is not None and no_improvement_count >= self.epoch_stop:
                print("No improvement in the last {} epochs. Training stopped."
                      .format(self.epoch_stop), flush=True)
                break
    def RMSE(self, l):
        s = 0.0
        for x,y in l:
            s += (x-y)**2
        s /= len(l)
        s = s**0.5
        return s
    def test(self, test_data, batch_size, id_maxlen = 1024):
        self.eval()
        self.to(self.device)
        dataset = self.RegData(test_data, self.tokenizer, maxlen = id_maxlen)
        reslist = [[] for k in range(self.label_num)]
        dataloader = self.DataLoader(dataset, batch_size = batch_size)
        with torch.no_grad():
            for inp, label in dataloader:
                out = self(inp[0].to(self.device),attention_mask = inp[1].to(self.device))
                out_no_grad = out.tolist()
                for outcome in range(len(out_no_grad)):
                    for item in range(self.label_num):
                        if type(label[outcome][item]) != type(torch.tensor(1.1)):
                            continue
                        reslist[item].append((out_no_grad[outcome][item],label[outcome][item].item()))
        RMSE_list = []
        for idx in range(len(reslist)):
            item_RMSE = self.RMSE(reslist[idx])
            RMSE_list.append(item_RMSE)
        return (RMSE_list, reslist)
