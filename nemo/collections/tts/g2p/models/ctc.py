# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import string
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo.collections.tts.g2p.data.ctc import CTCG2PBPEDataset, LengthBucketingBatchSampler
from nemo.collections.tts.models.base import G2PModel
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LengthsType, LossType, NeuralType, TokenIndex
from nemo.utils import logging

try:
    from nemo.collections.asr.losses.ctc import CTCLoss
    from nemo.collections.asr.metrics.wer import WER
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.collections.asr.parts.mixins import ASRBPEMixin
    from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig

    ASR_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    ASR_AVAILABLE = False


__all__ = ['CTCG2PModel']


@dataclass
class CTCG2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class CTCG2PModel(G2PModel, ASRBPEMixin, Exportable):
    """
    CTC-based grapheme-to-phoneme model.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        self.mode = cfg.model_name.lower()

        self.supported_modes = ["byt5", "conformer_bpe"]
        if self.mode not in self.supported_modes:
            raise ValueError(f"{self.mode} is not supported, choose from {self.supported_modes}")

        # Setup phoneme tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Setup grapheme tokenizer
        self.tokenizer_grapheme = self.setup_grapheme_tokenizer(cfg)

        # Initialize vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))
        self.vocabulary = cfg.decoder.vocabulary
        self.labels_tkn2id = {l: i for i, l in enumerate(self.vocabulary)}
        self.labels_id2tkn = {i: l for i, l in enumerate(self.vocabulary)}

        super().__init__(cfg, trainer)

        self._setup_encoder()
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )

        # Setup decoding objects
        decoding_cfg = self.cfg.get('decoding', None)

        # In case decoding config not found, use default config
        if decoding_cfg is None:
            decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg):
                self.cfg.decoding = decoding_cfg

        self.decoding = CTCBPEDecoding(self.cfg.decoding, tokenizer=self.tokenizer)

        self.wer = WER(
            decoding=self.decoding,
            use_cer=False,
            log_prediction=False,
            dist_sync_on_step=True,
        )
        self.per = WER(
            decoding=self.decoding,
            use_cer=True,
            log_prediction=False,
            dist_sync_on_step=True,
        )

    def setup_grapheme_tokenizer(self, cfg):
        """Initialized grapheme tokenizer"""

        if self.mode == "byt5":
            # Load appropriate tokenizer from HuggingFace
            grapheme_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_grapheme.pretrained)
            self.max_source_len = cfg.get("max_source_len", grapheme_tokenizer.model_max_length)
            self.max_target_len = cfg.get("max_target_len", grapheme_tokenizer.model_max_length)

            # TODO store byt5 vocab file
        elif self.mode == "conformer_bpe":
            # 输入（grapheme）词表的单一真源 = 预处理产出的 grapheme_vocab.txt（含该语言全部字符，
            # 例如德语 ä ö ü ß、法语重音字母、连字符 - 等）。优先使用它，让"加一种语言"变成纯数据改动、
            # 不再改代码。仅当既没显式 tokenizer_grapheme.vocab_file、也无法从 tokenizer.dir 推导出
            # grapheme_vocab.txt 时，才回退到旧的写死 ASCII（仅够英文）逻辑。
            src_grapheme_vocab = self._resolve_grapheme_vocab_source(cfg)
            if src_grapheme_vocab is not None:
                vocab_file = self._build_char_vocab_file_from_grapheme_vocab(src_grapheme_vocab)
                logging.info(
                    f"conformer_bpe grapheme tokenizer built from data-driven grapheme vocab "
                    f"{src_grapheme_vocab} (covers all language-specific characters)."
                )
            else:
                grapheme_unk_token = (
                    cfg.tokenizer_grapheme.unk_token if cfg.tokenizer_grapheme.unk_token is not None else ""
                )
                chars = string.ascii_lowercase + grapheme_unk_token + " " + "'"

                if not cfg.tokenizer_grapheme.do_lower:
                    chars += string.ascii_uppercase

                if cfg.tokenizer_grapheme.add_punctuation:
                    punctuation_marks = string.punctuation.replace('"', "").replace("\\", "").replace("'", "")
                    chars += punctuation_marks

                vocab_file = os.path.join(tempfile.mkdtemp(prefix="g2p_char_vocab_"), "char_vocab.txt")
                with open(vocab_file, "w", encoding="utf-8") as f:
                    [f.write(f'"{ch}"\n') for ch in chars]
                    f.write('"\\""\n')  # add " to the vocab
                logging.warning(
                    "conformer_bpe grapheme tokenizer fell back to the hardcoded ASCII inventory "
                    "(ascii letters + space + apostrophe). This only covers English: any non-ASCII "
                    "letter (ä ö ü ß, accents, ...) will be dropped/unk at train AND serve time. "
                    "Provide model.tokenizer_grapheme.vocab_file (or place grapheme_vocab.txt next to "
                    "model.tokenizer.dir) to use the language's real grapheme inventory."
                )

            self.register_artifact("tokenizer_grapheme.vocab_file", vocab_file)
            grapheme_tokenizer = instantiate(cfg.tokenizer_grapheme.dataset, vocab_file=vocab_file)
            self.max_source_len = cfg.get("max_source_len", 512)
            self.max_target_len = cfg.get("max_target_len", 512)
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")
        return grapheme_tokenizer

    # Grapheme (input) special tokens mirrored from the preprocessing script's SPECIAL_TOKENS
    # (examples/dataset/preprocess_ipa_childes_split.py). They occupy the first ids so that
    # <pad> == 0 lines up with the input embedding's padding_idx=0.
    _GRAPHEME_SPECIAL_TOKENS = {"<pad>": "pad_token", "<unk>": "unk_token"}

    def _resolve_grapheme_vocab_source(self, cfg) -> Optional[str]:
        """Return the path to a data-driven grapheme vocab (one token per line: <pad>, <unk>,
        then one character per line), or ``None`` to fall back to the hardcoded ASCII inventory.

        Priority:
          1. ``cfg.tokenizer_grapheme.vocab_file`` if set (explicit override);
          2. ``<cfg.tokenizer.dir>/grapheme_vocab.txt`` if it exists (co-located with the phoneme
             ``vocab.txt`` the same preprocessing run writes) — so a single ``tokenizer.dir`` per
             language wires both input and output vocabs.
        """
        explicit = cfg.tokenizer_grapheme.get("vocab_file", None)
        if explicit:
            p = os.path.expanduser(str(explicit))
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"model.tokenizer_grapheme.vocab_file={explicit!r} was set but does not exist. "
                    f"Point it to the preprocessing-produced grapheme_vocab.txt, or unset it to fall "
                    f"back to the ASCII inventory."
                )
            return p

        tok_dir = cfg.get("tokenizer", {}).get("dir", None) if cfg.get("tokenizer", None) else None
        if tok_dir:
            candidate = os.path.join(os.path.expanduser(str(tok_dir)), "grapheme_vocab.txt")
            if os.path.isfile(candidate):
                return candidate
        return None

    def _build_char_vocab_file_from_grapheme_vocab(self, src_path: str) -> str:
        """Convert a preprocessing ``grapheme_vocab.txt`` (plain one-token-per-line, with <pad>/<unk>
        first and a literal space as its own line) into the on-disk format ``CharTokenizer`` expects:
        an optional first-line JSON of special tokens, then one Python char-literal per line.

        Order is preserved so token ids match ``grapheme_vocab.txt`` exactly (``<pad>``==0, ``<unk>``==1,
        then the characters), which keeps the input embedding's ``padding_idx=0`` valid and makes the
        exported ``grapheme_vocab`` / ``grapheme_unk_id`` line up with inference.
        """
        with open(src_path, "r", encoding="utf-8") as f:
            # Keep the trailing-space line intact: only strip the newline, nothing else.
            raw_tokens = [line.rstrip("\n") for line in f]
        # Drop only fully empty lines (never the single-space grapheme, which is "\n" -> " ").
        tokens = [t for t in raw_tokens if t != ""]

        specials: Dict[str, str] = {}
        chars: List[str] = []
        for tok in tokens:
            if tok in self._GRAPHEME_SPECIAL_TOKENS:
                specials[self._GRAPHEME_SPECIAL_TOKENS[tok]] = tok
                continue
            if len(tok) != 1:
                raise ValueError(
                    f"grapheme vocab {src_path!r} has a multi-character entry {tok!r} that is neither "
                    f"<pad> nor <unk>. The grapheme tokenizer is character-level; fix the preprocessing "
                    f"output or the vocab file."
                )
            chars.append(tok)

        out_dir = tempfile.mkdtemp(prefix="g2p_char_vocab_")
        out_path = os.path.join(out_dir, "char_vocab.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            if specials:
                # CharTokenizer assigns special ids first in dict order; force pad before unk.
                ordered = {}
                for name in ("pad_token", "unk_token"):
                    if name in specials:
                        ordered[name] = specials[name]
                f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
            for ch in chars:
                if ch == '"':
                    f.write('"\\""\n')  # the double-quote character
                elif ch == "\\":
                    f.write('"\\\\"\n')  # the backslash character
                else:
                    f.write(f'"{ch}"\n')
        return out_path

    def _setup_encoder(self):
        if self.mode == "byt5":
            config = AutoConfig.from_pretrained(self._cfg.tokenizer_grapheme.pretrained)
            if self._cfg.encoder.dropout is not None:
                config.dropout_rate = self._cfg.encoder.dropout
                print(f"\nDROPOUT: {config.dropout_rate}")
            self.encoder = AutoModel.from_pretrained(self._cfg.encoder.transformer, config=config).encoder
            # add encoder hidden dim size to the config
            if self.cfg.decoder.feat_in is None:
                self._cfg.decoder.feat_in = self.encoder.config.d_model
        elif self.mode == "conformer_bpe":
            # The input embedding is indexed by GRAPHEME ids (forward() does self.embedding(input_ids)
            # where input_ids come from tokenizer_grapheme), so it must be sized by the grapheme vocab,
            # NOT the phoneme vocab. Sizing it by the phoneme vocab was the reason a phoneme-only
            # vocab.txt used to crash with an index-out-of-range (and forced graphemes to be merged
            # into vocab.txt as a workaround). Decoupling here lets the output vocab stay pure phonemes.
            self.embedding = torch.nn.Embedding(
                embedding_dim=self._cfg.embedding.d_model,
                num_embeddings=self.tokenizer_grapheme.vocab_size,
                padding_idx=0,
            )
            self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)
            with open_dict(self._cfg):
                if "feat_in" not in self._cfg.decoder or (
                    not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
                ):
                    self._cfg.decoder.feat_in = self.encoder._feat_out
                if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                    raise ValueError("param feat_in of the decoder's config is not set!")
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")

    # @typecheck()
    def forward(self, input_ids, attention_mask, input_len):
        if self.mode == "byt5":
            encoded_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_len = input_len
            # encoded_input = [B, seq_len, hid_dim]
            # swap seq_len and hid_dim dimensions to get [B, hid_dim, seq_len]
            encoded_input = encoded_input.transpose(1, 2)
        elif self.mode == "conformer_bpe":
            input_embedding = self.embedding(input_ids)
            input_embedding = input_embedding.transpose(1, 2)
            encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)
        else:
            raise ValueError(f"{self.mode} is not supported. Choose from {self.supported_modes}")

        log_probs = self.decoder(encoder_output=encoded_input)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, greedy_predictions, encoded_len

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )

        loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, dataloader_idx=0, split="val"):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )

        self.wer.update(
            predictions=log_probs, targets=targets, targets_lengths=target_lengths, predictions_lengths=encoded_len
        )
        wer, wer_num, wer_denom = self.wer.compute()
        self.wer.reset()

        self.per.update(
            predictions=log_probs, targets=targets, targets_lengths=target_lengths, predictions_lengths=encoded_len
        )
        per, per_num, per_denom = self.per.compute()
        self.per.reset()

        self.log(f"{split}_loss", val_loss)
        loss = {
            f"{split}_loss": val_loss,
            f"{split}_wer_num": wer_num,
            f"{split}_wer_denom": wer_denom,
            f"{split}_wer": wer,
            f"{split}_per_num": per_num,
            f"{split}_per_denom": per_denom,
            f"{split}_per": per,
        }

        if split == 'val':
            if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
                self.validation_step_outputs[dataloader_idx].append(loss)
            else:
                self.validation_step_outputs.append(loss)
        elif split == 'test':
            if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
                self.test_step_outputs[dataloader_idx].append(loss)
            else:
                self.test_step_outputs.append(loss)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, dataloader_idx, split="test")

    def multi_validation_epoch_end(self, outputs, dataloader_idx=0, split="val"):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        avg_loss = torch.stack([x[f"{split}_loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", avg_loss, prog_bar=True)

        wer_num = torch.stack([x[f"{split}_wer_num"] for x in outputs]).sum()
        wer_denom = torch.stack([x[f"{split}_wer_denom"] for x in outputs]).sum()
        wer = wer_num / wer_denom

        per_num = torch.stack([x[f"{split}_per_num"] for x in outputs]).sum()
        per_denom = torch.stack([x[f"{split}_per_denom"] for x in outputs]).sum()
        per = per_num / per_denom

        if split == "test":
            dataloader_name = self._test_names[dataloader_idx].upper()
        else:
            dataloader_name = self._validation_names[dataloader_idx].upper()

        self.log(f"{split}_wer", wer)
        self.log(f"{split}_per", per)

        self.log(f"{split}_per", per)
        # to save all PER values for each dataset in WANDB
        self.log(f"{split}_per_{dataloader_name}", per)

        logging.info(f"PER: {per * 100}% {dataloader_name}")
        logging.info(f"WER: {wer * 100}% {dataloader_name}")

    def multi_test_epoch_end(self, outputs, dataloader_idx=0):
        self.multi_validation_epoch_end(outputs, dataloader_idx, split="test")

    def _setup_infer_dataloader(self, cfg: DictConfig) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.
        Returns:
            A pytorch DataLoader.
        """
        dataset = CTCG2PBPEDataset(
            manifest_filepath=cfg.manifest_filepath,
            grapheme_field=cfg.grapheme_field,
            tokenizer_graphemes=self.tokenizer_grapheme,
            tokenizer_phonemes=self.tokenizer,
            do_lower=self._cfg.tokenizer_grapheme.do_lower,
            labels=self.vocabulary,
            max_source_len=self._cfg.max_source_len,
            with_labels=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

    @torch.no_grad()
    def _infer(
        self,
        config: DictConfig,
    ) -> List[int]:
        """
        Runs model inference.

        Args:
            Config: configuration file to set up DataLoader
        Returns:
            all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            infer_datalayer = self._setup_infer_dataloader(config)

            for batch in infer_datalayer:
                input_ids, attention_mask, input_len = batch
                log_probs, greedy_predictions, encoded_len = self.forward(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask if attention_mask is None else attention_mask.to(device),
                    input_len=input_len.to(device),
                )

                preds_hyps = self.decoding.ctc_decoder_predictions_tensor(
                    log_probs, decoder_lengths=encoded_len, return_hypotheses=False
                )
                preds_str = [hyp.text for hyp in preds_hyps]
                all_preds.extend(preds_str)

                del greedy_predictions
                del log_probs
                del batch
                del input_len
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg: DictConfig, name: str):
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        if not os.path.exists(cfg.manifest_filepath):
            raise ValueError(f"{cfg.dataset.manifest_filepath} not found")

        dataset = instantiate(
            cfg.dataset,
            manifest_filepath=cfg.manifest_filepath,
            phoneme_field=cfg.dataset.phoneme_field,
            grapheme_field=cfg.dataset.grapheme_field,
            tokenizer_graphemes=self.tokenizer_grapheme,
            do_lower=self._cfg.tokenizer_grapheme.do_lower,
            tokenizer_phonemes=self.tokenizer,
            labels=self.vocabulary,
            max_source_len=self.max_source_len,
            with_labels=True,
        )

        # Optional length bucketing: groups similar-length samples per batch to
        # cut padding waste in the Conformer encoder. Big single-GPU speedup on
        # variable-length G2P data. Enable via cfg.dataloader_params.use_length_bucketing=true
        dataloader_params = dict(cfg.dataloader_params)
        use_length_bucketing = dataloader_params.pop("use_length_bucketing", False)
        if use_length_bucketing:
            batch_size = dataloader_params.pop("batch_size")
            shuffle = dataloader_params.pop("shuffle", False)
            drop_last = dataloader_params.pop("drop_last", False)
            batch_sampler = LengthBucketingBatchSampler(
                lengths=dataset.get_lengths(),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            logging.info(f"[{name}] Length bucketing enabled (batch_size={batch_size}, shuffle={shuffle}).")
            return torch.utils.data.DataLoader(
                dataset, collate_fn=dataset.collate_fn, batch_sampler=batch_sampler, **dataloader_params
            )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)

    def setup_training_data(self, cfg: DictConfig):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                "Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if not val_data_config or val_data_config.manifest_filepath is None:
            self._validation_dl = None
            return
        super().setup_multiple_validation_data(val_data_config)

    def setup_multiple_test_data(self, test_data_config: Union[DictConfig, Dict] = None):
        if not test_data_config or test_data_config.manifest_filepath is None:
            self._test_dl = None
            return
        super().setup_multiple_test_data(test_data_config)

    def setup_validation_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                "Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="val")

    def setup_test_data(self, cfg: Optional[DictConfig]):
        if not cfg or cfg.manifest_filepath is None:
            logging.info(
                "Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer

    @property
    def per(self):
        return self._per

    @per.setter
    def per(self, per):
        self._per = per

    # Methods for model exportability
    def _prepare_for_export(self, **kwargs):
        super()._prepare_for_export(**kwargs)

        # Define input_types and output_types as required by export()
        self._input_types = {
            "input_ids": NeuralType(('B', 'T'), TokenIndex()),
            "input_len": NeuralType(tuple('B'), LengthsType()),
        }
        self._output_types = {
            # "preds_str": NeuralType(('B', 'T'), LabelsType()),
            "log_probs": NeuralType(('B', 'T'), LossType()),
            "encoded_len": NeuralType(('B', 'T'), LengthsType()),
        }

    def _export_teardown(self):
        self._input_types = self._output_types = None

    @property
    def input_types(self):
        return self._input_types

    @property
    def output_types(self):
        return self._output_types

    def input_example(self, max_batch=1, max_dim=44):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        # par = next(self.fastpitch.parameters())
        sentence = "Kupil sem si bicikel in mu zamenjal stol."
        input_ids = [self.tokenizer_grapheme.text_to_ids(sentence)]
        input_len = [len(entry) for entry in input_ids]
        max_len = max(input_len)
        input_ids = [entry + [0] * (max_len - entry_len) for entry, entry_len in zip(input_ids, input_len)]
        inputs = (torch.tensor(input_ids).to(self.device), torch.tensor(input_len).to(self.device))
        return inputs

    def forward_for_export(self, input_ids, input_len):
        input_embedding = self.embedding(input_ids)
        input_embedding = input_embedding.transpose(1, 2)
        encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)

        log_probs = self.decoder(encoder_output=encoded_input)
        return (log_probs, encoded_len)
        # preds_str, _ = self.decoding.ctc_decoder_predictions_tensor(
        #    log_probs, decoder_lengths=encoded_len, return_hypotheses=True
        # )
        # results = [h.y_sequence for h in preds_str]

        # return tuple(results)
