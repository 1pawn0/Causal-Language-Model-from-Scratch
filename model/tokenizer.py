import os
import torch
import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.Load(model_path)
        self.n_words: int = self.sp_processor.GetPieceSize()
        self.bos_id: int = self.sp_processor.bos_id()
        self.eos_id: int = self.sp_processor.eos_id()
        self.pad_id: int = self.sp_processor.pad_id()

    def encode(self, text: str) -> torch.Tensor:
        assert isinstance(text, str)
        token_ids: torch.Tensor = torch.tensor(
            self.sp_processor.EncodeAsIds(text), dtype=torch.long
        )
        return token_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        text: str = self.sp_processor.DecodeIds(token_ids.tolist())
        return text
