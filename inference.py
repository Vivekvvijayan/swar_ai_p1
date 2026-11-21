import json
import logging
from collections import OrderedDict
import numpy as np
import torch
import torchaudio
from models.RagaNet import ResNetRagaClassifier
from utils import logging_utils

logging_utils.config_logger()


class Evaluator:
    def __init__(self, params):
        self.params = params
        self.device = self.params.device

        # Load raga labels directly from metadata
        with open(self.params.num_files_per_raga_path, "r", encoding="utf-8") as f:
            num_files_per_raga = json.load(f)

        self.raga_list = list(num_files_per_raga.keys())[:self.params.num_classes]
        self.label_list = list(range(len(self.raga_list)))
        self.raga2label = dict(zip(self.raga_list, self.label_list))

        # Initialize model
        self.model = ResNetRagaClassifier(params).to(self.device)

        # Load best checkpoint
        checkpoint_path = getattr(self.params, "best_checkpoint_path", "ckpts/resnet_0.7/150classes_alldata_cliplength30/training_checkpoints/best_ckpt.tar")
        logging.info("Loading checkpoint from %s", checkpoint_path)
        self.restore_checkpoint(checkpoint_path)
        self.model.eval()

    def normalize(self, audio):
        return (audio - torch.mean(audio, dim=1, keepdim=True)) / (torch.std(audio, dim=1, keepdim=True) + 1e-5)

    def pad_audio(self, audio):
        pad = (0, self.params.sample_rate * self.params.clip_length - audio.shape[1])
        return torch.nn.functional.pad(audio, pad=pad, value=0)

    def inference(self, k, audio):
        if audio is None:
            return {}

        sample_rate, audio_clip = audio

        # Ensure stereo channel shape (2, N)
        if len(audio_clip.shape) == 1:
            audio_clip = torch.tensor(audio_clip).unsqueeze(0).repeat(2, 1).to(torch.float32)
        else:
            audio_clip = torch.tensor(audio_clip).T.to(torch.float32)

        # Resample audio clip to target sample rate (8000 Hz)
        if sample_rate != self.params.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.params.sample_rate)
            audio_clip = resample(audio_clip)

        # Normalize the audio clip
        if self.params.normalize:
            audio_clip = self.normalize(audio_clip)

        # Pad audio with zeros if shorter than clip_length (30s)
        target_length = self.params.sample_rate * self.params.clip_length
        if audio_clip.size()[1] < target_length:
            audio_clip = self.pad_audio(audio_clip)

        assert not torch.any(torch.isnan(audio_clip))
        audio_clip = audio_clip.to(self.device)

        with torch.no_grad():
            length = audio_clip.shape[1]
            train_length = target_length

            pred_probs = torch.zeros((self.params.num_classes,)).to(self.device)

            # Loop over clip_length segments and accumulate probability distribution
            num_clips = max(1, int(np.floor(length / train_length)))
            for i in range(num_clips):
                clip = audio_clip[:, i * train_length : (i + 1) * train_length].unsqueeze(0)
                pred_distribution = self.model(clip).reshape(-1, self.params.num_classes)
                exp_dist = torch.exp(pred_distribution)
                pred_probs += (1.0 / num_clips) * (exp_dist / exp_dist.sum(axis=1, keepdim=True))[0]

        pred_probs, labels = pred_probs.sort(descending=True)
        top_k = int(min(k, self.params.num_classes))
        pred_probs_topk = pred_probs[:top_k]
        pred_ragas_topk = [self.raga_list[label.item()] for label in labels[:top_k]]
        result = {raga: prob.item() for raga, prob in zip(pred_ragas_topk, pred_probs_topk)}
        return result

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except Exception:
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:] if key.startswith("module.") else key
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
