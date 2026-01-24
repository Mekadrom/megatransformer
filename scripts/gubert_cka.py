import os
import sys

import torch
from tqdm import tqdm

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pretrain_gubert import GuBERTDataCollator, GuBERTShardedDataset, create_model
from utils.speaker_encoder import get_speaker_encoder


train_dataset = GuBERTShardedDataset('./cached_datasets/gubert_ctc_train', mode="ctc")
collator = GuBERTDataCollator(n_mels=80, max_mel_frames=1875, mode="ctc")

model = create_model(
    config_name="tiny_deep",
    num_speakers=train_dataset.num_speakers,
    vocab_size=train_dataset.vocab.vocab_size,
    n_mels=80,
    # CTC upsampling (relaxes CTC length constraint)
    ctc_upsample_factor=2,
    # Dropout regularization
    conv_dropout=0.05,
    feature_dropout=0.2,
    head_dropout=0.3,
    attention_head_drop=0.1,
    # Architectural options
    use_rotary_embedding=True,
    use_conformer_conv=True,
    conformer_kernel_size=31,
    use_macaron=True,
    activation="swiglu",
    # Speaker normalization (strips speaker statistics)
    use_instance_norm=False,
    instance_norm_affine=False,
    # Speaker classifier pooling strategy
    speaker_pooling="attentive_statistics",
).to('cuda')

model.load_state_dict(torch.load('runs/gubert/tiny_deep_0_4/checkpoint-71000/pytorch_model.bin'))

model.eval()

speaker_encoder = get_speaker_encoder('ecapa_tdnn', device='cuda')

speaker_encoder.eval()

batch_size = 16
max_samples = 5000

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collator,
)

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Linear CKA between two representation matrices.
    
    Args:
        X: [n_samples, features_x]
        Y: [n_samples, features_y]
    
    Returns:
        CKA similarity score in [0, 1]
    """
    # Center both
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Gram matrices
    XX = X @ X.T
    YY = Y @ Y.T
    
    # HSIC estimates (Frobenius inner products)
    hsic_xy = (XX * YY).sum()
    hsic_xx = (XX * XX).sum()
    hsic_yy = (YY * YY).sum()
    
    return (hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))).item()


# Accumulate over many batches
all_layer_4 = []
all_layer_6 = []
all_layer_8 = []
all_layer_10 = []
all_layer_12 = []
all_ctc = []
all_speaker_emb = []

for batch in tqdm(dataloader):
    mel_specs = batch['mel_specs'].to('cuda')
    lengths = batch['mel_lengths'].to('cuda')


    with torch.no_grad():
        # Run GuBERT, collect activations
        output = model(mel_specs, lengths, return_all_hiddens=True)
        speaker_embedding = speaker_encoder(mel_spec=mel_specs, lengths=lengths)
        
        l4: torch.Tensor = output['all_hiddens'][4]    # [B, T, D]
        l6: torch.Tensor = output['all_hiddens'][6]    # [B, T, D]
        l8: torch.Tensor = output['all_hiddens'][8]    # [B, T, D]
        l10: torch.Tensor = output['all_hiddens'][10]  # [B, T, D]
        l12: torch.Tensor = output['all_hiddens'][12]  # [B, T, D]
        ctc: torch.Tensor = output['asr_logits']

        B, T, _ = l4.shape
        speaker_embedding_expanded = speaker_embedding.unsqueeze(1).expand(B, T, -1).flatten(0, 1)

        # Flatten time into batch: [B, T, D] → [B*T, D]
        all_layer_4.append(l4.flatten(0, 1))
        all_layer_6.append(l6.flatten(0, 1))
        all_layer_8.append(l8.flatten(0, 1))
        all_layer_10.append(l10.flatten(0, 1))
        all_layer_12.append(l12.flatten(0, 1))
        all_ctc.append(ctc.flatten(0, 1))
        all_speaker_emb.append(speaker_embedding_expanded)
    
    if len(all_layer_4) * batch_size > max_samples:  # Enough samples
        break

# Concat and compute
layer_4 = torch.cat(all_layer_4)[:max_samples]   # Cap for memory
layer_6 = torch.cat(all_layer_6)[:max_samples]
layer_8 = torch.cat(all_layer_8)[:max_samples]
layer_10 = torch.cat(all_layer_10)[:max_samples]
layer_12 = torch.cat(all_layer_12)[:max_samples]
ctc = torch.cat(all_ctc)[:max_samples]
speaker_emb = torch.cat(all_speaker_emb)[:max_samples]

cka_4_ctc = linear_cka(layer_4, ctc)
cka_6_ctc = linear_cka(layer_6, ctc)
cka_8_ctc = linear_cka(layer_8, ctc)
cka_10_ctc = linear_cka(layer_10, ctc)
cka_12_ctc = linear_cka(layer_12, ctc)

cka_4_6 = linear_cka(layer_4, layer_6)
cka_4_8 = linear_cka(layer_4, layer_8)
cka_4_10 = linear_cka(layer_4, layer_10)
cka_4_12 = linear_cka(layer_4, layer_12)
cka_6_8 = linear_cka(layer_6, layer_8)
cka_10_12 = linear_cka(layer_10, layer_12)

cka_4_speaker = linear_cka(layer_4, speaker_emb)
cka_6_speaker = linear_cka(layer_6, speaker_emb)
cka_8_speaker = linear_cka(layer_8, speaker_emb)
cka_10_speaker = linear_cka(layer_10, speaker_emb)
cka_12_speaker = linear_cka(layer_12, speaker_emb)

print(f"Layer 4  ↔ CTC: {cka_4_ctc:.3f}")
print(f"Layer 6  ↔ CTC: {cka_6_ctc:.3f}")
print(f"Layer 8  ↔ CTC: {cka_8_ctc:.3f}")
print(f"Layer 10 ↔ CTC: {cka_10_ctc:.3f}")
print(f"Layer 12 ↔ CTC: {cka_12_ctc:.3f}")


print(f"Layer 4  ↔ Layer 6: {cka_4_6:.3f}")
print(f"Layer 4  ↔ Layer 8: {cka_4_8:.3f}")
print(f"Layer 4  ↔ Layer 10: {cka_4_10:.3f}")
print(f"Layer 4  ↔ Layer 12: {cka_4_12:.3f}")
print(f"Layer 6  ↔ Layer 8: {cka_6_8:.3f}")
print(f"Layer 10 ↔ Layer 12: {cka_10_12:.3f}")

print(f"Layer 4  ↔ Speaker: {cka_4_speaker:.3f}")
print(f"Layer 6  ↔ Speaker: {cka_6_speaker:.3f}")
print(f"Layer 8  ↔ Speaker: {cka_8_speaker:.3f}")
print(f"Layer 10 ↔ Speaker: {cka_10_speaker:.3f}")
print(f"Layer 12 ↔ Speaker: {cka_12_speaker:.3f}")

results = {
    'layer_ctc': {
        'layer_4': cka_4_ctc,
        'layer_6': cka_6_ctc,
        'layer_8': cka_8_ctc,
        'layer_10': cka_10_ctc,
        'layer_12': cka_12_ctc,
    },
    'layer_layer': {
        'layer_4_6': cka_4_6,
        'layer_4_8': cka_4_8,
        'layer_4_10': cka_4_10,
        'layer_4_12': cka_4_12,
        'layer_6_8': cka_6_8,
        'layer_10_12': cka_10_12,
    },
    'layer_speaker': {
        'layer_4': cka_4_speaker,
        'layer_6': cka_6_speaker,
        'layer_8': cka_8_speaker,
        'layer_10': cka_10_speaker,
        'layer_12': cka_12_speaker,
    },
}

# Populate and save
torch.save(results, 'logs/cka_analysis.pt')
