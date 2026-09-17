"""Z-Image conditioning adapter (image OUTPUT arm for the frozen Z-Image-Turbo path).

Maps the recurrent trunk's image gen-query outputs to the conditioning a frozen
Z-Image S3-DiT cross-attends over, instead of predicting an image latent:

    gen-query states (B, K_in, d_model)
        -> self-attn mix over the K_in queries
        -> seq_len learned output slots cross-attend the K_in queries
        -> seq head : (B, seq_len, 2560)   # Qwen3-4B penultimate hidden states

Unlike SDXL there is NO pooled vector (Z-Image has none). The conditioning length
`seq_len` is free: Z-Image cross-attends over however many tokens we supply, so the
adapter emits a fixed K and the world trainer resamples the (variable-length) Qwen3
target to the same K for a token-wise MSE (see utils/zimage_text_encoder.py).

BASELINE loss = pure MSE (config.contrastive_weight defaults to 0). The manifold /
InfoNCE / whitened-MSE / through-DiT loss tiers are deferred (see project memory);
the contrastive hook below is kept dormant so they can be switched on later.

Selected by the world model ONLY when its image_coda_config is a `ZImageAdapterConfig`;
existing DiT / direct / SDXL configs are untouched (backwards compatible).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _info_nce(pred, tgt, temp, neg=None):
    """Symmetric InfoNCE: pred_i must be closer to tgt_i than to tgt_j (j!=i).

    `neg` (N, D), if given, are EXTRA negatives (a memory queue of past targets) appended
    to the pred->tgt direction only, making that direction a (B+N)-way discrimination
    instead of B-way. The tgt->pred direction stays in-batch: there is no queue of past
    PREDICTIONS because those go stale as the model trains (the targets do not — they come
    from a frozen encoder).
    """
    p = F.normalize(pred, dim=-1)
    t = F.normalize(tgt, dim=-1)
    b = p.shape[0]
    logits = p @ t.T / temp
    labels = torch.arange(b, device=p.device)
    if neg is not None and neg.shape[0] > 0:
        logits = torch.cat([logits, p @ F.normalize(neg, dim=-1).T / temp], dim=1)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits[:, :b].T, labels))


class ZImageConditioningAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.adapter_dim
        self.contrastive_weight = float(getattr(config, "contrastive_weight", 0.0))
        self.contrastive_temp = float(getattr(config, "contrastive_temp", 0.07))

        self.in_proj = nn.Linear(config.d_model, d)
        enc = nn.TransformerEncoderLayer(
            d, config.n_heads, d * 4, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.self_enc = nn.TransformerEncoder(enc, config.n_layers)

        # POSITION-WISE CODA (AR-through-trunk). The trunk is already the causal backbone, so this
        # mode bypasses in_proj/self_enc/cross_dec entirely -- all three mix positions -- and maps
        # each trunk state to its own conditioning token with a per-token velocity net. Causality
        # is then structural rather than masked, generation is O(L), and the sequence length is
        # whatever the interleaver gave the image block.
        self.pw_coda = None
        if bool(getattr(config, "coda_positionwise", False)):
            from megatransformer.model.image.pw_flow_coda import PositionwiseFlowCoda
            self.pw_coda = PositionwiseFlowCoda(
                seq_dim=config.seq_dim, ctx_dim=config.d_model,
                dim=int(getattr(config, "ar_dim", 512)),
                flow_layers=int(getattr(config, "ar_flow_layers", 3)),
                steps=int(getattr(config, "flow_steps", 8)),
                time_sampling=getattr(config, "flow_time_sampling", "logit_normal"),
                cfg_dropout=float(getattr(config, "flow_cfg_dropout", 0.0)),
                guidance=float(getattr(config, "flow_guidance", 1.0)),
                x_skip=bool(getattr(config, "flow_x_skip", False)))

        # cross_dec (the Q-Former) is optional. use_cross_dec=False drops it and out_queries
        # ENTIRELY -- no params, no compute, no gradient path -- and the aux point head then
        # reads the self_enc'd trunk states. Contrast flow_ctx="trunk", which only reroutes the
        # flow head and leaves cross_dec built, executed and trained via seq_pred.
        self.use_cross_dec = bool(getattr(config, "use_cross_dec", True))
        self.seq_len = int(config.seq_len)
        if self.use_cross_dec:
            self.out_queries = nn.Parameter(torch.randn(config.seq_len, d) * 0.02)
            dec = nn.TransformerDecoderLayer(
                d, config.n_heads, d * 4, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.cross_dec = nn.TransformerDecoder(dec, config.n_cross_layers)
        else:
            self.out_queries = None
            self.cross_dec = None

        self.seq_norm = nn.LayerNorm(d)
        self.seq_head = nn.Linear(d, config.seq_dim)
        # Small (not zero) init: near-zero conditioning for a stable regression start,
        # but non-zero so a later InfoNCE term's F.normalize has a finite gradient.
        nn.init.normal_(self.seq_head.weight, std=0.02)
        nn.init.zeros_(self.seq_head.bias)

        # Tier-1 InfoNCE projection head (dormant unless contrastive_weight>0). Contrast the
        # mean-pooled conditioning in a LEARNED space — raw cosine on LM features is
        # near-saturated (0.999+ even for wrong captions), so it can't separate them.
        pdim = int(getattr(config, "contrastive_proj_dim", 256))
        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.seq_dim, pdim), nn.GELU(), nn.Linear(pdim, pdim))

        # MoCo-style memory queue of past TARGET conditioning (pooled, in loss space).
        # In-batch InfoNCE is only a batch_size-way task -- at batch 8 that is solved to
        # ~0 loss almost immediately and contributes no gradient. The queue raises it to a
        # (B + queue_size)-way task, which is where contrastive objectives actually bite.
        #
        # Unlike MoCo there is NO momentum encoder and the stored vectors are NOT
        # pre-projected: the targets come from a FROZEN Qwen3, so raw target features never
        # go stale, and re-projecting the whole queue through the CURRENT head each step
        # gives exactly-consistent (zero-lag) negatives. Gradient is allowed to flow through
        # the queue's projection -- that is what teaches the head to spread targets apart,
        # and it cannot degenerate because a queued vector is drawn from the same
        # distribution as the positives (it WAS a positive on an earlier step).
        #
        # Non-persistent: kept out of the checkpoint (42MB at 4096x2560) since it is a
        # transient training artifact and refills in queue_size/batch_size steps.
        self.contrastive_queue_size = int(getattr(config, "contrastive_queue_size", 0))
        if self.contrastive_queue_size > 0:
            self.register_buffer(
                "contrastive_queue",
                torch.zeros(self.contrastive_queue_size, config.seq_dim),
                persistent=False)
            self.register_buffer("contrastive_queue_ptr", torch.zeros((), dtype=torch.long),
                                 persistent=False)
            self.register_buffer("contrastive_queue_fill", torch.zeros((), dtype=torch.long),
                                 persistent=False)

        # Tier-0 whitening: regress in a per-dim z-scored Qwen3 space. Centering removes
        # the massive near-constant outlier dims (LLM "massive activations"), scaling
        # equalizes each dim's loss contribution -> attacks the MSE-mean mode-collapse.
        # Stats (mean/std over the target distribution) are injected by the trainer via
        # set_whiten_stats and persist as buffers (so eval/chat de-whiten from the ckpt).
        # Identity (mean 0, std 1) by default = no-op. The head outputs WHITENED space when
        # on; the surfaced image_clip_seq_pred is de-whitened back to Qwen3 space.
        self.whiten = bool(getattr(config, "whiten_target", False))
        self.register_buffer("whiten_mean", torch.zeros(config.seq_dim))
        self.register_buffer("whiten_std", torch.ones(config.seq_dim))

        # Inference-only dispersion correction (see ZImageAdapterConfig.output_gain): scales
        # the WHITENED prediction before de-whitening, undoing the shrinkage an MSE-optimal
        # point estimate is forced into. Touches only the surfaced prediction, never the loss.
        self.output_gain = float(getattr(config, "output_gain", 1.0))

        # TIER-3: replace the point estimate with a flow-matching SAMPLER over the whitened
        # conditioning. An MSE point head is structurally under-dispersed (shrinks by 1-R^2)
        # and the frozen DiT renders that as bland; a sampler lands on-manifold at full
        # dispersion per-caption, with no global gain to tune. The Q-Former output is the
        # conditioning context; seq_head is kept as a cheap monitoring/aux point estimate
        # (flow_aux_mse_weight) so alpha/R^2 stay measurable against the regression runs.
        self.flow_head = None
        if bool(getattr(config, "flow_head", False)):
            from megatransformer.model.image.cond_flow_head import CondFlowHead
            self.flow_head = CondFlowHead(
                seq_dim=config.seq_dim, ctx_dim=d,
                dim=int(getattr(config, "flow_dim", 512)),
                n_heads=int(getattr(config, "flow_heads", 8)),
                n_layers=int(getattr(config, "flow_layers", 4)),
                dropout=config.dropout,
                steps=int(getattr(config, "flow_steps", 8)),
                time_sampling=getattr(config, "flow_time_sampling", "logit_normal"),
                cfg_dropout=float(getattr(config, "flow_cfg_dropout", 0.0)),
                guidance=float(getattr(config, "flow_guidance", 1.0)),
                max_len=int(getattr(config, "flow_max_len", 128)),
                pos_embed=bool(getattr(config, "flow_pos_embed", False)),
                # dFM repulsion weight. DISTINCT from `contrastive_weight` above, which is the
                # Tier-1 InfoNCE term on the pooled POINT estimate -- a different object, and one
                # that measurably improved retrieval without moving the render. This one acts on
                # the VELOCITY field during training.
                contrastive_weight=float(getattr(config, "flow_contrastive_weight", 0.0)),
                x_skip=bool(getattr(config, "flow_x_skip", False)),
                x1_pred=bool(getattr(config, "flow_x1_pred", False)),
                x1_sigma_min=float(getattr(config, "flow_x1_sigma_min", 0.02)),
                loss_weighting=str(getattr(config, "flow_loss_weighting", "none")),
                min_snr_gamma=float(getattr(config, "flow_min_snr_gamma", 5.0)),
                var_loss_weight=float(getattr(config, "flow_var_loss_weight", 0.0)),
                var_barrier_weight=float(getattr(config, "flow_var_barrier_weight", 0.0)),
                var_steps=int(getattr(config, "flow_var_steps", 4)))
        self.flow_aux_mse_weight = float(getattr(config, "flow_aux_mse_weight", 0.1))
        # Detached aux head: the MSE (and only the MSE) trains seq_head, never the trunk.
        self.flow_aux_mse_detach = bool(getattr(config, "flow_aux_mse_detach", False))
        if self.flow_aux_mse_detach:
            if self.flow_head is None:
                raise ValueError(
                    "flow_aux_mse_detach=True with no flow head: the point-head MSE is then the "
                    "ONLY objective, and detaching it would train nothing but seq_head. Use it "
                    "only on flow (T3+) configs.")
            if self.contrastive_weight > 0:
                raise ValueError(
                    "flow_aux_mse_detach=True with contrastive_weight>0: the Tier-1 InfoNCE term "
                    "also reads seq_pred, so detaching would silently make it a no-op on the "
                    "shared representation instead of the representation-shaping loss it is.")
        # T5: run the PARALLEL head at the caption's native Qwen3 length instead of resampling
        # the target to K slots. Same one-shot sampler as T3 (no sequential inference), just
        # without the K=64 interpolation -- so ~2/3 of the supervised slots stop being linear
        # blends of neighbouring hidden states. The Q-Former context stays at a fixed seq_len
        # slots; only the OUTPUT length becomes native, exactly as in the AR head.
        if bool(getattr(config, "flow_x1_pred", False)) and bool(getattr(config, "flow_x_skip", False)):
            raise ValueError("flow_x1_pred SUPERSEDES flow_x_skip -- set one, not both: x1 "
                             "prediction already subtracts x_t analytically, so a learned skip "
                             "on top would double-count it")
        self.flow_ctx = str(getattr(config, "flow_ctx", "qformer"))
        if self.flow_ctx not in ("qformer", "trunk"):
            raise ValueError(f"flow_ctx must be 'qformer' or 'trunk', got {self.flow_ctx!r}")
        self.flow_project = None          # None | "proj" | "renorm"; set at INFERENCE only
        self.flow_native_length = bool(getattr(config, "flow_native_length", False))
        if self.flow_native_length and self.flow_head is None:
            raise ValueError("flow_native_length requires flow_head")

        # T4: autoregressive per-token flow over the conditioning (ar_cond_flow_head.py).
        # Mutually exclusive with the parallel flow head. Targets arrive at NATIVE length with
        # a mask (no K-slot resample); the sample length comes from the caption's tokenization.
        self.ar_flow_head = None
        if bool(getattr(config, "ar_flow_head", False)):
            if getattr(self, "flow_head", None) is not None:
                raise ValueError("set either flow_head or ar_flow_head, not both")
            from megatransformer.model.image.ar_cond_flow_head import ARCondFlowHead
            self.ar_flow_head = ARCondFlowHead(
                seq_dim=config.seq_dim, ctx_dim=d,
                dim=int(getattr(config, "ar_dim", 512)),
                n_heads=int(getattr(config, "ar_heads", 8)),
                n_layers=int(getattr(config, "ar_layers", 4)),
                flow_layers=int(getattr(config, "ar_flow_layers", 3)),
                max_len=int(getattr(config, "ar_max_len", 128)),
                dropout=config.dropout,
                steps=int(getattr(config, "flow_steps", 8)),
                time_sampling=getattr(config, "flow_time_sampling", "logit_normal"),
                cfg_dropout=float(getattr(config, "flow_cfg_dropout", 0.0)),
                guidance=float(getattr(config, "flow_guidance", 1.0)),
                x_skip=bool(getattr(config, "flow_x_skip", False)),
                pos_mode=str(getattr(config, "ar_pos_mode", "learned")))

    def _flow_out_basis(self):
        """Orthonormal basis of the flow head's REACHABLE output subspace, cached.

        `out` is Linear(flow_dim -> seq_dim) with flow_dim << seq_dim, so every velocity the head
        can predict lies in the (at most flow_dim-dimensional) column space of out.weight. Euler
        integration only ever ADDS velocities to the initial noise, so the noise component
        orthogonal to this subspace reaches the output untouched -- measured at 59% of the emitted
        energy at w=1 and 43% at w=3 on t3_2. This basis lets inference project that leak away.
        """
        head = self.flow_head if self.flow_head is not None else self.ar_flow_head
        W = (head.out if self.flow_head is not None else head.flow.out).weight
        key = (W.data_ptr(), W.shape, W.device)
        if getattr(self, "_flow_basis_key", None) != key:
            Q, _ = torch.linalg.qr(W.detach().float())
            self._flow_basis_key, self._flow_basis = key, Q
        return self._flow_basis

    def _project_flow_output(self, x):
        """Drop the component of a flow sample that the head could never have written.

        mode "proj"   -- hard projection onto the reachable subspace.
        mode "renorm" -- project, then rescale to UNIT per-dim std. Whitened targets have std 1
                         by construction, and projection removes most of the energy (~59% at
                         w=1), so the raw projection lands well under-dispersed -- the exact
                         failure mode this arm of the project keeps rediscovering.
        """
        head = self.flow_head if self.flow_head is not None else self.ar_flow_head
        if bool(getattr(head, "x_skip", False)):
            # The basis below is the column space of `out.weight` -- precisely the subspace
            # flow_x_skip exists to escape. Projecting an x_skip checkpoint onto it would DELETE
            # the skip's full-rank noise cancellation, i.e. undo the fix. flow_project is for
            # legacy heads that cannot remove their own noise.
            raise ValueError(
                "--flow_project is for heads WITHOUT flow_x_skip. This checkpoint has the skip, "
                "which already removes the leak during sampling; projecting onto out.weight's "
                "range would discard exactly that correction.")
        Q = self._flow_out_basis()
        xf = x.float()
        p = (xf @ Q) @ Q.T
        if self.flow_project == "renorm":
            p = p / p.std().clamp_min(1e-6)
        return p.to(x.dtype)

    def set_whiten_stats(self, mean, std, eps: float = 1e-6):
        """Load per-dim Qwen3 target mean/std into the whitening buffers and enable it."""
        m = torch.as_tensor(mean, dtype=self.whiten_mean.dtype).flatten()
        s = torch.as_tensor(std, dtype=self.whiten_std.dtype).flatten().clamp_min(eps)
        assert m.numel() == self.whiten_mean.numel(), \
            f"whiten mean size {m.numel()} != seq_dim {self.whiten_mean.numel()}"
        self.whiten_mean.copy_(m.to(self.whiten_mean.device))
        self.whiten_std.copy_(s.to(self.whiten_std.device))
        self.whiten = True

    def _queue_negatives(self):
        """Filled slice of the target memory queue, or None if disabled/empty.

        CLONED, not a view: _enqueue writes into the buffer in place during the same
        forward, which would otherwise bump the version of the exact tensor the projection
        head consumed and make backward fail ("modified by an inplace operation").
        """
        if self.contrastive_queue_size <= 0:
            return None
        fill = int(self.contrastive_queue_fill)
        return self.contrastive_queue[:fill].clone() if fill > 0 else None

    @torch.no_grad()
    def _enqueue(self, vecs):
        """Ring-buffer write of pooled target vectors (B, seq_dim) into the memory queue."""
        if self.contrastive_queue_size <= 0:
            return
        q = self.contrastive_queue
        v = vecs.detach().to(q.dtype).to(q.device)
        n, cap = v.shape[0], q.shape[0]
        if n >= cap:                                   # batch alone overfills: keep the tail
            q.copy_(v[-cap:])
            self.contrastive_queue_ptr.zero_()
            self.contrastive_queue_fill.fill_(cap)
            return
        ptr = int(self.contrastive_queue_ptr)
        end = ptr + n
        if end <= cap:
            q[ptr:end] = v
        else:                                          # wrap
            head = cap - ptr
            q[ptr:] = v[:head]
            q[:end - cap] = v[head:]
        self.contrastive_queue_ptr.fill_(end % cap)
        self.contrastive_queue_fill.fill_(min(cap, int(self.contrastive_queue_fill) + n))

    def forward(
        self,
        encoder_hidden_states,          # (B, K_in, d_model) trunk image gen-query outputs
        cond_labels=None,               # (B, seq_len, 2560) resampled Qwen3 target
        sample_mask=None,               # (B,) bool: rows to include in the loss (synthesis only)
        latent_labels=None,             # accepted + ignored (shared generator call signature)
        **kw,
    ):
        if self.pw_coda is not None:
            # Trunk states ARE the per-position context; nothing upstream of the velocity net.
            ctx = encoder_hidden_states
            mask = kw.get("cond_mask")
            if cond_labels is None:
                surf = self.pw_coda.sample(
                    ctx, steps=kw.get("flow_steps"),
                    generator=kw.get("flow_generator", getattr(self, "flow_generator", None)),
                    guidance=kw.get("flow_guidance"))
                if self.whiten:
                    surf = surf * self.whiten_std.view(1, 1, -1) + self.whiten_mean.view(1, 1, -1)
                return {"image_clip_seq_pred": surf}
            tgt = ((cond_labels - self.whiten_mean.view(1, 1, -1)) / self.whiten_std.view(1, 1, -1)
                   if self.whiten else cond_labels)
            if sample_mask is not None and mask is not None and mask.shape[0] == sample_mask.shape[0]:
                mask = mask[sample_mask.bool()]
            loss = self.pw_coda.loss(tgt, ctx, mask=mask)
            return {"image_clip_loss": loss, "image_flow_loss": loss.detach()}

        x = self.in_proj(encoder_hidden_states)          # (B, K_in, d)
        x = self.self_enc(x)
        if self.use_cross_dec:
            q = self.out_queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, seq_len, d)
            q = self.cross_dec(q, x)                     # seq_len slots attend the K_in queries
        else:
            # cross_dec DROPPED: the trunk states are the conditioning directly. Only valid when
            # K_in == seq_len, because cross_dec is the sole K -> seq_len remap.
            if x.shape[1] != self.seq_len:
                raise ValueError(
                    f"use_cross_dec=False needs K_in == seq_len, got K_in={x.shape[1]} vs "
                    f"seq_len={self.seq_len}. cross_dec is the only module that remaps K to "
                    f"seq_len; drop it only when n_image_gen_positions == seq_len.")
            q = x
        # Detach here (not at the loss) so EVERY consumer of seq_pred is off the trunk's graph,
        # not just the MSE term. No-op at inference.
        q_for_seq = q.detach() if self.flow_aux_mse_detach else q
        seq_pred = self.seq_head(self.seq_norm(q_for_seq))  # (B, seq_len, seq_dim); WHITENED space if self.whiten
        # WHAT THE GENERATIVE HEAD IS CONDITIONED ON. Default "qformer" = the cross_dec output q,
        # as shipped. "trunk" hands it `x` instead -- the self_enc'd trunk states -- bypassing
        # cross_dec entirely for the head (seq_pred still uses q, so the aux point head is
        # unaffected and the ablation stays single-variable).
        # WHY THIS IS WORTH ABLATING: self_enc + cross_dec are 33.1M params, 49% of the adapter,
        # and cross_dec's stated purpose -- decoupling K input queries from the output length --
        # is UNUSED at the shipped sizes (image_gen_queries 64 -> out_queries 64, an
        # identity-shaped map). It is also the same primitive the flow head already applies:
        # learned queries cross-attending to trunk states, which _FlowBlock does at every block
        # of every Euler step. So it may be a redundant bottleneck on a conditioning path that is
        # already long and narrow (text -> trunk -> 64 positions -> Q-Former -> head).
        # NOTE at NATIVE length cross_dec would finally do real K->L work, so this question can
        # have different answers at K=64 and at native length.
        head_ctx = x if getattr(self, "flow_ctx", "qformer") == "trunk" else q

        # WHAT GETS SURFACED (in whitened space):
        #   flow head + no labels (inference) -> INTEGRATE FROM NOISE. Full-dispersion,
        #     on-manifold, per-caption -- the whole point of T3.
        #   otherwise -> the point head (also used during training, where surfacing the
        #     cheap regression keeps the alpha/R^2 diagnostics comparable to the T0/T1 runs
        #     and avoids paying `steps` extra head passes on every training forward).
        if self.ar_flow_head is not None and cond_labels is None:
            # AR inference: emit exactly L tokens, L deterministic from the caption's Qwen3
            # tokenization (callers pass cond_length); no stop token, no length head.
            seq_surf = self.ar_flow_head.sample(
                head_ctx, int(kw.get("cond_length") or seq_pred.shape[1]),
                generator=kw.get("flow_generator", getattr(self, "flow_generator", None)))
        elif self.flow_head is not None and self.flow_native_length and cond_labels is None:
            # Native length: emit exactly as many slots as Qwen3 would produce for this caption
            # (deterministic from the caption, same as the AR path -- no stop token needed).
            seq_surf = self.flow_head.sample(
                head_ctx, int(kw.get("cond_length") or seq_pred.shape[1]),
                generator=kw.get("flow_generator", getattr(self, "flow_generator", None)))
        elif self.flow_head is not None and cond_labels is None:
            # Seeded sampling matters for EVAL COMPARABILITY: the head emits a DISTRIBUTION, so
            # an unseeded draw makes checkpoint-to-checkpoint differences a mix of training
            # progress and sampling noise. Setting `.flow_generator` pins the noise so the same
            # draw is compared across checkpoints; leave it None for genuinely random samples.
            seq_surf = self.flow_head.sample(
                head_ctx, seq_pred.shape[1],
                generator=kw.get("flow_generator", getattr(self, "flow_generator", None)))
        else:
            seq_surf = seq_pred
        # Inference-time leak removal. Only meaningful for a SAMPLED output (the point head is
        # not rank-limited this way), so it is gated on cond_labels being absent.
        if getattr(self, "flow_project", None) and cond_labels is None \
                and (self.flow_head is not None or self.ar_flow_head is not None):
            seq_surf = self._project_flow_output(seq_surf)

        # Surface the prediction in Qwen3 space: de-whiten the head output when whitening
        # is on (identity otherwise). generate()/eval/chat render this directly.
        if self.whiten:
            mean = self.whiten_mean.view(1, 1, -1)
            std = self.whiten_std.view(1, 1, -1)
            # output_gain scales in WHITENED space (target mean 0), i.e. it scales the
            # deviation from the target mean and leaves the mean itself alone.
            out_seq = (seq_surf * self.output_gain) * std + mean
        else:
            if self.output_gain != 1.0:
                raise ValueError(
                    "output_gain requires whiten_target: without whitening the prediction is in "
                    "raw Qwen3 space, where scaling would also blow up the massive near-constant "
                    "dims (|mean| up to 1013) instead of scaling the deviation from the mean.")
            out_seq = seq_surf
        # Reuse the "image_clip_*" output keys so the world model / trainer / generate()
        # plumbing is shared with the SDXL adapter; pooled is None (Z-Image has none).
        out = {"image_clip_seq_pred": out_seq, "image_clip_pooled_pred": None}
        if cond_labels is not None:
            # Loss in the SAME space as seq_pred: whiten the target when enabled.
            target = ((cond_labels - self.whiten_mean.view(1, 1, -1)) / self.whiten_std.view(1, 1, -1)
                      if self.whiten else cond_labels)
            sp, sl, ctx = seq_pred, target, head_ctx
            # Restrict the loss to flagged rows (transcription rows carry an INPUT image,
            # not a gen target). Guard on matching length so a bad mask is ignored.
            if sample_mask is not None and sample_mask.shape[0] == seq_pred.shape[0]:
                m = sample_mask.bool()
                if int(m.sum()) == 0:
                    return out                            # no synthesis rows this batch
                sp, sl, ctx = seq_pred[m], target[m], head_ctx[m]
            sl = sl.to(sp.dtype)
            if self.ar_flow_head is not None:
                # T4: native-length masked AR flow. There is no comparable point-head MSE here
                # (seq_pred is a fixed-K estimate while the target is variable-length), so the
                # flow loss is the whole signal.
                cmask = kw.get("cond_mask")
                if cmask is not None and sample_mask is not None and \
                        cmask.shape[0] == sample_mask.shape[0]:
                    cmask = cmask[sample_mask.bool()]
                flow = self.ar_flow_head.loss(sl, ctx, mask=cmask)
                out["image_clip_loss"] = flow
                out["image_flow_loss"] = flow.detach()
                return out
            if self.flow_head is not None and self.flow_native_length:
                # As in the AR branch there is no comparable point-head MSE here: seq_pred is a
                # fixed-K estimate while the target is variable-length, so the flow loss is the
                # whole signal.
                cmask = kw.get("cond_mask")
                if cmask is not None and sample_mask is not None and \
                        cmask.shape[0] == sample_mask.shape[0]:
                    cmask = cmask[sample_mask.bool()]
                flow = self.flow_head.loss(sl, ctx, mask=cmask)
                out["image_clip_loss"] = flow
                out["image_flow_loss"] = flow.detach()
                return out
            mse = F.mse_loss(sp, sl)
            if self.flow_head is not None:
                # T3: the flow-matching objective IS the training signal. The point-head MSE
                # is kept only as a small auxiliary so seq_pred stays a usable diagnostic
                # (alpha / R^2 / retrieval) against the regression runs -- set the weight to
                # 0 to train a pure sampler.
                flow = self.flow_head.loss(sl, ctx)
                loss = flow + self.flow_aux_mse_weight * mse
                out["image_flow_loss"] = flow.detach()
            else:
                loss = mse
            if self.contrastive_weight > 0 and sp.shape[0] >= 2:   # Tier-1: needs >=2 rows
                tgt_pool = sl.mean(1)                              # (B, seq_dim), loss space
                neg = self._queue_negatives()
                nce = _info_nce(
                    self.contrastive_proj(sp.mean(1)),
                    self.contrastive_proj(tgt_pool),
                    self.contrastive_temp,
                    neg=self.contrastive_proj(neg.to(sp.dtype)) if neg is not None else None)
                loss = loss + self.contrastive_weight * nce
                out["image_contrastive_loss"] = nce.detach()
                out["image_contrastive_negatives"] = torch.as_tensor(
                    float(0 if neg is None else neg.shape[0]), device=sp.device)
                # Enqueue AFTER the loss so the current batch's own targets are never in
                # its own negative set (they are already the in-batch negatives).
                if self.training:
                    self._enqueue(tgt_pool)
            out["image_clip_loss"] = loss
            out["image_clip_mse_loss"] = mse.detach()
        return out
