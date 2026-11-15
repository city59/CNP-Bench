
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
BUY = 4
CHUNK_ELEMS = 2 ** 26


def _full_ce(z, pos_logit, logits_fn, V):
    chunk = max(4096, CHUNK_ELEMS // max(1, len(z)))
    parts = [checkpoint(lambda zz, s=s, e=min(s + chunk, V): torch.logsumexp(logits_fn(zz, s, e), 1), z)
             for s in range(0, V, chunk)]
    lse = torch.logsumexp(torch.stack(parts, 1), 1)
    return (lse - pos_logit).mean()


def _import_models(name):
    sys.modules.setdefault("pytorch_lightning", types.SimpleNamespace(LightningModule=nn.Module))
    sys.path.insert(0, str(ROOT / "third_party" / name))
    import importlib
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]
    mod = importlib.import_module("models")
    sys.path.pop(0)
    return mod


class ClozeMB(nn.Module):

    def __init__(self, kind, n_items, n_users, ds):
        super().__init__()
        self.kind, self.n_items = kind, n_items
        self.L, self.d = 50, 16
        retail = ds == "retailrocket"
        self.hp = {"batch": 128 if retail else 32, "accum": 1 if retail else 8, "lr": 1e-3, "wd": 1e-6,
                   "betas": (0.9, 0.999), "mask_prob": 0.2 if retail else 0.3, "dropout": 0.2, "layers": 2,
                   "heads": 2, "d": 16, "max_len": 50, "loss": "full softmax", "config": "retail" if retail else "ijcai",
                   "patience": 10, "max_epochs": 1000}
        if kind == "mbstr":
            m = _import_models("mbstr")
            self.backbone = m.BERT(max_len=self.L, num_items=n_items, n_layer=2, n_head=2, n_b=4, d_model=self.d,
                                   dropout=0.2, battn=True, bpff=True, brpb=True)
            self.head = m.CGCDotProductPredictionHead(self.d, 4, 3, 1, n_items, self.backbone.embedding.token)
        else:
            m = _import_models("pbat")
            self._wd = importlib.import_module("models.utils").wasserstein_distance_matmul
            self.backbone = m.BERT(max_len=self.L, num_items=n_items, n_layer=2, n_head=2, num_users=n_users, n_b=4,
                                   d_model=self.d, dropout=0.2)
            self.head = m.WassersteinPredictionHead(self.d, n_items, self.backbone.item_embedding_m.token,
                                                    self.backbone.item_embedding_c.token)

    def _enc(self, items, behs, uids):
        if self.kind == "mbstr":
            return self.backbone(items, behs)
        xm, xc, _ = self.backbone(items, behs, uids.unsqueeze(1))
        return torch.cat([xm, xc], -1)

    def _head(self, h, behs, cands):
        if self.kind == "mbstr":
            return self.head(h, behs, cands)
        return self.head(h[:, :self.d], h[:, self.d:], behs, cands).view(cands.shape)

    def batches(self, data, rng):
        yield from data.cloze_windows(self.L, self.hp["batch"], rng)

    def train_loss(self, batch, rng):
        items, behs, uids = batch
        m = (torch.rand(items.shape, device=items.device) < self.hp["mask_prob"]) & (items > 0)
        if not m.any():
            return None
        h = self._enc(items.masked_fill(m, self.n_items + 1), behs, uids)
        hv, bv, pos = h[m], behs[m], items[m]
        V = self.n_items + 1
        if self.kind == "mbstr":
            z = self.head.mmoe_process(hv, bv)
            W = self.head.token_embeddings.weight
            return _full_ce(z, (z * W[pos]).sum(-1), lambda zz, s, e: zz @ W[s:e].T, V)
        xm, xc = self.head.out(hv[:, :self.d]), self.head.out(hv[:, self.d:])
        Wm, Wc = self.head.token_embeddings_m.weight, self.head.token_embeddings_c.weight
        dist = self._wdist
        pos_logit = dist(xm.unsqueeze(1), xc.unsqueeze(1), Wm[pos].unsqueeze(1),
                         self.head.activation(Wc[pos]).unsqueeze(1) + 1).view(-1)
        return _full_ce(torch.cat([xm, xc], 1), pos_logit,
                        lambda zz, s, e: dist(zz[:, None, :self.d], zz[:, None, self.d:], Wm[s:e],
                                              self.head.activation(Wc[s:e]) + 1)[:, 0], V)

    def _wdist(self, m1, c1, m2, c2):
        return self._wd(m1, c1, m2, c2)

    def state(self, items, behs, uids):
        it = torch.cat([items[:, 1:], torch.full_like(items[:, :1], self.n_items + 1)], 1)
        bh = torch.cat([behs[:, 1:], torch.full_like(behs[:, :1], BUY)], 1)
        return self._enc(it, bh, uids)[:, -1]

    def score(self, h, cands):
        return self._head(h, torch.full((len(h),), BUY, device=h.device), cands)

    def full(self, h):
        if self.kind == "mbstr":
            return self.head(h, torch.full((len(h),), BUY, device=h.device))[:, 1:self.n_items + 1]
        return self.head(h[:, :self.d], h[:, self.d:], None).view(len(h), -1)[:, 1:self.n_items + 1]

    def item_embeddings(self):
        tok = self.backbone.embedding.token if self.kind == "mbstr" else self.backbone.item_embedding_m.token
        return tok.weight[1:self.n_items + 1].detach().cpu().numpy()


class _Enum:
    def __getattr__(self, k):
        return k


def _load_recbole_model(name, module_file, class_name):
    base = ROOT / "third_party" / name
    utils = types.ModuleType("recbole.utils")
    for k in ("ModelType", "InputType", "FeatureSource", "FeatureType"):
        setattr(utils, k, _Enum())
    utils.set_color = lambda s, *a, **k: s
    pkg = types.ModuleType("recbole")
    pkg.__path__ = []
    mpkg = types.ModuleType("recbole.model")
    mpkg.__path__ = []
    sys.modules.update({"recbole": pkg, "recbole.utils": utils, "recbole.model": mpkg})
    for mod_name, fname in (("recbole.model.layers", "layers.py"), ("recbole.model.abstract_recommender", "abstract_recommender.py")):
        spec = importlib.util.spec_from_file_location(mod_name, base / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location(f"{name}_model", base / module_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


class _FakeDataset:
    def __init__(self, n_items):
        self.n_items = n_items
        self.field2token_id = {"item_type_list": {"0": BUY}}

    def num(self, field):
        return self.n_items + 1


RECBOLE_NAMES = {"taobao": "tmall_beh", "tmall": "tmall_beh", "ijcai15": "ijcai_beh", "retailrocket": "retail_beh"}


class RecBoleMB(nn.Module):

    def __init__(self, kind, n_items, n_users, ds):
        super().__init__()
        self.kind, self.n_items = kind, n_items
        name = RECBOLE_NAMES[ds]
        self.L = 199
        cfg = {"USER_ID_FIELD": "session_id", "ITEM_ID_FIELD": "item_id", "LIST_SUFFIX": "_list",
               "ITEM_LIST_LENGTH_FIELD": "item_length", "NEG_PREFIX": "neg_",
               "n_layers": 2, "n_heads": 2, "hidden_size": 64, "inner_size": 256, "hidden_dropout_prob": 0.5,
               "attn_dropout_prob": 0.5, "hidden_act": "gelu", "layer_norm_eps": 1e-12, "initializer_range": 0.02,
               "mask_ratio": 0.2, "loss_type": "CE", "dataset": name, "enable_ms": 1}
        cfg["MAX_ITEM_LIST_LENGTH"] = 200
        cfg.update({"enable_hg": 1, "hyper_len": 10 if name == "ijcai_beh" else 6,
                    "scales": [5, 4, 20] if name == "retail_beh" else [10, 4, 20]})
        cls = _load_recbole_model("mbht", "mbht.py", "MBHT")
        batch = 32 if name == "ijcai_beh" else 64
        self.m = cls(cfg, _FakeDataset(n_items))
        if int(torch.__version__.split(".")[0]) >= 2:
            build = self.m.build_Gs_unique
            self.m.build_Gs_unique = lambda seqs, item_sim, group_len: build(seqs, item_sim.cpu(), group_len)
        self.infer_bs = 16
        self.hp = {"batch": batch, "lr": 1e-3, "wd": 0.0, "betas": (0.9, 0.999), "d": 64, "max_len": 200,
                   "dropout": 0.5, "mask_ratio": 0.2, "layers": 2, "heads": 2, "loss": "full softmax (CE)",
                   "patience": 11, "max_epochs": 300,
                   "config": {k: v for k, v in cfg.items() if not k.isupper()}}

    @staticmethod
    def _right(items, behs):
        order = torch.sort((items == 0).int(), dim=1, stable=True).indices
        return items.gather(1, order), behs.gather(1, order)

    def batches(self, data, rng):
        yield from data.purchase_instances(self.L, self.hp["batch"], rng)

    def train_loss(self, batch, rng):
        items, behs, uids, target = batch
        it, bh = self._right(items, behs)
        dev = it.device
        mseq, pos_items, midx, tseq = self.m.reconstruct_train_data(it, bh, target)
        n_mask = torch.count_nonzero(pos_items, dim=1)
        out = self.m.forward(mseq, tseq, mask_positions_nums=(midx, n_mask), session_id=uids)
        sel = pos_items > 0
        W = self.m.item_embedding.weight
        outs = out if isinstance(out, list) else [out]
        loss = 0.0
        for o in outs:
            h = o.gather(1, midx.unsqueeze(-1).expand(-1, -1, o.shape[-1]))[sel]
            pos = pos_items[sel]
            loss = loss + _full_ce(h, (h * W[pos]).sum(-1), lambda zz, s, e: zz @ W[s:e].T, W.shape[0])
        return loss

    def state(self, items, behs, uids):
        it, bh = self._right(items, behs)
        n = torch.count_nonzero(it, dim=1)
        seq, typ = self.m.reconstruct_test_data(it, n, bh)
        out = self.m.forward(seq, typ)
        outs = out if isinstance(out, list) else [out]
        return torch.cat([self.m.gather_indexes(o, n) for o in outs], 1)

    def score(self, h, cands):
        hl = h.view(len(h), -1, self.hp["d"])
        return torch.einsum("blh,bch->blc", hl, self.m.item_embedding(cands)).max(1).values

    def full(self, h):
        hl = h.view(len(h), -1, self.hp["d"])
        w = self.m.item_embedding.weight[1:self.n_items + 1]
        return torch.stack([hl[:, j] @ w.T for j in range(hl.shape[1])], 0).max(0).values


class MISSLMB(nn.Module):

    CONFIGS = {
        "ijcai": {"batch": 8, "accum": 8, "epsilon": 0.6, "item_beh_lambda": 0.05, "next_item_lambda": 1.0,
                  "next_beh_lambda": 1.0, "inter_intent_lambda": 0.05, "intra_intent_lambda": 0.05},
        "retail": {"batch": 128, "accum": 1, "epsilon": 1.0, "item_beh_lambda": 0.01, "next_item_lambda": 1.0,
                   "next_beh_lambda": 0.5, "inter_intent_lambda": 0.01, "intra_intent_lambda": 0.01},
    }

    def __init__(self, kind, n_items, n_users, ds):
        super().__init__()
        import os
        name = "retail" if ds == "retailrocket" else "ijcai"
        c = self.CONFIGS[name]
        self.n_items, self.L, self.d = n_items, 50, 16
        m = _import_models("missl")
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        self.cl = m
        self.backbone = m.MISSL(max_len=self.L, num_items=n_items, n_layer=2, n_head=2, n_b=4, d_model=self.d,
                                dropout=0.2, buy=2, intent_num=64, nb=24, hhgt_num_head=2, hhgt_num_layer=2,
                                epsilon=c["epsilon"], sim="cos", temp=1.0, item_beh_lambda=c["item_beh_lambda"],
                                next_item_lambda=c["next_item_lambda"], next_beh_lambda=c["next_beh_lambda"],
                                inter_intent_lambda=c["inter_intent_lambda"],
                                intra_intent_lambda=c["intra_intent_lambda"], weight=1e-4, cl_sample_rate=0.2)
        self.head_i = m.DotProductPredictionHead(self.d, n_items, self.backbone.embedding.token)
        self.head_b = m.DotProductPredictionHead(self.d, 4, self.backbone.embedding.typeToken)
        self.hp = {**c, "config": name, "lr": 1e-3, "wd": 1e-6, "betas": (0.9, 0.999), "mask_prob": 0.7, "d": self.d,
                   "max_len": self.L, "layers": 2, "heads": 2, "intent_num": 64, "loss": "full softmax + behavior CE + InfoNCE",
                   "patience": 10, "max_epochs": 1000}

    def batches(self, data, rng):
        yield from data.cloze_windows(self.L + 1, self.hp["batch"], rng)

    def train_loss(self, batch, rng):
        items, behs, uids = batch
        inp, binp, nxt, bnxt = items[:, :-1], behs[:, :-1], items[:, 1:], behs[:, 1:]
        keep = (torch.rand(nxt.shape, device=items.device) < self.hp["mask_prob"]) & (nxt > 0) & (inp > 0)
        if not keep.any():
            return None
        bb = self.backbone
        item_out, type_out, item_type_pos, bs_intent, ba_intent, idx_bs, idx_ba = bb(inp, binp)
        H = item_out.shape[-1]
        flat_i, flat_t = item_out.reshape(-1, H), type_out.reshape(-1, H)
        valid = keep.reshape(-1)
        lab_i, lab_b = nxt.reshape(-1)[valid], bnxt.reshape(-1)[valid]
        z = self.head_i.ln(self.head_i.out(flat_i[valid]))
        W, bias = self.head_i.token_embeddings.weight, self.head_i.bias[0]
        V = self.n_items + 1
        loss_i = _full_ce(z, (z * W[lab_i]).sum(-1) + bias[lab_i], lambda zz, s, e: zz @ W[s:e].T + bias[s:e], V)
        loss_b = F.cross_entropy(self.head_b(flat_t[valid], None), lab_b, ignore_index=0)
        rows = keep.any(1)
        pick = (torch.rand(keep.shape, device=keep.device) * keep).argmax(1)
        sidx = (torch.arange(len(keep), device=keep.device) * keep.shape[1] + pick)[rows]
        if len(sidx) > 1:
            loss_ib = self.cl.info_nce_item_beh(flat_i[sidx], flat_t[sidx], bb.temp, len(sidx))
        else:
            loss_ib = 0.0
        loss_rec = bb.ni_lambda * loss_i + bb.nb_lambda * loss_b + bb.ib_lambda * loss_ib
        loss_inter = bb.inter_intent_lambda * self.cl.info_nce_inter_intent(
            item_type_pos, bs_intent, ba_intent, idx_bs, idx_ba, inp != 0, binp, bb.temp, bb.sim, bb.n_b, bb.n_intent,
            bb.n_bs_intents, bb.n_ba_intents)
        loss_intra = bb.intra_intent_lambda * self.cl.info_nce_intra_intent(
            bs_intent, ba_intent, idx_bs, idx_ba, inp != 0, bb.temp, bb.sim, bb.n_intents, bb.n_bs_intents,
            bb.n_ba_intents)
        return loss_rec + loss_inter + loss_intra

    def state(self, items, behs, uids):
        return self.backbone(items[:, -self.L:], behs[:, -self.L:])[0][:, -1]

    def score(self, h, cands):
        return self.head_i(h, None, cands)

    def full(self, h):
        return self.head_i(h, None)[:, 1:self.n_items + 1]


def _mbgen(*args):
    from .mbgen import MBGen
    return MBGen(*args)


MODELS = {"mbstr": ClozeMB, "pbat": ClozeMB, "mbht": RecBoleMB, "missl": MISSLMB, "mbgen": _mbgen}
