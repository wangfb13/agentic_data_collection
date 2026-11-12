"""Frozen tactile/visual features from saved PROBE windows; no hardware imports."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_legacy(dd):
    """Exact legacy uint16 channel normalization, including its clipping."""
    a = np.asarray(dd)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("dd_encoded must be H,W,3; no display RGB conversion")
    x = a.astype(np.float32)
    x[..., 0] /= 3000.
    x[..., 1:] = (x[..., 1:]-25000.)/10000.
    return np.clip(x, 0., 1.)


def channel_statistics(arrays):
    count = 0; lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    lower = np.zeros(3, np.int64); upper = lower.copy(); sample = []
    for a in arrays:
        a = np.asarray(a).reshape(-1, 3).astype(np.float64)
        count += len(a); lo = np.minimum(lo, a.min(0)); hi = np.maximum(hi, a.max(0))
        lower += (a < [0, 25000, 25000]).sum(0)
        upper += (a > [3000, 35000, 35000]).sum(0)
        sample.append(a[np.linspace(0, len(a)-1, min(256, len(a)), dtype=int)])
    if not count:
        return {"pixels": 0, "status": "no_frames"}
    sample = np.concatenate(sample)
    return {"pixels": count, "min": lo.tolist(), "max": hi.tolist(),
            "below_training_range_fraction": (lower/count).tolist(),
            "above_training_range_fraction": (upper/count).tolist(),
            "clipped_fraction": ((lower+upper)/count).tolist(),
            "sample_quantiles_01_50_99": np.quantile(sample, [.01, .5, .99], axis=0).tolist(),
            "quantiles_sampling": "up to 256 evenly spaced pixels per saved frame",
            "normalization": "clip([depth/3000,(u-25000)/10000,(v-25000)/10000],0,1)",
            "channel_order": ["depth", "deformation_u", "deformation_v"]}


def read_dd_frames(reference):
    import h5py
    if not reference:
        raise ValueError("missing saved DD frame reference")
    indices = reference.get("indices", [])
    if len(indices) != 5 or len(set(indices)) != 5:
        raise ValueError("PROBE feature needs five distinct saved source frames")
    with h5py.File(reference["h5_path"], "r") as f:
        dataset = f[reference["dataset"]]
        return [np.asarray(dataset[int(i)]) for i in indices]


def _module(source, name):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_vit_state_keys(state, expected_keys):
    """Rename equivalent HuggingFace ViT layers across transformer versions."""
    expected_keys=set(expected_keys)
    translations={".attention.q_proj.":".attention.attention.query.",
                  ".attention.k_proj.":".attention.attention.key.",
                  ".attention.v_proj.":".attention.attention.value.",
                  ".attention.o_proj.":".attention.output.dense.",
                  ".mlp.fc1.":".intermediate.dense.",
                  ".mlp.fc2.":".output.dense."}
    out={}
    for key,value in state.items():
        target=key
        if key not in expected_keys:
            for old,new in translations.items():
                if old in key and key.replace(old,new) in expected_keys:
                    target=key.replace(old,new);break
        if target in out:raise ValueError("Duplicate mapped checkpoint parameter")
        out[target]=value
    return out


class FrozenTactile:
    def __init__(self, checkpoint, source, device="cpu", kind="legacy_mae"):
        import torch
        self.torch = torch; self.device = device; self.kind = kind
        self.checkpoint_sha256 = file_sha256(checkpoint)
        source = Path(source)
        self.source_sha256 = file_sha256(source)
        module = _module(source, "state_graph_frozen_tactile")
        if kind == "legacy_mae":
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ck.get("model", ck.get("state_dict", ck))
            self.model = module.TactileMAE(mask_ratio=0.)
            original_keys=set(state)
            state=adapt_vit_state_keys(state,self.model.state_dict().keys())
            self.checkpoint_compatibility={"strict_load":True,"parameter_keys_renamed":len(original_keys-set(state)),
                "mapping":"bijective HuggingFace ViT q/k/v/o and MLP parameter names; no weights discarded"}
            self.model.load_state_dict(state, strict=True)
            self.normalization = {"offset": [0,25000,25000], "scale": [3000,10000,10000],
                                  "clip": [0,1], "image_size": [224,224]}
        elif kind == "current_mae":
            self.model = module.load_model(checkpoint, device="cpu")
            self.normalization = self.model.checkpoint_metadata["normalization"]
        else:
            raise ValueError("Unknown tactile checkpoint format")
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.version = hashlib.sha256(json.dumps({"checkpoint":self.checkpoint_sha256,
             "source":self.source_sha256,"kind":kind,"pool":"patch_mean_excluding_cls",
             "window":"median_of_five","normalization":self.normalization},sort_keys=True).encode()).hexdigest()

    def _prepare(self, arrays):
        import cv2
        images = []
        for dd in arrays:
            if self.kind == "legacy_mae":
                x = normalize_legacy(dd)
                x = cv2.resize(x, (224,224), interpolation=cv2.INTER_LINEAR)
            else:
                norm = self.normalization
                x = (np.asarray(dd,dtype=np.float32)-np.asarray(norm["offset"],dtype=np.float32)-np.asarray(norm["mean"],dtype=np.float32))/np.asarray(norm["std"],dtype=np.float32)
                h,w = self.model.image_size
                x = cv2.resize(x,(w,h),interpolation=cv2.INTER_LINEAR)
            images.append(np.transpose(x,(2,0,1)))
        return self.torch.from_numpy(np.stack(images)).to(self.device)

    def encode_frames(self, arrays, batch_size=8):
        torch = self.torch; output=[]
        with torch.inference_mode():
            for first in range(0,len(arrays),batch_size):
                x = self._prepare(arrays[first:first+batch_size])
                if self.kind == "legacy_mae":
                    # No random_masking call: keep every patch in original order.
                    m=self.model; x=m.patch_embed(x).flatten(2).transpose(1,2)
                    x=x+m.pos_embed[:,1:,:]
                    cls=(m.cls_token+m.pos_embed[:,:1,:]).expand(len(x),-1,-1)
                    x=torch.cat([cls,x],dim=1)
                    for block in m.encoder_blocks:
                        x=block(x)
                        if isinstance(x,tuple):x=x[0]
                    z=m.encoder_norm(x)[:,1:,:].mean(1)
                else:
                    z=self.model.encode(x)
                output.append(z.float().cpu().numpy())
        result=np.concatenate(output)
        if result.shape!=(len(arrays),768) or not np.isfinite(result).all():
            raise ValueError("Expected finite 768-dimensional patch-pooled features")
        return result


class FrozenVision:
    def __init__(self, device="cpu", hub_dir=None, model_name="dinov2_vits14"):
        import torch
        self.torch=torch; self.device=device
        hfpath=Path(hub_dir)/"hfmodel" if hub_dir else None
        self.hf=bool(hfpath and (hfpath/"model.safetensors").exists())
        if self.hf:
            from transformers import Dinov2Model
            self.model=Dinov2Model.from_pretrained(str(hfpath),local_files_only=True).to(device).eval()
            self.origin="facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056"
        else:
            if hub_dir:torch.hub.set_dir(str(hub_dir))
            self.model=torch.hub.load("facebookresearch/dinov2",model_name,pretrained=True,trust_repo=True).to(device).eval()
            self.origin="facebookresearch/dinov2:"+model_name
        self.model.requires_grad_(False)
        self.model_name=model_name
        digest=hashlib.sha256()
        for key,tensor in sorted(self.model.state_dict().items()):
            digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
        self.version=digest.hexdigest()

    def encode(self, image):
        import cv2
        image=np.asarray(image)
        if image.ndim!=3 or image.shape[-1]!=3:raise ValueError("Wrist frame must be an RGB image")
        h,w=image.shape[:2]; scale=256/min(h,w)
        image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_LINEAR)
        h,w=image.shape[:2];image=image[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]
        x=(image.astype(np.float32)/255.-[.485,.456,.406])/[.229,.224,.225]
        x=self.torch.from_numpy(np.transpose(x.astype(np.float32),(2,0,1))[None]).to(self.device)
        with self.torch.inference_mode():
            z=self.model(pixel_values=x).last_hidden_state[:,0] if self.hf else self.model(x)
        return z.float().cpu().numpy()[0].tolist()


def read_wrist(reference):
    import cv2
    import h5py
    if reference.get("path"):
        image=cv2.imread(reference["path"],cv2.IMREAD_COLOR)
        if image is None:raise ValueError("Cannot decode saved wrist image")
        return image[:,:,::-1].copy()
    with h5py.File(reference["h5_path"],"r") as f:
        a=np.asarray(f[reference["dataset"]][reference["index"]])
    if a.ndim==1:
        a=cv2.imdecode(a.astype(np.uint8),cv2.IMREAD_COLOR)
        if a is None:raise ValueError("Cannot decode wrist image byte stream")
        return a[:,:,::-1].copy()
    return a[:,:,::-1].copy() if reference.get("color_order")=="BGR" else a


def enrich_features(rows,out_dir,config=None):
    cfg=dict(config or {});out=Path(out_dir);feature_dir=out/"features";feature_dir.mkdir(parents=True,exist_ok=True)
    model_root=feature_dir/"models/legacy_mae"
    encoder=FrozenTactile(cfg.get("checkpoint",model_root/"checkpoint-99.pth"),
                          cfg.get("model_source",model_root/"model.py"),cfg.get("device","cpu"),cfg.get("encoder_kind","legacy_mae"))
    report={"schema":"state_graph_feature_report_v1","encoder_version":encoder.version,
            "checkpoint_sha256":encoder.checkpoint_sha256,"source_sha256":encoder.source_sha256,
            "normalization":encoder.normalization,"checkpoint_compatibility":getattr(encoder,"checkpoint_compatibility",None),"mask_ratio":0,"pooling":"mean_of_patch_tokens_excluding_CLS",
            "window_aggregation":"median_of_5_independent_frames","augmentations":[],"failures":[],
            "visual":{"model":cfg.get("vision_model","dinov2_vits14"),"encoded_nodes":0,"missing_nodes":0}}
    vision=None; vision_error=None
    if cfg.get("vision",True) and any(r.get("wrist_frame") for r in rows):
        try:
            vision=FrozenVision(cfg.get("device","cpu"),feature_dir/"models/dinov2",cfg.get("vision_model","dinov2_vits14"))
            report["visual"]["weights_sha256"]=vision.version
            report["visual"]["source"]=vision.origin
        except (ValueError,OSError,RuntimeError) as exc:
            vision_error=str(exc);report["visual"]["initialization_error"]=vision_error
    joint_version=hashlib.sha256(json.dumps({"tactile":encoder.version,"visual":getattr(vision,"version",None)},sort_keys=True).encode()).hexdigest()
    report["joint_encoder_version"]=joint_version
    input_digest=hashlib.sha256()
    raw_samples=[]; tac={side:np.full((len(rows),768),np.nan,dtype=np.float32) for side in ("left","right")}
    for i,row in enumerate(rows):
        row["encoder_version"]=joint_version
        row["feature_input_sha256"]={}
        for side in ("left","right"):
            try:
                arrays=read_dd_frames(row.get("frames",{}).get(side))
                raw_samples.extend(arrays)
                row["feature_input_sha256"][side]=[hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays]
                z=encoder.encode_frames(arrays,cfg.get("batch_size",8))
                tac[side][i]=np.median(z,axis=0);row["z_tac_"+side]=tac[side][i].tolist()
            except (KeyError,ValueError,OSError) as exc:
                row["z_tac_"+side]=None
                report["failures"].append({"node_id":row["node_id"],"modality":"tac_"+side,"reason":str(exc)})
        if row.get("wrist_frame") and cfg.get("vision",True):
            try:
                if vision is None:raise RuntimeError(vision_error or "Visual encoder unavailable")
                wrist=read_wrist(row["wrist_frame"])
                row["feature_input_sha256"]["wrist_rgb"]=hashlib.sha256(str(wrist.dtype).encode()+str(wrist.shape).encode()+np.ascontiguousarray(wrist).tobytes()).hexdigest()
                row["z_vis"]=vision.encode(wrist);report["visual"]["encoded_nodes"]+=1
            except (KeyError,ValueError,OSError,RuntimeError) as exc:
                row["z_vis"]=None;report["failures"].append({"node_id":row["node_id"],"modality":"vis","reason":str(exc)})
        else:
            row["z_vis"]=None;report["visual"]["missing_nodes"]+=1
        input_digest.update(json.dumps({"node_id":row["node_id"],"inputs":row["feature_input_sha256"]},sort_keys=True).encode())
        if (i+1)%20==0:print(json.dumps({"feature_nodes_done":i+1,"total":len(rows)}),flush=True)
    report["encoded_input_manifest_sha256"]=input_digest.hexdigest()
    report["input_hash_convention"]="sha256(dtype string + shape tuple string + C-order raw array bytes), before normalization"
    report["channel_distribution"]=channel_statistics(raw_samples)
    for side,array in tac.items():np.save(feature_dir/("z_tac_"+side+".npy"),array)
    visual_dim=max((len(r["z_vis"]) for r in rows if r.get("z_vis") is not None),default=384)
    np.save(feature_dir/"z_vis.npy",np.asarray([r["z_vis"] if r.get("z_vis") is not None else [np.nan]*visual_dim for r in rows],dtype=np.float32))
    (feature_dir/"node_ids.json").write_text(json.dumps([r["node_id"] for r in rows],ensure_ascii=False))
    (feature_dir/"feature_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--nodes",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--checkpoint");p.add_argument("--model-source");p.add_argument("--no-vision",action="store_true")
    a=p.parse_args(argv);cfg={"device":a.device,"vision":not a.no_vision}
    if a.checkpoint:cfg["checkpoint"]=a.checkpoint
    if a.model_source:cfg["model_source"]=a.model_source
    source=Path(a.nodes).read_text()
    try:value=json.loads(source);rows=value["nodes"] if isinstance(value,dict) else value
    except json.JSONDecodeError:rows=[json.loads(line) for line in source.splitlines() if line.strip()]
    rows=enrich_features(rows,a.out,cfg)
    (Path(a.out)/"features"/"nodes_with_features.json").write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"nodes":len(rows),"out":str(Path(a.out)/"features/nodes_with_features.json")}))


if __name__=="__main__":main()
