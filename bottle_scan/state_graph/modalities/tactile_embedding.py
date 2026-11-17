"""Optional frozen DD embeddings. Torch is loaded only for a cache miss.

Inputs are encoded sensor channels, not a rendered tactile RGB image. Both
contact surfaces use the same encoder; no labels or robot coordinates are read.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def compose(left,right,representation):
    if representation=='single_left':return np.asarray(left)
    if representation=='single_right':return np.asarray(right)
    if representation=='concat':return np.concatenate([left,right])
    if representation=='concat_difference':return np.concatenate([left,right,np.asarray(left)-right])
    raise ValueError('Unknown tactile pair representation: '+str(representation))


def read_frames(reference,count=5):
    import h5py
    if not reference:raise ValueError('Tactile embedding requires saved encoded DD frames')
    indices=reference.get('indices',[])
    if len(indices)!=count or len(set(indices))!=count:raise ValueError('Expected distinct recorded tactile frames')
    with h5py.File(reference['h5_path'],'r') as h:
        return np.stack([np.asarray(h[reference['dataset']][int(i)]) for i in indices])


def spatial_readout(image,tokens,patch_size,readout,grid=(2,3),temperature=1.):
    """Frozen deterministic readouts. Coordinates are on the normalized sensor grid."""
    import torch
    import torch.nn.functional as F
    if temperature<=0:raise ValueError('Readout temperature must be positive')
    depth=image[:,:1].abs()
    shear=torch.linalg.vector_norm(image[:,1:],dim=1,keepdim=True)
    h,w=image.shape[-2:]
    def coord(height,width):
        y,x=torch.meshgrid(torch.linspace(-1,1,height,device=image.device),
                           torch.linspace(-1,1,width,device=image.device),indexing='ij')
        return torch.stack((x,y),dim=-1).reshape(-1,2)
    if readout=='soft_argmax':
        weights=torch.cat((depth,shear),1).flatten(2).div(temperature).softmax(-1)
        return (weights@coord(h,w)).flatten(1)
    gh,gw=h//patch_size,w//patch_size
    if tokens is None or tokens.shape[1]!=gh*gw:raise ValueError('Missing spatial patch tokens')
    if readout=='mean':return tokens.mean(1)
    if readout=='grid':
        patches=tokens.transpose(1,2).reshape(len(tokens),-1,gh,gw)
        return F.adaptive_avg_pool2d(patches,grid).flatten(1)
    if readout=='attention':
        score=F.avg_pool2d(depth+shear,patch_size).flatten(1)
        weights=(score/temperature).softmax(-1)
        return torch.cat(((weights[:,:,None]*tokens).sum(1),weights@coord(gh,gw)),1)
    raise ValueError('Unknown frozen tactile readout: '+str(readout))


class FrozenEncoder:
    def __init__(self,settings):
        self.settings=dict(settings);self.artifact=json.loads(Path(settings['artifact']).read_text())
        if self.artifact.get('schema')!='tactile_encoder_artifact_v1':raise ValueError('Unexpected tactile artifact schema')
        self.representation=settings.get('representation','concat')
        self.count=int(settings.get('frames_per_finger',5));self.device=settings.get('device','cpu')
        self.dimension=int(self.artifact['model_config']['encoder_dim'])
        self.normalization=self.artifact['normalization']
        # Identity follows the actual artifact. Different checkpoints never
        # share feature caches, even when representation names are the same.
        for field,path in [('checkpoint_sha256','checkpoint'),('source_sha256','model_source')]:
            if sha256(self.artifact[path])!=self.artifact[field]:
                raise ValueError('Tactile artifact metadata does not match '+path+'; register the replacement checkpoint with its own normalization')
        self.recipe={'checkpoint':self.artifact['checkpoint_sha256'],'source':self.artifact['source_sha256'],
                     'normalization':self.normalization,'model_config':self.artifact['model_config'],
                     'preprocessing':'float32(encoded-offset-mean)/std;torch_bilinear_align_corners_false',
                     'pool':'mean_patch_tokens_excluding_cls_before_force_projector','mask_ratio':0.,
                     'frames_per_finger':self.count,'temporal_pool':'median_of_per_frame_embeddings',
                     'representation':self.representation,'precision':'float32','baseline_subtracted':False}
        self.readout=settings.get('readout','mean')
        if self.readout!='mean':
            self.recipe.update(pool=self.readout,grid=settings.get('grid',[2,3]),temperature=settings.get('temperature',1.), spatial_recipe_version=1)
        self.version='frozen_tactile_'+digest(self.recipe)
        self.cache=Path(settings['cache_dir'])/self.version
        self.model=None;self.cache_hits=0;self.computed_windows=0

    def _load(self):
        if self.model is not None:return
        import torch
        source=self.artifact['model_source']
        definition=importlib.util.spec_from_file_location('tactile_source_'+self.artifact['source_sha256'][:16],source)
        module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
        self.model=module.load_model(self.artifact['checkpoint'],device=self.device).eval().requires_grad_(False)
        if self.model.checkpoint_metadata['normalization']!=self.normalization:raise ValueError('Checkpoint normalization differs from artifact metadata')
        if self.model.config!=self.artifact['model_config']:raise ValueError('Checkpoint architecture differs from artifact metadata')
        if self.model.encoder_dim!=self.dimension:raise ValueError('Feature dimension mismatch')
        self.torch=torch

    def prepare(self,frames):
        import torch
        import torch.nn.functional as F
        frames=np.asarray(frames)
        expected=tuple(self.artifact['source_encoded_shape'])
        if frames.dtype!=np.uint16 or frames.ndim!=4 or tuple(frames.shape[1:])!=expected:
            raise ValueError('Expected encoded uint16 DD frames with shape N,'+str(expected))
        norm=self.normalization
        x=(frames.astype(np.float32)-np.asarray(norm['offset'],np.float32)-np.asarray(norm['mean'],np.float32))/np.asarray(norm['std'],np.float32)
        target=self.artifact['model_config']['image_size']
        if target[0]*expected[1]!=target[1]*expected[0]:raise ValueError('Tactile resizing must preserve aspect ratio')
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(0,3,1,2)))
        return F.interpolate(x,size=target,mode='bilinear',align_corners=False)

    def encode_arrays(self,frames):
        self._load();outputs=[]
        batch=int(self.settings.get('batch_size',10))
        with self.torch.inference_mode():
            for start in range(0,len(frames),batch):
                image=self.prepare(frames[start:start+batch]).to(self.device)
                # The trained encode() uses all patches and mean pooling; it
                # does not call the auxiliary force projector/prediction head.
                if self.readout=='mean':
                    result=self.model.encode(image)
                else:
                    patches=None
                    if self.readout!='soft_argmax':
                        tokens,_,restore=self.model.forward_encoder(image,mask_ratio=0.)
                        patches=tokens[:,1:].gather(1,restore[:,:,None].expand(-1,-1,tokens.shape[-1]))
                    result=spatial_readout(image,patches,self.artifact['model_config']['patch_size'],self.readout,
                                           self.settings.get('grid',(2,3)),self.settings.get('temperature',1.))
                outputs.append(result.float().cpu().numpy())
        return np.concatenate(outputs)

    def window(self,window):
        if 'tactile_arrays' in window:
            arrays={side:np.asarray(window['tactile_arrays'][side]) for side in ('left','right')}
        else:
            refs=window.get('tactile_frames') or {}
            arrays={side:read_frames(refs.get(side),self.count) for side in ('left','right')}
        hashes={}
        for side,frames in arrays.items():
            if frames.dtype!=np.uint16 or frames.shape!=(self.count,*self.artifact['source_encoded_shape']):
                raise ValueError('Wrong tactile frame shape/dtype for '+side)
            hashes[side]=[hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]
        key=digest({'version':self.version,'frames':hashes});path=self.cache/(key+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if result.get('version')!=self.version or result.get('evidence',{}).get('frame_sha256')!=hashes:
                raise ValueError('Inconsistent tactile feature cache')
            self.cache_hits+=1;return result
        if self.settings.get('cache_only',False):
            raise ValueError('Tactile embedding cache missing. Run the offline encoder precompute command in the Torch environment first.')
        z=self.encode_arrays(np.concatenate([arrays['left'],arrays['right']]))
        left=np.median(z[:self.count],axis=0);right=np.median(z[self.count:],axis=0)
        values=compose(left,right,self.representation)
        if not np.isfinite(values).all():raise ValueError('Nonfinite tactile embedding')
        result={'values':values.tolist(),'version':self.version,'evidence':{'recipe':self.recipe,'frame_sha256':hashes,
                'single_finger_dimension':len(left),'output_dimension':len(values),'cache_key':key,
                'artifact':self.settings['artifact'],'shared_encoder':True,'robot_started':False}}
        self.cache.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.'+str(os.getpid())+'.tmp')
        tmp.write_text(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n');tmp.replace(path)
        self.computed_windows+=1;return result


_ENCODERS={}


def embedding(window,spec):
    obs=spec.get('observation',{});settings=dict(spec['tactile_encoder'])
    if 'tac_embed_readout' in obs:
        settings.update(readout=obs['tac_embed_readout'],grid=obs.get('tac_grid',(2,3)),temperature=obs.get('tac_temperature',1.))
    key=digest(settings)
    if key not in _ENCODERS:_ENCODERS[key]=FrozenEncoder(settings)
    return _ENCODERS[key].window(window)
