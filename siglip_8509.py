"""Load only the SAME checkpoint's image tower; SKU retrieval never uses text."""
import numpy as np

from experiments.hybrid_pipeline.hf_encoders import HuggingFaceImageEncoder


def load_image_weights(model_path,vision_class,vision_config):
    """Read only image tensors using owned CPU buffers, avoiding Windows mmap faults."""
    import torch
    from pathlib import Path
    from safetensors import safe_open
    with torch.device('meta'):
        model=vision_class(vision_config)
    required=set(model.state_dict())
    weights={}
    # The installed safetensors pread backend performs ordinary reads. NumPy
    # copies give PyTorch owned writable storage, independent of the file mapping.
    with safe_open(str(Path(model_path)/'model.safetensors'),framework='np',backend='pread') as saved:
        missing=required-set(saved.keys())
        if missing:raise RuntimeError('Pinned checkpoint is missing image weights: '+', '.join(sorted(missing)))
        for name in sorted(required):
            weights[name]=torch.from_numpy(saved.get_tensor(name).copy())
    model.load_state_dict(weights,strict=True,assign=True)
    # Nonpersistent position indices are constructed by Transformers, not stored
    # in the checkpoint. Recreate them exactly after the meta-only constructor.
    for name,buffer in list(model.named_buffers()):
        if buffer.is_meta:
            if name!='vision_model.embeddings.position_ids' or buffer.ndim!=2 or buffer.shape[0]!=1:
                raise RuntimeError('Unexpected uninitialized image buffer: '+name)
            parent,attribute=name.rsplit('.',1)
            model.get_submodule(parent).register_buffer(attribute,torch.arange(buffer.shape[1],dtype=buffer.dtype).expand((1,-1)),persistent=False)
    if any(p.is_meta for p in list(model.parameters())+list(model.buffers())):
        raise RuntimeError('The pinned image tower did not fully load.')
    return model.eval().requires_grad_(False)


class ImageTowerEncoder(HuggingFaceImageEncoder):
    def __init__(self, model_path, revision):
        import torch
        from transformers import AutoImageProcessor
        self.torch=torch;self.kind='siglip2';self.name='siglip2';self.revision=revision
        self.device=torch.device('cpu')
        torch.set_num_threads(2)
        self.processor=AutoImageProcessor.from_pretrained(str(model_path),use_fast=False,local_files_only=True,revision=revision)
        import json
        from pathlib import Path
        configuration=json.loads((Path(model_path)/'config.json').read_text())
        # The pinned fixed-resolution SigLIP2 checkpoint declares model_type=siglip.
        # NaFlex checkpoints declare siglip2 and have a different patch interface.
        if configuration['model_type']=='siglip':
            from transformers import SiglipVisionConfig as VisionConfig, SiglipVisionModel as VisionModel
        elif configuration['model_type']=='siglip2':
            from transformers import Siglip2VisionConfig as VisionConfig, Siglip2VisionModel as VisionModel
        else:raise ValueError('Unexpected checkpoint architecture; no replacement model was loaded.')
        vision_config=VisionConfig(**configuration['vision_config'])
        self.model=load_image_weights(model_path,VisionModel,vision_config)

    def encode(self, images, batch_size=1):
        if not images:return np.empty((0,768),dtype=np.float32)
        vectors=[]
        with self.torch.inference_mode():
            for offset in range(0,len(images),batch_size):
                inputs=self.processor(images=[im.convert('RGB') for im in images[offset:offset+batch_size]],return_tensors='pt')
                # Siglip2Model.get_image_features returns exactly vision_model.pooler_output.
                features=self.model(**inputs).pooler_output
                features=self.torch.nn.functional.normalize(features,dim=1)
                vectors.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors)
