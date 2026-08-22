# A<sup>3</sup>CFM: Learning attribution-aware affective cognition from unpaired multi-view data with foundation models

A<sup>3</sup>CFM addresses a practical challenge in affective computing: publicly available facial emotion images, audio, and physiological stress signals are often collected from different subjects, times, and devices, making strictly paired multimodal training unavailable. To solve this problem, A<sup>3</sup>CFM learns a shared affective latent space through unpaired multi-view alignment, reliability-aware dynamic fusion, and affect-conditioned foundation model reasoning.

## Method summary
<ul>
<li>Stage 1: unpaired optimal transport data alignment</li>
<li>Stage 2: dynamic multi-view data fusion</li>
<li>Stage 3: affect-conditioned LFM inference</li>
</ul>

## Repository layout
 * mme_rag
 * runs
 * configs
    * [config.yaml](./configs/config.yaml)
    * [audio_pretrain.yaml](./configs/audio_pretrain.yaml)
   * [face_pretrain.yaml](./configs/face_pretrain.yaml)
   * [physio_pretrain.yaml](./configs/physio_pretrain.yaml)
 * scripts
     * [prepare_audio_manifest.py](./scripts/prepare_audio_manifest.py)
   * [prepare_face_manifest.py](./scripts/prepare_face_manifest.py)
 * [app_multimodal.py](./app_multimodal.py)
 * [config.yaml](./config.yaml)
 * [convert.py](./convert.py)
 * [train_audio.py](./train_audio.py)
  * [train_face.py](./train_face.py)
   * [train_physio.py](./train_physio.py)
 * [README.md](./README.md)
 * [requirements.txt](./requirements.txt)

 ## Installation
 Use an existing CUDA/PyTorch environment if available. The experiments were run with Python 3.9 and PyTorch 2.1.

 <code>
 git clone git@github.com:hehe0225/A3CFM.git
cd A3CFM
pip install -r requirements.txt
</code>

## Data Preparation
Physio dataset: WESAD、UBFC-Phys</br>
Audio dataset: IEMOCAP
MELD</br>
Image dataset: AffectNet
RAF-DB</br>

## Training
pretrain files:

```python
python scripts/prepare_audio_manifest.py \
  --iemocap-root {replaced by your dataset path} \
  --out data/audio_manifest.csv \
  --split-strategy session5_test \
  --force
```


```python
python scripts/prepare_face_manifest.py \
  --root /data/Datasets/SenseVoice/AffectNet \
  --out data/face_manifest.csv
```



train files:
```python
CUDA_VISIBLE_DEVICES=0,1 python train_physio.py \
  --set data.wesad_root={replaced by your dataset path} \
        data.rebuild_cache=true \
        train.out=runs/physio
```

```python
CUDA_VISIBLE_DEVICES=0,1 python train_audio.py \
  --set data.iemocap_root={replaced by your dataset path} \
        data.audio_manifest=data/audio_manifest.csv \
        data.prepare_manifest=true \
        train.out=runs/audio
```

```python
CUDA_VISIBLE_DEVICES=0,1  python train_face.py --config configs/face_pretrain.yaml 
```

```python
CUDA_VISIBLE_DEVICES=0,1 python train_triview_alignment.py \
  --set data.wesad_root={replaced by your dataset path} \
        data.iemocap_root={replaced by your dataset path} \
        data.affectnet_root={replaced by your dataset path} \
        data.audio_manifest=data/audio_manifest.csv \
        data.face_manifest=data/face_manifest.csv \
        init.physio_checkpoint=runs/physio/best_ok.pt \
        init.audio_checkpoint=runs/audio/best_ok.pt \
        init.face_checkpoint=runs/face/best_ok.pt \
        train.out=runs/triview_alignment
```
## Testing and validation
```
runs/*/metrics.jsonl
```

## TensorBoard monitoring
```
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```
## Web UI application
```python
python app_multimodal.py   --checkpoint runs/triview_alignment/best.pt   --config config.yaml   --server-name 0.0.0.0   --server-port 7860   --share
```

## Citation
Add the paper citation here after publication.