<div align="center">
<img src="images/facepipe_logo.svg" alt="logo" width=300></img>
</div>

<!-- # FakeOut -->
# FacePipe: The Official Implementation of the Face Tracking Pipeline of [FakeOut](https://github.com/gilikn/FakeOut)

## Setup
First, clone the repository:
```python
cd /local/path/for/clone
git clone https://github.com/gilikn/FacePipe.git
```
Install requirements:
```python
pip install requirements.txt
```
Run the main script with the relevant paths to your videos:
```python
python main_face_tracking_pipeline.py 
    --base_directory_path /path/to/your/dataset 
    --mean_face_path /path/to/20words_mean_face.npy
```

## BibTex
If you find <i>FakePipe</i> useful for your research, please cite the FakeOut paper:
```bib
@article{knafo2022fakeout,
  title={FakeOut: Leveraging Out-of-domain Self-supervision for Multi-modal Video Deepfake Detection},
  author = {Knafo, Gil and Fried, Ohad},
  journal={arXiv preprint arXiv:2212.00773},
  year={2022}
}
```
