# Inter- and intra- patient ECG heartbeat classification for arrhythmia detection: a sequence to sequence deep learning approach

# Paper
 Our paper can be download from the [arxiv website](https://arxiv.org/pdf/1812.07421.pdf)
## Recruitments
*tensorflow/tensorflow-gpu
*numpy
*scipy
*scikit-learn
*matplotlib
*imblearn

## Dataset
[the PhysioNet MIT-BIH Arrhythmia database](https://www.physionet.org/physiobank/database/mitdb/)
* To download the pre-processed datasets use [this link](https://drive.google.com/drive/folders/1TGg1413qa5TkcC0zF6CUDhKWlNzJgPCJ?usp=sharing), then put them into the "data folder".

## Train

* Modify args settings in seq_seq_annot_aami.py for the intra-patient ECG heartbeat classification
* Modify args settings in seq_seq_annot_DS1DS2.py for the inter-patient ECG heartbeat classification

* Run each file to reproduce the model described in the paper, use:

```
python seq_seq_annot_aami.py
```
```
python seq_seq_annot_DS1DS2.py
```

## Citation
If you find it useful, please cite our paper as follows:

```
@article{mousavi2018inter,
  title={Inter-and intra-patient ECG heartbeat classification for arrhythmia detection: a sequence to sequence deep learning approach},
  author={Mousavi, Sajad and Afghah, Fatemeh},
  journal={arXiv preprint arXiv:1812.07421},
  year={2018}
}
```

## References
 [deepschool.io](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow)

