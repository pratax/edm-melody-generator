# EDM Melody Generator
This repository contains the work for the lecture Applied Deep Learning (194.077) for WS22.

## Project Topic
The topic of this project is to use Recurrent Neural Networks in order to generate Electronic Dance Music melodies.

## Project Type
The type of this project is **Bring your own data** and will mainly focus on the creation of a dataset of EDM melodies in order to be able to train models to generate such melodies.

## Description
Nowadays, with the increasing availability of powerful computers on the private market and with the diffusion of Digital Audio Workstations (DAWs), it has become incredibly easy to produce a song from the comfort of our beds and more and more people are trying their luck by producing and releasing songs, dreaming of becoming famous music producers. However, making a song is a really complex process and it is extremely difficult to produce a hit song that will "catch the ears" of millions of listeners. One of the most important elements to a memorable song, especially when it comes to Electronic Dance Music (EDM), is a catchy melody and oftentimes artists struggle with coming up with one. 

The goal of this project is to create an application that facilitates the melody creation process by using deep learning methodologies to automatically generate melodies that can be used as they are or serve as a starting point to spark the imagination and creativity of artists. In particular, the type of this project is "\textbf{Bring your own data}" and the main focus will be on collecting a dataset of EDM melodies and use them to train an already existing deep learning model to generate similar melodies.

The approach that this project will use are Recurrent Neural Networks (RNN) which have proven to be really efficient in generating melodies as we can see in various publications presented at the ISMIR conference (International Society for Music Information Retrieval). Among the publications there are melody generation models like VirtuosoNet, StructureNet and other architectures based on RNNs. The reason why RNNs are suited is that they can learn the relationship each note has to the other notes being played and can use that information to generate notes based on notes that have been played previously in time. Specifically for this project, we will use the collected dataset to train MelodyRNN, a recurrent neural network designed by Google to generate monophonic melodies.

## Dataset
Having an great and ample dataset is of really high importance and in virtually any deep learning application the amount and quality of data can make the difference between a great and a poor result.

For genres like classical and pop music, big datasets already exist and an example could be the POP909 dataset, a collection of 909 pop piano performances by various professional pianists. By using this dataset it is possible to train a RNN to generate pop melodies that are to the human ear as pleasing as melodies handcrafted by professional musicians. However, when it comes to EDM, a suitable dataset hasn't yet been created. 

The main focus of this project, therefore, will be on the collection of a suitable dataset to be able to train a deep learning model on EDM melody generation. The dataset will contain monophonic melodies (not more than one note played at the same time) created by EDM artists (e.g. Avicii, Kygo, ...) saved in MIDI format (.mid), which is the standard format when it comes to storing musical information on electronic devices. In particular, MIDI does not store any audio information or any information about the sound being reproduced but rather it stores the pitch, start time, stop time and other properties of each individual note being played and is used as a musical data format by many deep learning frameworks.

## Work-Breakdown structure
| Individual Task &nbsp; | Time Estimated &nbsp; | Time Used |
|-------------------------------------------|------|------|
| Research Topic                            | 5h   |  8h  |
| Dataset Collection                        | 40h  |      |
| Network Building                          | 5h   |      |
| Network Training                          | 20h  |      |
| Application Development                   | 15h  |      |
| Final Written Report                      | 10h  |      |
| Final Presentation                        | 5h   |      |

## References
For the initial tpic research the following references have been used:

[POP909 Dataset](https://arxiv.org/abs/2008.07142)
```
@article{wang2020pop909,
  title={Pop909: A pop-song dataset for music arrangement generation},
  author={Wang, Ziyu and Chen, Ke and Jiang, Junyan and Zhang, Yiyi and Xu, Maoran and Dai, Shuqi and Gu, Xianbin and Xia, Gus},
  journal={arXiv preprint arXiv:2008.07142},
  year={2020}
}
```
[VirtuosoNet](https://archives.ismir.net/ismir2019/paper/000112.pdf)
```
@inproceedings{Jeong2019VirtuosoNetAH,
  title={VirtuosoNet: A Hierarchical RNN-based System for Modeling Expressive Piano Performance},
  author={Dasaem Jeong and Taegyun Kwon and Yoojin Kim and Kyogu Lee and Juhan Nam},
  booktitle={ISMIR},
  year={2019}
}
```
[HRNN for Melody Generation](https://ieeexplore.ieee.org/abstract/document/8918424?casa_token=rIv9NgUcMTwAAAAA:UTvyKhPB5JTH7iMsNAb1aYzbuolXtRG5xrIvauVPyXTpESg7xMPWEaBeSL_ldt2q-QgG4Hc)
```
@article{8918424,
  author={Wu, Jian and Hu, Changran and Wang, Yulong and Hu, Xiaolin and Zhu, Jun},
  journal={IEEE Transactions on Cybernetics}, 
  title={A Hierarchical Recurrent Neural Network for Symbolic Melody Generation}, 
  year={2020},
  volume={50},
  number={6},
  pages={2749-2757},
  doi={10.1109/TCYB.2019.2953194}}
```
[StructureNet](https://archives.ismir.net/ismir2018/paper/000126.pdf)
```
@inproceedings{medeot2018structurenet,
  title={StructureNet: Inducing Structure in Generated Melodies.},
  author={Medeot, Gabriele and Cherla, Srikanth and Kosta, Katerina and McVicar, Matt and Abdallah, Samer and Selvi, Marco and Newton-Rex, Ed and Webster, Kevin},
  booktitle={ISMIR},
  pages={725--731},
  year={2018}
}
```
[MelodyRNN](https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn)
```
@misc{melody-rnn-2016,
    author = {Elliot Waite},
    title = {
        Generating Long-Term Structure in Songs and Stories
    },
    journal = {Magenta Blog},
    type = {Blog},
    year = {2016},
    howpublished = {\url{https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn}}
}
```