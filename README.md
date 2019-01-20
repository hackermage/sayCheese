# SayCheese: 

<img src='./results/gif/Mona_Lisa.gif' align="right" width=180>

In this project, we utilize Albert Pumarola's novel GAN conditioning scheme and build an easy-to-use ios app that can edit people's facial expression in both photos and paintings, bring him/her smile. See our home page at [Devpost](https://devpost.com/software/saycheese).

Presently, our model can generate natrual looked smiles in different styles and intensities for a large group of different types of faces, and of couse we are working on improving the model's performance.

Check out the repo for our ios app https://github.com/JinchengKim/Cruzhacks_iOS. For a simple tour of the app's usage, 
1. take picture within the app, or open one from your album
2. choose the face your want to SayCheese
3. wait a moment to see the magic, and it's done!

Now you can choose the smile face you like most from two main categories: big smile and small smile.
<!-- This code was made public to share our research for the benefit of the scientific community. Do NOT use it for immoral purposes.
 -->

## demo: SayCheese for photo

<img src='./results/show/11.jpg' align="left" hight=800>

&nbsp;

## demo: SayCheese for painting

<img src='./results/show/1.jpg' align="left" hight=800>

&nbsp;

## Prerequisites
- Install PyTorch (we use version 1.0.0), Torch Vision and dependencies from http://pytorch.org
- Install requirements.txt (```pip install -r requirements.txt```)

## Run

First, one must put the pretrained model(s) anywhere you like, they are files named **net_epoch_#epoch_id_G.pth** and **net_epoch_#epoch_id_D.pth** (#epoch refers to the index of epoch)

To run the demo:
```
python feedforward.py \
--model_path path/to/pretrained_model \
--load_epoch index_of_epoch_for_the_model \
--img_path path/to/img
```

## Citation
Our idea and work are based on Albert Pumarola's [GANimation](http://www.albertpumarola.com/research/GANimation/index.html). For more information about the model please refer to the [[Project]](http://www.albertpumarola.com/research/GANimation/index.html) and [[paper]](https://arxiv.org/abs/1807.09251).
```
@inproceedings{pumarola2018ganimation,
    title={GANimation: Anatomically-aware Facial Animation from a Single Image},
    author={A. Pumarola and A. Agudo and A.M. Martinez and A. Sanfeliu and F. Moreno-Noguer},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2018}
}
```
