# Emotion_Analysis_Music-Lyrics

## Project Overview 
 
This project explores the alignment of emotions between music and lyrics by leveraging MuLan embeddings, a joint audio-language embedding model. By mapping music audio and natural language descriptions into a shared embedding space, MuLan enables us to capture semantic and emotional similarities across modalities. Our aim is to analyze and visualize the emotional alignment between music and lyrics.


## Env Setup

- Use the conda install command to install the requierments: ```$ conda install --yes --file requirements.txt```

This might not be able to install [musiclm_pytorch](https://github.com/lucidrains/musiclm-pytorch) so you will need to install it separately.

```$ pip install musiclm-pytorch ```
Musiclm_pytorch requires some specific packages, which might lead to some issues. 
If this causes any error try pip install and follow these instuctions:
- Installing the requirements: ```$ pip install -r requirements.txt```
- Downgrade the pip version to 23.*: ```$ pip install pip==23.3.1``` (we need to use this version of pip since our requirements for **musiclm_pytorch** require specific dependecies that can be installed using pip23
- Intall hydracore: ```$ pip install "hydra-core<1.1,>=1.0.7"```
- Install omegaconf: ```$ pip install "omegaconf<2.1,>=2.0.5"```

Musiclm_pytorch also requires [fairseq](https://github.com/facebookresearch/fairseq). Follow the instructions on the github page to install

Now we are good to install the musiclm-pytorch
