<br>
<p align="center">
<h1 align="center"><strong>LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities</strong></h1>
  <p align="center">
	<br>
    <a href='https://zcmax.github.io//' target='_blank'>Chenming Zhu</a>&emsp;
	<a href='https://tai-wang.github.io/' target='_blank'>Tai Wang*</a>&emsp;
    <a href='https://zhangwenwei.cn/' target='_blank'>Wenwei Zhang</a>&emsp;
    <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang</a>&emsp;
	<a href='https://xh-liu.github.io//' target='_blank'>Xihui Liu*</a>&emsp;
    <br>
    The University of Hong Kong&emsp;Shanghai AI Laboratory
    <br>
  </p>
</p>


<div id="top" align="center">

<a href="https://arxiv.org/abs/2409.18125" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-LLaVA--3D-red?logo=arxiv" height="20" />
</a>
<a href="(https://zcmax.github.io/projects/LLaVA-3D/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-LLaVA--3D-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data" target="_blank">
    <img alt="HF Dataset: LLaVA-3D-Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Dataset-LLaVA--3D--Data-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
<a href="https://huggingface.co/ChaimZhu/LLaVA-3D-7B" target="_blank">
    <img alt="HF Dataset: LLaVA-3D-7B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-LLaVA--3D--7B-ffc107?color=ffc107&logoColor=white" height="20" />
</a>


</div>


## 🏠 Introducing LLaVA-3D
<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/llava-3d-teaser-combine-v2.png" alt="Dialogue_Teaser" width=100% >
</div>
LLaVA-3D could perform both 2D and 3D vision-language tasks. The left block (b) shows that compared with previous 3D LMMs, our LLaVA-3D achieves state-of-the-art performance across a wide range of 3D benchmarks while maintaining a comparable performance on various 2D benchmarks compared with LLaVA-1.5. The middle block (c) demonstrates that LLaVA-3D is built on the 2D LMM: LLaVA, and leverages 3D patches to endow it with 3D spatial awareness, enabling it to perform various 3D vision-and-language tasks in the physical world. The right blocks (d) and (e) highlights the significantly faster convergence and inference speeds of LLaVA-3D compared to existing 3D LMMs.

## 🔥 News
- [2025-07-11] :hearts: Our paper is accepted by ICCV 2025! See u in Hawaii! We release the full `LLaVA-3D-Instruct-86OK` data on [HuggingFace](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Data)!
- [2024-11-29] We update the custom data instruction tuning tutorial, now you can train the model on your own dataset!
- [2024-10-19] We release the inference codes with checkpoints as well as the image and 3D scene demos. You can chat with LLaVA-3D with your own machines.
- [2024-09-28] We release the [paper](https://arxiv.org/abs/2409.18125) of LLaVA-3D. &#x1F389;

<!-- contents with emoji -->
## 📋 Contents
- [🔍 Model Architecture](#-model-architecture)
- [🔨 Install](#-install)
- [📦 Model Zoo](#-model-zoo)
- [🤖 Demo](#-demo)
- [📝 TODO List](#-todo-list)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 🔍 Model Architecture
<p align="center">
  <img src="assets/llava-3d-method-v13.png" align="center" width="100%">
</p>
LLaVA-3D Architecture. Based on LLaVA, we directly add the corresponding 3D position embeddings to 2D patch visual tokens of multi-view images to construct the 3D Patches, then the 3D Patches will undergo 3D pooling and be sent into the projection layer of LLaVA to map into the LLM space and align with the LLM using 3D-visual-language data.


## 🔨 Install
We test our codes under the following environment:
* Python 3.10
* Pytorch 2.1.0
* CUDA Version 11.8

To start: 
1. Clone this repository.

```bash
git clone https://github.com/ZCMax/LLaVA-3D.git
cd LLaVA-3D
```

2. Install Packages

```Shell
conda create -n llava-3d python=3.10 -y
conda activate llava-3d
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -e .
```

3. Download the [Camera Parameters File](https://drive.google.com/file/d/1a-1MCFLkfoXNgn9XdlmS9Gnzplrzw7vf/view?usp=drive_link) and put the json file under the `./playground/data/annotations`.

4. Install additional packages for training cases

```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## 📦 Model Zoo

The trained model checkpoints are available [here](https://huggingface.co/ChaimZhu/LLaVA-3D-7B). Currently we only provide the 7B model, and we will continue to update the model zoo.

## 🤖 Demo

We currently support single image as inputs for 2D tasks and posed RGB-D images as inputs for 3D tasks. You can run the demo by using the script `llava/eval/run_llava_3d.py`. For 2D tasks, use the `image-file` parameter, and for 3D tasks, use the `video-path` parameter to provide the corresponding data. Here, we provide some demos as examples:

### 2D Tasks

```Shell
python llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --image-file https://llava-vl.github.io/static/images/view.jpg \
    --query "What are the things I should be cautious about when I visit here?"
```

### 3D Tasks

We provide the demo scene [here](https://huggingface.co/datasets/ChaimZhu/LLaVA-3D-Demo-Data). Download the demo data and put it under the `./demo`.

1. 3D Question Answering

```Shell
python llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-path ./demo/scannet/scene0356_00 \
    --query "Tell me the only object that I could see from the other room and describe the object."
```

2. 3D Dense Captioning

```Shell
python llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-path ./demo/scannet/scene0566_00 \
    --query "The related object is located at [0.981, 1.606, 0.430]. Describe the object in detail."
```

3. 3D Localization

```Shell
python llava/eval/run_llava_3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-path ./demo/scannet/scene0382_01 \
    --query "The related object is located at [-0.085,1.598,1.310]. Please output the 3D bounding box of the object and then describe the object."
```


## 📝 TODO List

- \[x\] Release the training and inference code.
- \[x\] Release the checkpoint, demo data and script.
- \[x\] Release the training datasets.
- \[ \] Release the full code.

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{zhu2024llava,
  title={LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness},
  author={Zhu, Chenming and Wang, Tai and Zhang, Wenwei and Pang, Jiangmiao and Liu, Xihui},
  journal={arXiv preprint arXiv:2409.18125},
  year={2024}
}
```

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements

This repo benefits from [3D-LLM](https://github.com/UMass-Foundation-Model/3D-LLM), [LLaVA](https://github.com/haotian-liu/LLaVA), and [ODIN](https://github.com/ayushjain1144/odin). 
