## Optimized MDNet for visual object tracking

This repository contains a [MatConvNet](http://www.vlfeat.org/matconvnet/) re-implementation for [MDNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf) algorithm, which is  ~10x and ~6x faster than the original matlab and python implementations, respectively.

- [MDNet-matlab-Org](https://github.com/HyeonseobNam/MDNet)

- [MDNet-python](https://github.com/HyeonseobNam/py-MDNet)


## Detail Comparisons
```shell
                |-------------------------------------------------------------------|
                |           | MDNet | pyMDNet | MDNet-Org (Ours) | MDNet-Opt (Ours) |
                |-------------------------------------------------------------------|
                | OTB-2015  |  67.9 |  65.2   |       66.4       |      67.2        |
                |-------------------------------------------------------------------|
                | VOT-2015  |  37.8 |   --    |     On going     |    On going      |
                |-------------------------------------------------------------------|
                | FPS (OTB) |   ~1  |   ~2    |       ~13        |      ~13         |
                |-------------------------------------------------------------------|
```

- MDNet: the original matlab implementation

- pyMDNet: python implementation

- MDNet-Org (Ours): our implementation with default parameters (see `setting_mdnet_org`)

- MDNet-Opt (Ours): our implementation with our settings (see `setting_mdnet_opt`)

All trackers are benchmarked on a single GPU (GTX 1080).


## Requirements and Dependencies

- NVIDIA GPU with compute capability 3.5+
- Matlab 2017a or above
- [MatConvNet](http://www.vlfeat.org/matconvnet/)


### Quick Start

To run pre-trained MDNet for OTB testing, please follow these steps:

1. Clone this repository into $MDNet:

   ```bash
   git clone git@github.com:ZjjConan/Optimized-MDNet.git $MDNet
   ```

2. Complie your MatConvNet

3. Change paths

- **`setup_optmdnet`:**

    *`lib_path`* for your own matconvnet

- **`run_evaluation_OPE`:**

    *`savePath`* for your tracking results

    *`videoPath`* for OTB dataset
    
    *`videoAttr`* for OTB subset (OTB2013 or OTB2015)

4. Models

    `mdnet_vot_otb:` training on VOT13/14/15 datasets for OTB testing.

    `mdnet_otb_vot:` training on OTB dataset for VOT15 testing.

### Training Your Own Model

please find detailed settings in **pretraining** fold for database setup and network training.


### Citations
If you use this project in your research, please cite the original MDNet paper:

    @InProceedings{nam2016mdnet,
        author = {Nam, Hyeonseob and Han, Bohyung},
        title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2016}
    }

### License

This software is being made available for research purpose only. Check LICENSE file for details.