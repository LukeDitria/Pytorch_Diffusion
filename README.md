
# Cold Diffusion
A simple general purpose Pytorch implementation of Denoising Cold Diffusion

# Training Examples
<br>
<b> Basic training command: </b><br>
Works well for 64x64 sized images with a low-to-average diversity dataset

```
python train.py -mn test_run --dataset_root #path to dataset root#
```

<br>
<b> Starting from an existing checkpoint: </b><br>
The code will attempt to load a checkpoint with the name provided in the "save_dir" specified.

```
python train.py -mn test_run --load_checkpoint --dataset_root #path to dataset root#
```

<br>
<b> Define a Custom Architecture: </b><br>
Example showing how to define each of the main parameters of the Unet Architecture.<br>
Increasing the model depth can help with larger images, increasing width can help with more diverse datasets

```
python train.py -mn test_run --block_widths 1 2 4 8 --ch_multi 64 --dataset_root #path to dataset root#
```

<br>
<b> Change number of diffusion steps: </b><br>

```
python train.py -mn test_run --num_steps 200 --dataset_root #path to dataset root#
```


<br>
