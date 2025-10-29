# DCGAN for MNIST (PyTorch)

A Deep Convolutional GAN (Radford, Metz, Chintala 2016) that learns to generate 28x28 handwritten-style digits, following the paper's recipe: strided / transposed convolutions instead of pooling or fully-connected layers, BatchNorm, ReLU/tanh in the generator, LeakyReLU(0.2) in the discriminator, N(0, 0.02) weight init, Adam(2e-4, beta1 = 0.5), the non-saturating generator loss, and one-sided label smoothing (real = 0.9).

> **MNIST was not downloaded or used for the results here.** Fetching it needs a network download I did not make on your behalf. The loader reads the four standard MNIST IDX files if you put them in `data/` (`python train.py --data mnist --mnist-dir data`; no torchvision needed). The reported results are on **procedurally drawn 7-segment digits** (`dcgan/data.py`, random stroke width, shift, slant, noise), a stand-in that lets the whole pipeline run offline on a CPU in a few minutes. It is an easier dataset than MNIST, so **do not read these numbers as MNIST performance**.

## Files
`dcgan/models.py` generator and discriminator, `dcgan/data.py` MNIST IDX reader + synthetic digits, `train.py` training loop with the GAN health diagnostics, `evaluate.py` quantitative check, `tests/` (5 tests).

## Results (6,000 synthetic digits, 25 epochs, CPU)
Training diagnostics (last epoch): D loss 0.60, G loss 2.33, **D(x) = 0.77, D(G(z)) = 0.13**. A healthy GAN keeps D(x) high but below 1 and D(G(z)) low but above 0; the discriminator is ahead but the generator is still learning, not collapsed.

Because a grid of pictures can flatter a model, `evaluate.py` measures it: a small CNN classifier trained on real digits (100% accurate on held-out real digits) is applied to 2,000 generated images:
```
mean classifier confidence 0.89 | classes produced 10/10 | class entropy 0.94 (1.00 = perfectly balanced)
class histogram: [105, 137, 49, 272, 237, 110, 145, 195, 404, 346]
pixel mean real -0.71 / fake -0.72;  std real 0.60 / fake 0.61
```
- **No mode collapse:** all ten digits appear (entropy 0.94), though not evenly: 8 and 9 are over-produced and 2 is rare.
- **Mostly convincing, not perfect:** the classifier is confident on average (0.89), and the pixel statistics match the data closely. Looking at `samples/epoch_025.png`, many digits are clean, but some have missing or broken segments. More epochs, more data, or a lower-noise discriminator would help.
- `samples/epoch_001.png` -> `epoch_025.png` shows the progression from noise to digits.

