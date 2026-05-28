# Speech Enhancement Generative Adversarial Network in PyTorch

### Requirements

```
SoundFile
scipy
librosa
h5py
numba
torch
matplotlib
numpy
pyfftw
tensorboardX
torchvision
```

# Datos para entrenamiento 

Descargar dataset FMA, disponible en [repositorio](https://github.com/mdeff/fma#). 
Guardar el dataset en data/raw_audio/ dentro de una carpeta

Para convertir MP3s a WAVs adecuados ejecutar(banderas opcionales): 

```
python preprocess_fma.py --src data/raw_audio/fma_small \
		--dst data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache
```

Para generar pares noisy primero se requiere descargar dataset [MUSAN](https://huggingface.co/datasets/FluidInference/musan/tree/main). Este tendrá que aplanarse y guardarse en data/raw_audio/ dentro de una carpeta. Yo utilicé únicamente uno de cada categoria. \
Después para generar pares clean/noisy ejecutar(banderas opcionales):

```
python make_noisy.py --clean_dir data/raw_audio/clean_fma \
		--noises_dir data/raw_audio/musan_noise \
		--out_root data \
		--snr_levels 0 5 10 15 \
		--valid_ratio 0.1 \
		--test_ratio 0.1 \
		--workers 4 \
		--seed 111 
```

Aparte yo trabajé únicamente con 1875 .wavs en total, con el split esto nos da aproximadamente 1500 de entrenamiento,
se pueden borrar en terminal con los siguientes comandos dentro del directorio de audios musicales limpios en .wav: 

```
ls *.wav | sort | tail -n +1876 | xargs -I{} rm {}
ls *.wav | wc -l 
```

El primero borra los primeros 1875 (asegurarse de añadir 1), el segundo verifica cuántos archivos hay en el directorio actual.

### Modelos preentrenados

Para ejecutar los modelos preentrenados se necesitan un conjunto de datos con ruido, los pesos del modelo y los párametros con los que fue entrenado. Una vez se tenga todo eso se puede ejecutar el siguiente comando

```
python clean.py --g_pretrained_ckpt <ruta de archivo con los pesos> \
		--cfg_file <ruta de archivo híperparametros> --synthesis_path <ruta para audios limpiados> \
		--test_files <ruta de conjunto de audios ruidosos> --soundfile
```

![SEGAN+_G](assets/segan+.png)

### References:

1. [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
2. [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
3. [Whispered-to-voiced Alaryngeal Speech Conversion with GANs (Pascual et al. 2018)](https://arxiv.org/abs/1808.10687)

### Cite

```
@article{pascual2017segan,
  title={SEGAN: Speech Enhancement Generative Adversarial Network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serr{\`a}, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```

### Notes

* Multi-GPU is not supported yet in this framework.
* Virtual Batch Norm is not included as in the very first SEGAN code, as similar results to those of original paper can be obtained with regular BatchNorm in D (ONLY D).
* If using this code, parts of it, or developments from it, please cite the above reference.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.

