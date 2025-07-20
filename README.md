# Anchored-branched Universal Physics Transformers

## Downloading the data
Download with
`sbatch src/download/download.sh [HF_TOKEN]`

Run the tutorial direclty in [colab](https://colab.research.google.com/github/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/tutorial.ipynb).

Implementation of [AB-UPT](https://arxiv.org/abs/2502.09692) containing:
- Model implementation [src/model.py](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/src/model.py)
- DrivAerML dataset implementation [src/drivaerml_dataset.py](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/src/drivaerml_dataset.py)
- Preprocessing pipeline [src/abupt_collator.py](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/src/abupt_collator.py)

The [tutorial.ipynb](https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/tutorial.ipynb) notebook showcases various aspects of our work:
- DrivAerML data download, inspection and visualization
- Preprocessing data for AB-UPT
- Running inference with AB-UPT to calculate MSE, L2 error, drag/lift coefficients and streamline velocity visualizations

We recommend to check it out yourself in an interactive [google colab](https://colab.research.google.com/github/Emmi-AI/anchored-branched-universal-physics-transformers/blob/main/tutorial.ipynb) runtime.

# Further resources

If our work sparked your interest, check out our other works and resources!

## Papers

- [UPT](https://arxiv.org/abs/2402.12365)
- [NeuralDEM](https://arxiv.org/abs/2411.09678)
- [AB-UPT](https://arxiv.org/abs/2502.09692)


## Videos/Visualizations

- [AB-UPT demo](https://demo.emmi.ai/)
- [UPT overview](https://youtu.be/mfrmCPOn4bs)
- [Computer vision for large-scale neural physics simulations (covering UPT and NeuralDEM)](https://youtu.be/6lK2E8qn5bE)


## Code

- [UPT code](https://github.com/ml-jku/UPT/)
- [UPT tutorial](https://github.com/BenediktAlkin/upt-tutorial)


# Citation

If you like our work, please consider giving it a star :star: and cite us

```
@article{alkin2025abupt,
  title={{AB-UPT}: Scaling Neural CFD Surrogates for High-Fidelity Automotive Aerodynamics Simulations via Anchored-Branched Universal Physics Transformers},
  author={Benedikt Alkin and Maurits Bleeker and Richard Kurle and Tobias Kronlachner and Reinhard Sonnleitner and Matthias Dorfer and Johannes Brandstetter},
  journal={arXiv preprint arXiv:2502.09692},
  year={2025}
}
```
