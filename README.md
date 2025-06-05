# Installation

Clone the repository and install all the dependencies with pip:

``` bash
git clone https://github.com/inigosaezcasares/MLS-CNN.git
cd MLS-CNN
pip install .
```

**_NOTE:_** By default, this will install torch using `pip install torch`. If you need other torch versions (e.g. other CUDA versions, other OS/platforms), you should install torch manually before running `pip install .`. See: https://pytorch.org/

Tested with python 3.12.

# Usage

An example input file to train the CNN is provided and can be used as:

``` bash
python main.py train -n 1 -f examples/input_cnn_xlum_example.toml -o examples/
```

**_NOTE:_** The required training dataset is not yet available. It will be made public soon (from zenodo).

More examples of input parameter files in inputs/.

More examples of bash scripts to run the code in scripts/.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, contact [inigo.saez@unimit.it](mailto:inigo.saez@unimit.it) or open an issue on GitHub.
