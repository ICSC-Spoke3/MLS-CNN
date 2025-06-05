# Installation

Clone the repository and install all the dependencies with pip:

``` bash
git clone https://github.com/inigosaezcasares/MLS-CNN.git
cd MLS-CNN
pip install -r requirements.txt
```

**_NOTE:_** By default, this will install torch using `pip install torch`. If you need other torch versions (e.g. other CUDA versions, other OS/platforms), you should install torch manually before running `pip install .`. See: https://pytorch.org/

Tested with python 3.12.

# Usage

Download and decompress the training dataset (about 5GB):
``` bash
cd MLS-CNN
wget -q -O- https://filesender.renater.fr/download.php?token=27cd7419-9b32-41f7-b0ef-368a7c2daa86&files_ids=54740575 | tar xvz 
```

An example input file to train the CNN is provided and can be used as:
``` bash
mkdir -p cnn_test
python main.py train -n 1 -f examples/input_cnn_xlum_example.toml -o cnn_test
```
Some figures and useful output files are created in the `cnn_test` directory once the training is over.

Another example for the number counts:
``` bash
mkdir -p number_counts_test
python main.py train -n 1 -f examples/input_number_counts_xlum_example.toml -o number_counts_test
```

This uses a small subset of the total dataset. You can change this (and other settings) by modifying the input files.

**_NOTE:_** The provided link to download the dataset is temporary. Soon a proper zenodo repository will be set up.

More examples of input parameter files in inputs/.

More examples of bash scripts to run the code in scripts/.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, contact [inigo.saez@unimit.it](mailto:inigo.saez@unimit.it) or open an issue on GitHub.
