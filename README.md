
## Sparse Covarying Multi-task Mixing Gaussian Processes (SC-MMGPs)
This repository includes Python and TensorFlow code, datasets and experimental scripts relating to the paper 'Direct construction of sparse Cholesky factors in covarying multi-task Gaussian processes'.

Source code may be found in `mmgp/`, datasets are in `datasets.zip` and experimental scripts are in `examples/`.

## Experiment scripts and data
Scripts and data relate to two applications: GP regression model for forecasting of solar power output (`examples/solar`) at P sites (P=25 and P=50), and Log Cox GP model for predicting flight delay count data (`examples/flight_delays`) at P airports (P=50). Please see the paper and supplement for detailed explanations of each application and descriptions of data.

To run models, see the [experimental setup](#experimental-setup).

### Solar
Solar datasets are:
- `p25_inputsdict.pickle`and `p50_inputsdict.pickle` (dictionary of train and test input and output data, wide format, for P=25, 50);
- `p25_linkinputsarray.pickle` and `p50_linkinputsarray.pickle` (numpy array containing H i.e. task-specific features for P=25, 50);
- `p25_mtg_inputsdict.pickle` and `p50_mtg_inputsdict.pickle` (dictionary of train and test input and output data, long format (incorporates H), for P=25, 50)

Solar experimental scripts correspond to experiments reported in the paper and supplement (a script is provided for each reported result), and follow the naming convention below. To reproduce any results, follow the [experimental setup](#experimental-setup) instructions below and choose the relevant script from `examples/solar`.
Naming convention:
- `p25_*` - experiments for P=25
- `p50_*` - experiments for P=50
- `*_explicit_scmmgp*` - SC-MMGP model adopting sparse prior where covariance between latent functions explicitly imposes conditional independence constraints
- `*_implicit_scmmgp*` - SC-MMGP model adopting sparse prior where covarying latent functions are automatically ("implicitly") conditionally independent
- `*_free_scmmgp*` - SC-MMGP model adopting sparse prior where covariance between latent functions is freely parameterised (within a sparse structure)
- `*_nonsparse*` - indicates covarying *non-sparse* prior in a MMGP model.
- `*_diag.py` - indicates diagonal variational posterior distribution. Otherwise, all scripts encode a sparse, Kronecker-structured variational posterior.

Benchmark models indicated by:
- `_lcm*` - linear coregional model
- `_gprn*` - Gaussian process regression network
- `_mtg*` - multi-task model with task-specific features (long-format data)

### Flight delay
Flight delay datasets are:
- `logcox_nozeroy_aggx_inputsdict.pickle` (dictionary of train and test input and output data, wide format, for airport P=50);
- `logcox_nozeroy_aggx_linkinputsarray.pickle` (numpy array containing H i.e. task-specific features for airport P=50);
- `logcox_nozeroy_aggx_pooled_inputsdict.pickle` (dictionary of train and test input and output data, long format (incorporates H), for airport P=50)

As for solar, flight delay experimental scripts in `examples/flight_delays`correspond to results reported in the paper and supplement, and can be reproduced following the instructions below. Naming conventions are as for solar datasets and are prefixed by `logcox_*`.

## Experimental Setup

#### Requirements
To ensure reproducibility, we recommend building the accompanying Docker image and launching the experiments within a Docker container.
For this you will require:
* [Docker](https://docs.docker.com/install/) (preferably >0.19)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (Note this is only supported on Linux distributions)
* [NVIDIA driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)

#### Running Experiments

1. Clone the repo onto the GPU-enabled machine
```
git clone https://github.com/axdahl/SC-MMGP.git
```

2. Adjust `setup.py` as necessary for compiler settings.
Currently this is hardcoded for `gcc4` compatibility to support triangular matrix ops code from GPflow (see [acknowledgements](#acknowledgements)).

3. Ensure that the docker default runtime environment is set to `nvidia`.
Edit/create the `/etc/docker/daemon.json` with the below content:
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

4. Restart docker and build the image
```
sudo systemctl restart docker.service
docker build -t mmgp:latest .
```

5. To launch the default experiment (`examples/solar/p25_nonsparse_cmmgp.py`), run a container as follows:
```
docker run -it --rm --runtime=nvidia -v $PWD/results:/experiments/results mmgp:latest
```

Include the `-d` flag to run in detached mode.

6. To specify a particular experiment you can override the entry command:
```
docker run -it --rm --runtime=nvidia -v $PWD/results:/experiments/results mmgp:latest python /examples/example.py
```

Outputs from experiments will be saved to the `results` directory.

## Acknowledgements
This repository makes use of code to support triangular matrix operations (under `mmgp/util/tf_ops`) from the GPflow repository (Hensman, Matthews et al. GPflow, http://github.com/GPflow/GPflow, 2016) and adapts code from GPflow for several kernels.
