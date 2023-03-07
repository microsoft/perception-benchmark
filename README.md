# perception-benchmark

## Overview

A benchmark that evaluates and analyzes multi-modal perception models.


## Usage

#### Run locally
1. install the conda environment with `docker/env.yml`
2. install packages in (utils_package)[https://github.com/AutonomousSystemsResearch/utils_package].
3. run `pip install -e .` in the root directory of this repo
4. update configs files. For example, mounting directory for azure storage.
5. run `python tasks/train.py base=configs/multiMAE.yaml` to pretrain multi-MAE model over habitat dataset 

#### Run aml
The `jobs/amlt.yaml` is already set up.
## Note

Certain directories in this repo are cloned from exist repos as submodules. When you first `git clone` this repo, those folders of submodules will be empty.You must run two commands: `git submodule init` to initialize your local configuration file, and `git submodule update` to fetch all the data from that project and check out the appropriate commit listed in your superproject. Or you can pass --recurse-submodules to the `git clone` command, it will automatically initialize and update each submodule in the repository, including nested submodules if any of the submodules in the repository have submodules themselves.

Those submodules are usually official repo of open-sourced codes, you should avoid modifying them unless necessary.
