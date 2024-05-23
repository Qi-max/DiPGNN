# DiPGNN

**[Paper](https://www.science.org/doi/10.1126/sciadv.adk2799)** 

This repository contains the official TensorFlow implementation of the research paper:

Predicting the Pathways of String-like Motions in Metallic Glasses via Path-Featurizing Graph Neural Networks. Science Advances, 10, eadk2799 (2024). 

It includes the code for preparing graph data and training the model.

## Content ##
0. [Prepare graph data](#Prepare Graph Data)
0. [Training the Model](#Training the Model)
0. [File Structure](#file-structure)
0. [Citation](#citation)
0. [Acknowledgement](#acknowledgement)

## Prepare Graph Data ##

As we are focused on the glass structures, which often at least contain thousands of atoms, thus generating the graph data can be time-consuming. Therefore, we choose to save the graph data into files, avoiding regenerating the graph data in every epoch.

The glass structures should be converted to pymatgen Structure objects. Then
one can refer to the DataContainer.from_structures() method in DiPGNN_internal/dipgnn_simple/data/data_container.py to convert your original data into graph data.

Here is an example:

Suppose we have N glass configurations with target labels for paths. And the configuration is stored in the format of LAMMPS dump file.

Step 1: generate the graph data and write them to pickled files

```bash
for sample in range(N):
    dump_file = "path to glass{}.dump".format(sample)
    species_dict = {1: "Al", 2: "Sm"}
  
    struct = dump_to_pmg_structure(dump_file, species_dict)
  
    dc = DataContainerV6.from_structures(
        structure_list=[struct],
        # the dict denotes the (source_id, target_id) and the target value, the atom id starts from 0
        targets_list=[{(2, 10): [0, 1], (3, 7): [1, 0], ...}],  
        target_type="path",
        source_ids=["sample{}".format(sample)],
        task_type="regression",
        neighbor_scheme="pmg_cutoff",
        neighbor_cutoff=4.5,
        atom_feature_scheme="specie_onehot",
        specie_to_features={v: k for k, v in species_dict.items()},  # 为什么每个label后面都有个1呢
        output_graph_data=True,
        output_targets=True,
        save_graph_data_batch_size=1,
        output_path=output_path)
```

Step 2: prepare a pandas dataframe (graph_data_file_df) containing the path to the saved graph data file.
The format can be pd.DataFrame(index=range(N), columns=["data_file_path"])


## Training the Model ##

The below commands will train the DiPGNN model for the prepared data.

```bash
python DiPGNN/dipgnn/main.py \
  --embedding_dropout 0.3 \
  --output_dropout 0.2 \
  --learning_rate 0.001 \
  --graph_data_file_df ${graph_data_file_df} \
  --targets_data_file ${targets_data_file} \
  --input_size 2 \
  --output_path $output_path \
  --num_layers 4 \
  --cutoff 4.5 \
  --atom_embedding_size 32 \
  --bond_embedding_size 32 \
  --hidden_size 32 \
  --num_readout_fc_layers 3 \
  --envelope_exponent 5 \
  --target_name ${target_name} \
  --target_type "path" \
  --task_type "classification" \
  --trainer_name "base_classification_trainer" \
  --task_name "classification_training_task" \
  --batch_size 1 \
  --train 0.8 \
  --val 0.1 \
  --test 0.1 \
  --random_seed 20220111 \
  --num_targets 2
```

## File Structure ##

1. [`data`](data) Contains the code for the data container and data provider of DiPGNN.
2. [`models`](models) Includes the code for the DiPGNN model and its layers.
3. [`tasks`](tasks) Contains the code for classification and regression tasks.
4. [`trainers`](trainers) Includes the code for classification and regression trainers.
5. [`utils`](utils) Contains various utility functions.
6. [`main.py`](main.py) The main code for training the model.


## Citation ##

Please consider citing the works below if this repository is helpful:

- [DiPGNN]():
    ```bibtex
    @inproceedings{
        dipgnn,
        title={{Predicting the Pathways of String-like Motions in Metallic Glasses via Path-Featurizing Graph Neural Networks}}, 
        author={Qi Wang, Long-Fei Zhang, Zhen-Ya Zhou, Hai-Bin Yu},
        url={https://www.science.org/doi/10.1126/sciadv.adk2799},
        volume={10},
        pages={eadk2799},
        year={2024},
    }
    ```


## Acknowledgement ##

The implementation is based on [TensorFlow](https://www.tensorflow.org/) and contains functions from [dimenet](https://github.com/gasteigerjo/dimenet).