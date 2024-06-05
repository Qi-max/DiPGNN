# DiPGNN

**[Link to Paper](https://www.science.org/doi/10.1126/sciadv.adk2799)** 

This repository hosts the TensorFlow implementation of the research paper titled "Predicting the Pathways of String-like Motions in Metallic Glasses via Path-Featurizing Graph Neural Networks," published in Science Advances, 10: eadk2799 (2024).

It includes the code for preparing graph data and training the model.

## Content ##
0. [Prepare Graph Data](#prepare-graph-data)
0. [Training the Model](#training-the-model)
0. [Code Structure](#code-structure)
0. [Citation](#citation)
0. [Acknowledgement](#acknowledgement)

## Prepare Graph Data ##

As we are focusing on glass structures which typically contain thousands of atoms, making the generation of graph data time-consuming. To address this, we store the graph data in files to avoid regenerating it in every epoch.

The glass structures are first converted to Pymatgen Structure objects. Then
one can refer to the ```DataContainer.from_structures()``` method in ```DiPGNN/dipgnn/data/data_container.py``` to convert your original data into graph data.

Here is an example:

Suppose we have *N* glass configurations, each with associated target labels for certain paths,  and these configurations are stored in the format of LAMMPS dump files.

Step 1: generate the graph data and write them to pickled files

```bash
from dipgnn.data.data_container import DataContainer
from dipgnn.utils.data import dump_to_pmg_structure

for sample in range(N):
    dump_file = "path to glass{}.dump".format(sample)
    species_dict = {1: "Al", 2: "Sm"}
  
    struct = dump_to_pmg_structure(dump_file, species_dict)
  
    dc = DataContainer.from_structures(
        structure_list=[struct],
        # the dict denotes the (source_id, target_id) and the target value, the atom id starts from 0
        targets_list=[{(2, 10): [0, 1], (3, 7): [1, 0], ...}],  
        target_type="path",
        source_ids=["sample{}".format(sample)],
        task_type="classification",
        neighbor_scheme="pmg_cutoff",
        neighbor_cutoff=4.5,
        atom_feature_scheme="specie_onehot",
        specie_to_features={v: k for k, v in species_dict.items()},  
        output_graph_data=True,
        output_targets=True,
        save_graph_data_batch_size=1,
        output_path=output_path)
```

Step 2: prepare a pandas dataframe ```${graph_data_file_df}``` containing the path to the saved graph data file.
The format can be ```pd.DataFrame(index=range(N), columns=["data_file_path"])```


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

## Code Structure ##

1. [`data`](data): code for the data container and data provider of DiPGNN.
2. [`models`](models): code for the DiPGNN model and its layers.
3. [`tasks`](tasks): code for classification and regression tasks.
4. [`trainers`](trainers): code for classification and regression trainers.
5. [`utils`](utils): various utility functions.
6. [`main.py`](main.py): the main code for training the model.


## Citation ##
```bibtex
@article{dipgnn,
    title={Predicting the Pathways of String-like Motions in Metallic Glasses via Path-Featurizing Graph Neural Networks}, 
    author={Qi Wang, Long-Fei Zhang, Zhen-Ya Zhou, Hai-Bin Yu},
    url={https://www.science.org/doi/10.1126/sciadv.adk2799},
    volume={10},
    pages={eadk2799},
    year={2024}
}
```

## Acknowledgement ##

The implementation is built on [TensorFlow](https://www.tensorflow.org/) and incorporates functions from [DimeNet](https://github.com/gasteigerjo/dimenet).
