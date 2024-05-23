from __future__ import division
import pickle
from io import StringIO
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.lammps.data import LammpsBox


def dump_to_pmg_structure(dump_file, species_dict,
                          sort_index=True):
    """
    Part of the codes are adapted from "pymatgen.io.lammps.outputs from_string".
    Args:
        dump_file: lammps dump file
        species_dict: indicates the species mapping for the dump file, such as {1: "Al", 2: "Sm"}
    Returns:
        structure: Pymatgen Structure object
    """
    with open(dump_file, 'r') as file_object:
        lines = file_object.readlines()
        # print("file length is: {}".format(len(lines)))

    box_arr = np.loadtxt(StringIO("\n".join(lines[5:8])))
    bounds = box_arr[:, :2]
    tilt = None
    if "xy xz yz" in lines[4]:
        tilt = box_arr[:, 2]
        x = (0, tilt[0], tilt[1], tilt[0] + tilt[1])
        y = (0, tilt[2])
        bounds -= np.array([[min(x), max(x)], [min(y), max(y)], [0, 0]])
    box = LammpsBox(bounds, tilt)
    lattice = box.to_lattice()

    data_columns = lines[8].replace("ITEM: ATOMS", "").split()
    data = pd.read_csv(StringIO("\n".join(lines[9:])), names=data_columns,
                       delim_whitespace=True)
    data.index = data["id"].tolist()

    if sort_index:
        data = data.sort_index()

    species = [species_dict[s] for s in data["type"]]

    if all(fc in data_columns for fc in ["x", "y", "z"]):
        coords_cols = ["x", "y", "z"]
        coords_are_cartesian = True
    elif all(fc in data_columns for fc in ["xs", "ys", "zs"]):
        coords_cols = ["xs", "ys", "zs"]
        coords_are_cartesian = False
    else:
        raise ValueError("The dump file must contain coordinates columns!")

    structure = Structure(lattice=lattice,
                          species=species,
                          coords=data[coords_cols].values,
                          validate_proximity=False,
                          to_unit_cell=False,
                          coords_are_cartesian=coords_are_cartesian)
    return structure


