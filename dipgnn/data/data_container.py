import os
import pickle
import numpy as np
import pandas as pd
from dipgnn.utils.register import registers
from sklearn.preprocessing import StandardScaler


@registers.data_container.register("data_container")
class DataContainer(object):
    def __init__(
        self,
        graph_data_list,
        target_data_list,
        target_normalizer=None,
        target_col="targets",
        target_type="atom",
        task_type="regression",
    ):
        self.graph_data_list = np.array(graph_data_list)

        atom_features = graph_data_list[0]["atom_features_list"]
        if isinstance(atom_features[0], (np.int32, np.int64, int)):
            self.atom_feature_scheme = "specie_onehot"
        elif isinstance(atom_features[0], (list, np.ndarray, tuple, pd.Series)):
            self.atom_feature_scheme = "external"
            self.atom_feature_len = atom_features.shape[-1]
        else:
            raise ValueError("Cannot determine atom_feature_scheme", atom_features[0])

        self.target_data_list = np.array(target_data_list)

        self.target_type = target_type
        self.task_type = task_type
        self.target_col = target_col

        if task_type == "regression":
            if target_normalizer is None:
                targets = list()
                for target_data in target_data_list:
                    if target_type == "structure":
                        targets.append(target_data[target_col])
                    else:
                        targets += list(target_data[target_col])
                self.target_normalizer = StandardScaler().fit(np.array(targets).reshape(-1, 1))
            else:
                self.target_normalizer = target_normalizer
        else:
            self.target_normalizer = None

    @classmethod
    def from_files(
        cls,
        graph_data_file_list,
        targets_data_file,
        target_normalizer=None,
        target_col="targets",
        target_type="atom",
        task_type="regression"
    ):

        graph_data_list = list()

        for graph_data_file in graph_data_file_list:
            try:
                with open(graph_data_file, "rb") as f:
                    graph_data = pickle.load(f)
                    graph_data_list.append(graph_data)
            except Exception:
                raise ValueError("Please provide paths to the pickled graph_data, "
                                 "and make sure the files can be pickle.load")

        try:
            with open(targets_data_file, "rb") as f:
                target_data_list = pickle.load(f)
        except Exception:
            raise ValueError("Please provide paths to the pickled target_data_list, "
                             "and make sure the files can be pickle.load")

        return cls(graph_data_list=graph_data_list,
                   target_data_list=target_data_list,
                   target_normalizer=target_normalizer,
                   target_type=target_type,
                   task_type=task_type,
                   target_col=target_col
                   )

    @classmethod
    def from_structures(
            cls,
            structure_list=None,
            targets_list=None,
            source_ids=None,
            target_type="atom",
            task_type="regression",
            neighbor_scheme="external",
            neighbor_cutoff=4.0,
            external_neighbors_list=None,
            bond_distances_list=None,
            atom_feature_scheme="external",
            external_atom_features_list=None,
            specie_to_features=None,  # for example, {"Al1": 0, "Sm1": 1}
            output_graph_data=True,
            output_targets=True,
            save_graph_data_batch_size=1,
            output_path=None):

        batch_graph_data_list = list()
        total_graph_data_list = list()
        total_target_data_list = list()
        batch_id = 0

        for structure_id, structure in enumerate(structure_list):
            atom_features_list = list()
            dist_list = list()
            angle_kj_list = list()
            angle_im_list = list()
            id_i_list = list()
            id_j_list = list()
            angle_kj_reduce_to_dist_list = list()
            angle_im_reduce_to_dist_list = list()

            map_ji_to_dist_id = dict()
            dist_kj_expand_to_angle_meta_list = list()
            dist_im_expand_to_angle_meta_list = list()
            dist_kj_ji_expand_to_angle_meta_list = list()
            dist_im_ji_expand_to_angle_meta_list = list()

            dist_id = 0
            nn_infos = list()
            if neighbor_scheme == "external":
                for i in range(len(structure)):
                    i_coords = structure[i].coords

                    js = external_neighbors_list[structure_id][i]

                    js_to_i_frac = structure.frac_coords[js] - structure[i].frac_coords
                    js_cross_pbc = np.abs(np.round(js_to_i_frac)).sum(axis=1) > 0

                    js_to_i_frac_image = js_to_i_frac - np.round(js_to_i_frac)
                    js_frac_coords = structure[i].frac_coords + js_to_i_frac_image

                    js_coords = np.dot(js_frac_coords, structure.lattice.matrix)

                    if bond_distances_list is None:
                        dists = np.linalg.norm(js_coords - i_coords, axis=1)
                    else:
                        dists = bond_distances_list[structure_id][i]

                    nn_infos.append({"nn_ids": js,
                                     "nn_coords_image": js_coords,
                                     "nn_frac_coords_image": js_frac_coords,
                                     "nn_dists": dists,
                                     "nn_cross_pbc": js_cross_pbc,
                                     "n_neighbors": len(js)})

            elif neighbor_scheme == "pmg_cutoff":
                for i in range(len(structure)):
                    nns = structure.get_neighbors(structure[i], r=neighbor_cutoff)
                    nn_infos.append({"nn_ids": [nn[2] for nn in nns],
                                     "nn_coords_image": [nn[0].coords for nn in nns],
                                     "nn_sites_image": [nn[0] for nn in nns],
                                     "nn_dists": [nn[1] for nn in nns],
                                     "nn_cross_pbc": [True if np.abs(nn[3]).sum() != 0 else False for nn in nns],
                                     "n_neighbors": len(nns)})

            else:
                raise ValueError("No support for neighbor_scheme = {}".format(neighbor_scheme))

            for i in range(len(structure)):
                # i's coords and features
                i_coords = structure[i].coords

                if atom_feature_scheme == "external":
                    i_feature = external_atom_features_list[structure_id][i]
                elif atom_feature_scheme == "specie_onehot":
                    i_feature = specie_to_features[structure.sites[i].species.alphabetical_formula]
                else:
                    raise ValueError("No support for atom_feature_scheme = {}".format(atom_feature_scheme))

                atom_features_list.append(i_feature)

                # iterate neighbors of i, namely j
                nn_info = nn_infos[i]
                for idx in range(nn_info["n_neighbors"]):
                    j = nn_info["nn_ids"][idx]
                    j_coords = nn_info["nn_coords_image"][idx]
                    dist = nn_info["nn_dists"][idx]
                    j_cross_pbc = nn_info["nn_cross_pbc"][idx]

                    if neighbor_scheme == "external":
                        j_frac_coords = nn_info["nn_frac_coords_image"][idx]

                    elif neighbor_scheme == "pmg_cutoff":
                        j_site = nn_info["nn_sites_image"][idx]

                    else:
                        raise ValueError("No support for neighbor_scheme = {}".format(neighbor_scheme))

                    map_ji_to_dist_id["{}_{}".format(j, i)] = dist_id

                    id_i_list.append(i)
                    id_j_list.append(j)

                    dist_list.append(dist)

                    # k: neighbors of j
                    ks = nn_infos[j]["nn_ids"]
                    if j_cross_pbc:
                        if neighbor_scheme == "external":
                            ks_to_j_frac = structure.frac_coords[ks] - j_frac_coords
                            ks_to_j_frac_image = ks_to_j_frac - np.round(ks_to_j_frac)
                            ks_coords = np.dot(j_frac_coords + ks_to_j_frac_image, structure.lattice.matrix)

                        elif neighbor_scheme == "pmg_cutoff":
                            j_nns_image = structure.get_neighbors(j_site, r=neighbor_cutoff)
                            ks_coords = np.array([nn[0].coords for nn in j_nns_image])

                        else:
                            raise ValueError("No support for neighbor_scheme = {}".format(neighbor_scheme))
                    else:
                        ks_coords = nn_infos[j]["nn_coords_image"]

                    for k, k_coords in zip(ks, ks_coords):
                        if (k != i) and (k != j):
                            # Angle: ij vs jk
                            vector1 = i_coords - j_coords
                            vector2 = j_coords - k_coords
                            angle_kj_ji = DataContainer._calculate_neighbor_angles(vector1, vector2)
                            angle_kj_list.append(angle_kj_ji)

                            dist_kj_expand_to_angle_meta_list.append("{}_{}".format(k, j))
                            dist_kj_ji_expand_to_angle_meta_list.append("{}_{}".format(j, i))
                            angle_kj_reduce_to_dist_list.append(dist_id)

                    # m: neighbors of i
                    ms = nn_infos[i]["nn_ids"]
                    m_coords = nn_infos[i]["nn_coords_image"]
                    for m, m_coords in zip(ms, m_coords):
                        if (m != j):
                            # Angle: im vs ji
                            vector1 = i_coords - j_coords
                            vector2 = i_coords - m_coords
                            angle_im_ji = DataContainer._calculate_neighbor_angles(vector1, vector2)
                            angle_im_list.append(angle_im_ji)

                            dist_im_expand_to_angle_meta_list.append("{}_{}".format(m, i))
                            dist_im_ji_expand_to_angle_meta_list.append("{}_{}".format(j, i))

                            angle_im_reduce_to_dist_list.append(dist_id)
                    dist_id += 1

            dist_kj_expand_to_angle_list = list(map(lambda x: map_ji_to_dist_id[x], dist_kj_expand_to_angle_meta_list))
            dist_im_expand_to_angle_list = list(map(lambda x: map_ji_to_dist_id[x], dist_im_expand_to_angle_meta_list))
            dist_kj_ji_expand_to_angle_list = list(map(lambda x: map_ji_to_dist_id[x], dist_kj_ji_expand_to_angle_meta_list))
            dist_im_ji_expand_to_angle_list = list(map(lambda x: map_ji_to_dist_id[x], dist_im_ji_expand_to_angle_meta_list))

            # now deal with targets
            targets = targets_list[structure_id]
            
            if target_type == "structure":
                # if structure-level targets, no need to deal with targets
                reduce_to_target_indices = [0] * len(structure)
            
            elif target_type == "atom":
                if isinstance(targets, dict):
                    reduce_to_target_indices = list(targets.keys())
                    targets = list(targets.values())
                elif isinstance(targets, (list, np.ndarray, tuple, pd.Series)):
                    reduce_to_target_indices = np.arange(len(structure))
                else:
                    raise ValueError("Targets_list of atom-level dataset: "
                                     "Only supports list of lists or list of dicts")

            elif target_type == "bond":
                if isinstance(targets, dict):
                    bond_indices = ["{}_{}".format(x, y) for x, y in targets.keys()]
                    reduce_to_target_indices = list(map(lambda x: map_ji_to_dist_id[x], bond_indices))
                    targets = list(targets.values())
                else:
                    raise ValueError("Targets_list of bond-level dataset: "
                                     "Only supports list of dicts, with the key as tuple of bond indices")

            else:
                raise ValueError("Only supports structure, atom or bond-level targets")

            graph_data_dict = {
                "atom_features_list": np.array(atom_features_list).astype(np.float32 if atom_feature_scheme=="external" else np.int32),
                "dist_list": np.array(dist_list).astype(np.float32),
                "angle_kj_list": np.array(angle_kj_list).astype(np.float32),
                "angle_im_list": np.array(angle_im_list).astype(np.float32),
                "id_i_list": np.array(id_i_list).astype(np.int32),
                "id_j_list": np.array(id_j_list).astype(np.int32),
                "dist_kj_expand_to_angle_list": np.array(dist_kj_expand_to_angle_list).astype(np.int32),
                "dist_im_expand_to_angle_list": np.array(dist_im_expand_to_angle_list).astype(np.int32),
                "dist_kj_ji_expand_to_angle_list": np.array(dist_kj_ji_expand_to_angle_list).astype(np.int32),
                "dist_im_ji_expand_to_angle_list": np.array(dist_im_ji_expand_to_angle_list).astype(np.int32),
                "angle_kj_reduce_to_dist_list": np.array(angle_kj_reduce_to_dist_list).astype(np.int32),
                "angle_im_reduce_to_dist_list": np.array(angle_im_reduce_to_dist_list).astype(np.int32),
                "n_structures": 1,
            }
            target_data_dict = {
                "reduce_to_target_indices": np.array(reduce_to_target_indices).astype(np.int32),
                "targets": np.array(targets).astype(np.float32 if task_type == "regression" else np.int32)}

            total_graph_data_list.append(graph_data_dict)
            total_target_data_list.append(target_data_dict)

            if output_graph_data and save_graph_data_batch_size is not None:
                os.makedirs(output_path, exist_ok=True)

                # save the pkl file of each graph sample to file. No need to recalculate anymore.
                if save_graph_data_batch_size == 1:
                    graph_data_id = source_ids[structure_id] if source_ids is not None else structure_id
                    with open(os.path.join(output_path, "graph_data_{}.pkl".format(graph_data_id)), "wb") as f:
                        pickle.dump(graph_data_dict, f, protocol=4)

                else:
                    batch_graph_data_list.append(graph_data_dict)
                    # Default: drop last
                    if (structure_id + 1) % save_graph_data_batch_size == 0:
                        with open(os.path.join(output_path, "graph_data_batchsize{}_{}.pkl".format(
                                save_graph_data_batch_size, batch_id)), "wb") as f:
                            pickle.dump(batch_graph_data_list, f, protocol=4)
                        batch_id += 1
                        batch_graph_data_list = list()

        if output_graph_data and save_graph_data_batch_size == len(structure_list):
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "graph_data.pkl"), "wb") as f:
                pickle.dump(total_graph_data_list, f, protocol=4)

        if output_targets:
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "target_data.pkl"), "wb") as f:
                pickle.dump(total_target_data_list, f, protocol=4)

        return cls(graph_data_list=total_graph_data_list,
                   target_data_list=total_target_data_list,
                   target_type=target_type,
                   task_type=task_type)

    @staticmethod
    def collate_pool(graph_data_slice, target_data_slice,
                     target_type="atom", task_type="regression",
                     target_normalizer=None, target_col="targets"):

        n_structures = len(graph_data_slice)
        if n_structures == 1:
            input_target_data_dict_for_slice = graph_data_slice[0]

            input_target_data_dict_for_slice["reduce_to_target_indices"] = target_data_slice[0]["reduce_to_target_indices"]

            targets = target_data_slice[0][target_col]

            input_target_data_dict_for_slice["n_structures"] = 1
        else:
            atom_features_list = list()
            dist_list = list()
            angle_kj_list = list()
            angle_im_list = list()
            id_i_list = list()
            id_j_list = list()
            dist_kj_expand_to_angle_list = list()
            dist_im_expand_to_angle_list = list()
            dist_kj_ji_expand_to_angle_list = list()
            dist_im_ji_expand_to_angle_list = list()
            angle_kj_reduce_to_dist_list = list()
            angle_im_reduce_to_dist_list = list()
            reduce_to_target_indices = list()
            targets = list()

            atom_num = 0
            dist_num = 0

            for idx, (graph_data_dict, target_data_dict) in enumerate(zip(graph_data_slice, target_data_slice)):
                atom_features_list.append(graph_data_dict["atom_features_list"])
                dist_list.append(graph_data_dict["dist_list"])
                angle_kj_list.append(graph_data_dict["angle_kj_list"])
                angle_im_list.append(graph_data_dict["angle_im_list"])
                id_i_list.append(graph_data_dict["id_i_list"] + atom_num)
                id_j_list.append(graph_data_dict["id_j_list"] + atom_num)
                dist_kj_expand_to_angle_list.append(graph_data_dict["dist_kj_expand_to_angle_list"] + dist_num)
                dist_im_expand_to_angle_list.append(graph_data_dict["dist_im_expand_to_angle_list"] + dist_num)
                dist_kj_ji_expand_to_angle_list.append(graph_data_dict["dist_kj_ji_expand_to_angle_list"] + dist_num)
                dist_im_ji_expand_to_angle_list.append(graph_data_dict["dist_im_ji_expand_to_angle_list"] + dist_num)
                angle_kj_reduce_to_dist_list.append(graph_data_dict["angle_kj_reduce_to_dist_list"] + dist_num)
                angle_im_reduce_to_dist_list.append(graph_data_dict["angle_im_reduce_to_dist_list"] + dist_num)

                if target_type == "structure":
                    reduce_to_target_indices.append(target_data_dict["reduce_to_target_indices"] + idx)
                elif target_type == "atom":
                    reduce_to_target_indices.append(target_data_dict["reduce_to_target_indices"] + atom_num)
                elif target_type == "bond":
                    reduce_to_target_indices.append(target_data_dict["reduce_to_target_indices"] + dist_num)

                targets.append(target_data_dict[target_col])

                atom_num += len(graph_data_dict["atom_features_list"])
                dist_num += len(graph_data_dict["dist_list"])

            # return a dict，containing input and target
            input_target_data_dict_for_slice = {
                "atom_features_list": np.concatenate(atom_features_list),
                "dist_list": np.concatenate(dist_list),
                "angle_kj_list": np.concatenate(angle_kj_list),
                "angle_im_list": np.concatenate(angle_im_list),
                "id_i_list": np.concatenate(id_i_list),
                "id_j_list": np.concatenate(id_j_list),
                "dist_kj_expand_to_angle_list": np.concatenate(dist_kj_expand_to_angle_list),
                "dist_im_expand_to_angle_list": np.concatenate(dist_im_expand_to_angle_list),
                "dist_kj_ji_expand_to_angle_list": np.concatenate(dist_kj_ji_expand_to_angle_list),
                "dist_im_ji_expand_to_angle_list": np.concatenate(dist_im_ji_expand_to_angle_list),
                "angle_kj_reduce_to_dist_list": np.concatenate(angle_kj_reduce_to_dist_list),
                "angle_im_reduce_to_dist_list": np.concatenate(angle_im_reduce_to_dist_list),
                "n_structures": n_structures,
                "reduce_to_target_indices": np.concatenate(reduce_to_target_indices),
            }

            if target_type != "structure":
                targets = np.concatenate(targets)
            else:
                targets = np.array(targets)

        # transform targets
        if task_type == "regression":
            if len(targets.shape) == 1:
                targets = targets[:, np.newaxis]

            if target_normalizer:
                targets = target_normalizer.transform(targets)

        else:
            if len(targets.shape) != 2 or targets.shape[1] < 2:
                raise ValueError("The targets for classification task_types should be 2-dimensional, "
                                 "and target value should be one-hot type with n_classes > 1.")
        input_target_data_dict_for_slice["targets"] = targets

        return input_target_data_dict_for_slice

    @staticmethod
    def _calculate_neighbor_angles(vector1, vector2):
        x = np.sum(vector1 * vector2, axis=-1)
        y = np.cross(vector1, vector2)
        y = np.linalg.norm(y, axis=-1)
        angle = np.arctan2(y, x)
        return angle

    def __len__(self):
        return len(self.graph_data_list)

    @staticmethod
    def int_keys():
        return ['id_i_list', 'id_j_list',
                'dist_kj_expand_to_angle_list', 'dist_im_expand_to_angle_list',
                'dist_kj_ji_expand_to_angle_list', 'dist_im_ji_expand_to_angle_list',
                'angle_kj_reduce_to_dist_list', 'angle_im_reduce_to_dist_list',
                "reduce_to_target_indices"]

    @staticmethod
    def float_keys():
        return ['dist_list', 'angle_kj_list', 'angle_im_list']

    @staticmethod
    def int_number_keys():
        return ["n_structures"]

    def __getitem__(self, idx):
        return self.collate_pool(
            graph_data_slice=self.graph_data_list[idx],
            target_data_slice=self.target_data_list[idx],
            target_type=self.target_type,
            task_type=self.task_type,
            target_normalizer=self.target_normalizer,
            target_col=self.target_col
        )
