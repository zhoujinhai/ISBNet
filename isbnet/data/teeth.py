import numpy as np
import torch

from .custom import CustomDataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class TeethDataset(CustomDataset):

    CLASSES = (
        "gum",
        "teeth"
    )

    BENCHMARK_SEMANTIC_IDXS = [i for i in range(2)]  # NOTE DUMMY values just for save results

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def getDataWeights(self):
        label_weights = np.zeros(self.sem_n_class)
        for idx, npy_file in enumerate(self.filenames):
            # print(idx, npy_file)
            data = np.load(npy_file).astype(np.float32)   
            labels = data[:, -1].astype(np.int32) 
            labels[labels > 0] = 1 
            tmp, _ = np.histogram(labels, range(self.sem_n_class + 1))
            label_weights += tmp
        print("label_number: ", label_weights)
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        label_weights = np.amax(label_weights) / label_weights 
        print("label_weights: ", label_weights)
        return label_weights

    def load(self, filename):
        n_feat = 6
        data = np.load(filename).astype(np.float32)
        labels = data[:, -1].astype(np.int32)
        point_set = data[:, 0:n_feat]
        labels = data[:, -1].astype(np.int32)
        coord = pc_normalize(point_set[:, 0:3]) 
        feat  = pc_normalize(point_set[:, 3:n_feat])  
        # coord = point_set[:, 0:3]
        # feat  = point_set[:, 3:n_feat]

        coord_min = np.min(coord, 0)
        coord -= coord_min
         
        xyz = coord
        instance_label = labels.copy()
        # instance_label[instance_label == 0] = -100
        semantic_label = labels
        semantic_label[semantic_label > 0] = 1
        # semantic_label = torch.LongTensor(semantic_label) 
        # NOTE currently teeth does not have spps, we will add later
        # spp = np.zeros(xyz.shape[0], dtype=np.long)
        spp = np.arange(xyz.shape[0], dtype=np.long)

        return xyz, feat, semantic_label, instance_label, spp
