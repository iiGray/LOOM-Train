import bisect, random
from loomtrain.core.data.dataset.base import CollateDataset

    

class BlendedDataset(CollateDataset):
    def __init__(self, 
                 datasets: "list[CollateDataset]",
                 sample_ratios: "list[float]" = None,
                 sample_counts: "list[int]" = None,
                 random_seed: int = 42):   
        if sample_ratios is None:
            sample_ratios = [1] * len(datasets)
        else: assert max(sample_ratios) <= 1, sample_ratios

        if sample_counts is None:
            sample_counts = [round(len(dataset) * ratio) for \
                            dataset, ratio in zip(datasets, sample_ratios)]
        else:
            sample_counts = [min(len(dataset), count) for \
                            dataset, count in zip(datasets, sample_counts)]
            
        self.datasets = datasets
        self.sample_ratios = sample_ratios
        self.sample_counts = sample_counts

        self.sampled = []
        random.seed(random_seed)
        for count, dataset in zip(sample_counts, datasets):
            indices = random.sample(range(len(dataset)), count)
            self.sampled += [sorted(indices)]
        self.lens = [0] + [len(index) for index in self.sampled]
        for i in range(1, len(self.lens)):
            self.lens[i] += self.lens[i-1]
        
        self.dataset_ids = []
        self.samples_ids = []
        for idx in range(len(self)):
            dataset_idx = bisect.bisect_right(self.lens, idx)
            data_idx = self.sampled[dataset_idx - 1][idx - self.lens[dataset_idx - 1]]
            self.dataset_ids += [dataset_idx - 1]
            self.samples_ids += [data_idx]


    def __len__(self): return self.lens[-1]

    def __getitem__(self, idx):
        dataset_idx = self.dataset_ids[idx]
        data_idx = self.samples_ids[idx]
        return self.datasets[dataset_idx][data_idx]