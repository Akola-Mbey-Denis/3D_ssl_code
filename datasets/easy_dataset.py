# # --------------------------------------------------------
# # A dataset base class with resizable and composable behavior
# # --------------------------------------------------------

# import numpy as np
# from .batched_sampler import BatchedRandomSampler


# class EasyDataset:
#     """
#     A dataset that supports arithmetic-like composition.
    
#     Supported operations:
#     ---------------------
#     - 2 * dataset      => duplicate each element 2x (data * 2)
#     - 10 @ dataset     => resize to 10 samples (random sampling with replacement)
#     - dataset1 + dataset2 => concatenate datasets
#     """

#     def __add__(self, other):
#         return CatDataset([self, other])

#     def __rmul__(self, factor):
#         return MulDataset(factor, self)

#     def __rmatmul__(self, factor):
#         return ResizedDataset(factor, self)

#     def set_epoch(self, epoch):
#         """Override if dataset needs epoch-based reshuffling."""
#         pass

#     def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
#         if not shuffle:
#             raise NotImplementedError("Non-shuffling sampler not supported.")
#         num_of_aspect_ratios = len(self._resolutions)
#         return BatchedRandomSampler(
#             self, batch_size, num_of_aspect_ratios,
#             world_size=world_size, rank=rank, drop_last=drop_last
#         )


# class MulDataset(EasyDataset):
#     """Repeat the dataset N times."""

#     def __init__(self, multiplicator, dataset):
#         assert isinstance(multiplicator, int) and multiplicator > 0
#         self.multiplicator = multiplicator
#         self.dataset = dataset

#     def __len__(self):
#         return self.multiplicator * len(self.dataset)

#     def __repr__(self):
#         return f"{self.multiplicator} * {repr(self.dataset)}"

#     def __getitem__(self, idx):
#         if isinstance(idx, tuple):
#             idx, other = idx
#             return self.dataset[idx // self.multiplicator, other]
#         return self.dataset[idx // self.multiplicator]


# class ResizedDataset(EasyDataset):
#     """Resize dataset to a fixed number of samples using randomized cycling."""

#     def __init__(self, new_size, dataset):
#         assert isinstance(new_size, int) and new_size > 0
#         self.new_size = new_size
#         self.dataset = dataset
#         self._idxs_mapping = None

#     def __len__(self):
#         return self.new_size

#     def __repr__(self):
#         size_str = f"{self.new_size:,}".replace(",", "_")
#         return f"{size_str} @ {repr(self.dataset)}"

#     def set_epoch(self, epoch):
#         """Randomly sample indices with cycling to match target size."""
#         rng = np.random.default_rng(seed=epoch + 777)
#         perm = rng.permutation(len(self.dataset))
#         repeat_factor = 1 + (self.new_size - 1) // len(self.dataset)
#         shuffled_idxs = np.tile(perm, repeat_factor)[:self.new_size]
#         self._idxs_mapping = shuffled_idxs

#     def __getitem__(self, idx):
#         if self._idxs_mapping is None:
#             raise RuntimeError("Call `set_epoch()` before indexing ResizedDataset.")
#         mapped_idx = self._idxs_mapping[idx]
#         if isinstance(idx, tuple):
#             _, other = idx
#             return self.dataset[mapped_idx, other]
#         return self.dataset[mapped_idx]


# class CatDataset(EasyDataset):
#     """Concatenate multiple EasyDatasets."""

#     def __init__(self, datasets):
#         assert all(isinstance(ds, EasyDataset) for ds in datasets)
#         self.datasets = datasets
#         self._cum_sizes = np.cumsum([len(ds) for ds in datasets])

#     def __len__(self):
#         return self._cum_sizes[-1]

#     def __repr__(self):
#         return " + ".join(repr(ds) for ds in self.datasets)

#     def set_epoch(self, epoch):
#         for ds in self.datasets:
#             ds.set_epoch(epoch)

#     def __getitem__(self, idx):
#         other = None
#         if isinstance(idx, tuple):
#             idx, other = idx

#         if not (0 <= idx < len(self)):
#             raise IndexError("Index out of range.")

#         db_idx = np.searchsorted(self._cum_sizes, idx, side='right')
#         dataset = self.datasets[db_idx]
#         base = self._cum_sizes[db_idx - 1] if db_idx > 0 else 0
#         new_idx = idx - base
#         return dataset[(new_idx, other)] if other is not None else dataset[new_idx]



# --------------------------------------------------------
# A dataset base class with resizable and composable behavior
# --------------------------------------------------------
# # --------------------------------------------------------
# # A dataset base class with resizable and composable behavior
# # --------------------------------------------------------

# import numpy as np
# from .batched_sampler import BatchedRandomSampler


# class EasyDataset:
#     """
#     A dataset that supports arithmetic-like composition.
    
#     Supported operations:
#     ---------------------
#     - 2 * dataset      => duplicate each element 2x (data * 2)
#     - 10 @ dataset     => resize to 10 samples (random sampling with replacement)
#     - dataset1 + dataset2 => concatenate datasets
#     """

#     def __add__(self, other):
#         return CatDataset([self, other])

#     def __rmul__(self, factor):
#         return MulDataset(factor, self)

#     def __rmatmul__(self, factor):
#         return ResizedDataset(factor, self)

#     def set_epoch(self, epoch):
#         """Override if dataset needs epoch-based reshuffling."""
#         pass

#     def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
#         if not shuffle:
#             raise NotImplementedError("Non-shuffling sampler not supported.")
#         num_of_aspect_ratios = len(self._resolutions)
#         return BatchedRandomSampler(
#             self, batch_size, num_of_aspect_ratios,
#             world_size=world_size, rank=rank, drop_last=drop_last
#         )


# class MulDataset(EasyDataset):
#     """Repeat the dataset N times."""

#     def __init__(self, multiplicator, dataset):
#         assert isinstance(multiplicator, int) and multiplicator > 0
#         self.multiplicator = multiplicator
#         self.dataset = dataset

#     def __len__(self):
#         return self.multiplicator * len(self.dataset)

#     def __repr__(self):
#         return f"{self.multiplicator} * {repr(self.dataset)}"

#     def __getitem__(self, idx):
#         if isinstance(idx, tuple):
#             idx, other = idx
#             return self.dataset[idx // self.multiplicator, other]
#         return self.dataset[idx // self.multiplicator]


# class ResizedDataset(EasyDataset):
#     """Resize dataset to a fixed number of samples using randomized cycling."""

#     def __init__(self, new_size, dataset):
#         assert isinstance(new_size, int) and new_size > 0
#         self.new_size = new_size
#         self.dataset = dataset
#         self._idxs_mapping = None

#     def __len__(self):
#         return self.new_size

#     def __repr__(self):
#         size_str = f"{self.new_size:,}".replace(",", "_")
#         return f"{size_str} @ {repr(self.dataset)}"

#     def set_epoch(self, epoch):
#         """Randomly sample indices with cycling to match target size."""
#         rng = np.random.default_rng(seed=epoch + 777)
#         perm = rng.permutation(len(self.dataset))
#         repeat_factor = 1 + (self.new_size - 1) // len(self.dataset)
#         shuffled_idxs = np.tile(perm, repeat_factor)[:self.new_size]
#         self._idxs_mapping = shuffled_idxs

#     def __getitem__(self, idx):
#         if self._idxs_mapping is None:
#             raise RuntimeError("Call `set_epoch()` before indexing ResizedDataset.")
#         mapped_idx = self._idxs_mapping[idx]
#         if isinstance(idx, tuple):
#             _, other = idx
#             return self.dataset[mapped_idx, other]
#         return self.dataset[mapped_idx]


# class CatDataset(EasyDataset):
#     """Concatenate multiple EasyDatasets."""

#     def __init__(self, datasets):
#         assert all(isinstance(ds, EasyDataset) for ds in datasets)
#         self.datasets = datasets
#         self._cum_sizes = np.cumsum([len(ds) for ds in datasets])

#     def __len__(self):
#         return self._cum_sizes[-1]

#     def __repr__(self):
#         return " + ".join(repr(ds) for ds in self.datasets)

#     def set_epoch(self, epoch):
#         for ds in self.datasets:
#             ds.set_epoch(epoch)

#     def __getitem__(self, idx):
#         other = None
#         if isinstance(idx, tuple):
#             idx, other = idx

#         if not (0 <= idx < len(self)):
#             raise IndexError("Index out of range.")

#         db_idx = np.searchsorted(self._cum_sizes, idx, side='right')
#         dataset = self.datasets[db_idx]
#         base = self._cum_sizes[db_idx - 1] if db_idx > 0 else 0
#         new_idx = idx - base
#         return dataset[(new_idx, other)] if other is not None else dataset[new_idx]



# --------------------------------------------------------
# A dataset base class with resizable and composable behavior
# --------------------------------------------------------

import numpy as np
from .batched_sampler import BatchedRandomSampler


class EasyDataset:
    """
    A dataset that supports arithmetic-like composition.
    
    Supported operations:
    ---------------------
    - 2 * dataset      => duplicate each element 2x (data * 2)
    - 10 @ dataset     => resize to 10 samples (random sampling with replacement)
    - dataset1 + dataset2 => concatenate datasets
    """

    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset(factor, self)

    def set_epoch(self, epoch):
        """Override if dataset needs epoch-based reshuffling."""
        pass

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
        if not shuffle:
            raise NotImplementedError("Non-shuffling sampler not supported.")
        num_of_aspect_ratios = len(self._resolutions)
        return BatchedRandomSampler(
            self, batch_size, num_of_aspect_ratios,
            world_size=world_size, rank=rank, drop_last=drop_last
        )


class MulDataset(EasyDataset):
    """Repeat the dataset N times."""

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __repr__(self):
        return f"{self.multiplicator} * {repr(self.dataset)}"

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, other = idx
            return self.dataset[idx // self.multiplicator, other]
        return self.dataset[idx // self.multiplicator]


class ResizedDataset(EasyDataset):
    """Resize dataset to a fixed number of samples using randomized cycling."""

    def __init__(self, new_size, dataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset
        self._idxs_mapping = None

    def __len__(self):
        return self.new_size

    def __repr__(self):
        size_str = f"{self.new_size:,}".replace(",", "_")
        return f"{size_str} @ {repr(self.dataset)}"

    def set_epoch(self, epoch):
        """Randomly sample indices with cycling to match target size.
        Fixed seed -> the same subset every epoch (also safe with persistent workers)."""
        rng = np.random.default_rng(seed=777)
        perm = rng.permutation(len(self.dataset))
        repeat_factor = 1 + (self.new_size - 1) // len(self.dataset)
        shuffled_idxs = np.tile(perm, repeat_factor)[:self.new_size]
        self._idxs_mapping = shuffled_idxs

    def __getitem__(self, idx):
        if self._idxs_mapping is None:
            raise RuntimeError("Call `set_epoch()` before indexing ResizedDataset.")
        mapped_idx = self._idxs_mapping[idx]
        if isinstance(idx, tuple):
            _, other = idx
            return self.dataset[mapped_idx, other]
        return self.dataset[mapped_idx]


class CatDataset(EasyDataset):
    """Concatenate multiple EasyDatasets."""

    def __init__(self, datasets):
        assert all(isinstance(ds, EasyDataset) for ds in datasets)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __repr__(self):
        return " + ".join(repr(ds) for ds in self.datasets)

    def set_epoch(self, epoch):
        for ds in self.datasets:
            ds.set_epoch(epoch)

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            idx, other = idx

        if not (0 <= idx < len(self)):
            raise IndexError("Index out of range.")

        db_idx = np.searchsorted(self._cum_sizes, idx, side='right')
        dataset = self.datasets[db_idx]
        base = self._cum_sizes[db_idx - 1] if db_idx > 0 else 0
        new_idx = idx - base
        return dataset[(new_idx, other)] if other is not None else dataset[new_idx]