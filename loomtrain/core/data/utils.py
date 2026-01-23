from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from typing import Type, TypeVar, Callable, List, Tuple, Iterable

def split_dataset(dataset: Dataset, 
                  ratios_dict: dict = dict(
                      train = 8,
                      val = 1,
                      test = 1
                  ),
                  shuffle: bool = False,
                  seed: int | None = 42) -> DatasetDict:
    assert len(ratios_dict) > 2, ratios_dict
    full = sum(ratios_dict.values())
    for key in ratios_dict:
        ratios_dict[key] /= full
    
    sample_ratios = ratios_dict.values()
    sample_counts = [round(ratio * len(dataset)) \
                     for ratio in sample_ratios]

    tmp_dataset = dataset
    splits = []
    for sample_count in sample_counts[:-1]:
        tmp_dict = tmp_dataset.train_test_split(train_size = sample_count,
                                                shuffle = shuffle, 
                                                seed = seed)
        splits += [tmp_dict["train"]]
        tmp_dataset = tmp_dict["test"]
    
    splits += [tmp_dataset]

    return DatasetDict({k: v for k, v in zip(ratios_dict.keys(),
                                             splits)})



def mapping(objs: Iterable, 
            worker: Callable, 
            tool: Callable = lambda :tuple(),
            num_processes: int = 8):
    '''tools : a function building global variables costing time, like tokenizer'''
    from mptools import Marks
    from tqdm import tqdm
    import multiprocessing as mp
    def mapping_worker(wk, tl, get_queue: mp.Queue, to_queue: mp.Queue): 
        t = tl() 
        while True:
            obj = get_queue.get()
            if obj == Marks.stop: break
            to_queue.put(wk(obj, *t))

    to_queue = mp.Queue()
    get_queue = mp.Queue()

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=mapping_worker,
                       args=(worker, tool, to_queue, get_queue))
        p.start()
        processes += [p]

    total, ret = 0, []
    for obj in tqdm(objs, desc = "Preparing"): 
        total += 1
        to_queue.put(obj)
    for _ in range(num_processes): to_queue.put(Marks.stop)
    for _ in tqdm(range(total), desc = "Mapping"): ret += [get_queue.get()]
    for p in processes: p.join()

    return ret
    



def filtering(objs: Iterable, 
              filter: Callable, 
              tool: Callable = lambda :tuple(), 
              num_processes: int = 8):
    '''tools : a function building global variables costing time, like tokenizer'''
    from mptools import Marks
    from tqdm import tqdm
    import multiprocessing as mp
    def filtering_worker(ft, tl, get_queue: mp.Queue, to_queue: mp.Queue): 
        t = tl() 
        while True:
            obj = get_queue.get()
            if obj == Marks.stop: break
            if ft(obj, *t): to_queue.put(obj)
            else: to_queue.put(Marks.stop)

    to_queue = mp.Queue()
    get_queue = mp.Queue()

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=filtering_worker,
                       args=(filter, tool, to_queue, get_queue))
        p.start()
        processes += [p]

    total, ret = 0, []
    for obj in tqdm(objs, desc = "Preparing"): 
        total += 1
        to_queue.put(obj)
    for _ in range(num_processes): to_queue.put(Marks.stop)
    for _ in tqdm(range(total), desc="Filtering"): 
        obj = get_queue.get()
        if obj != Marks.stop: ret += [obj]
    for p in processes: p.join()

    return ret
    


def CTX_MAPPING(length, ks = [0, 1, 2, 4, 8, 16, 32, 64, 128]):
    length /= 1000
    distances = [abs(length-k) for k in ks]
    min_ids = min(distances)
    for dis, l in zip(reversed(distances), reversed(ks)):
        if min_ids == dis:
            return f"{l}k"
    






    
