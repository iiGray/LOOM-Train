from linktrain.core.loop import FitLoop
from linktrain.core.distributed import RayRole, RayResourcePoolManager


class ReinforcedLoop(FitLoop):
    def __init__(
        self, 
        data_module: "DataModule",
        train_module: "TrainModule", 
        rollout_module: "RolloutModule",
        critic_module: "CriticModule", 
    
    ):
        super().__init__()

        with RayResourcePoolManager(strict_mode = True) as manager:
            with manager.create_pool(
                nprocs = 8,
                ncpus_per_proc = 2,
                ngpus_per_proc = 1,
                roles = [RayRole.DATA, RayRole.ACTOR, RayRole.ROLLOUT]
            ) as pool:
                self.data_module = pool.init_worker(data_module)
                self.train_module = pool.init_worker(train_module)

            with manager.create_pool(
                nprocs = 8,
                ncpus_per_proc = 2,
                ngpus_per_proc = 1,
                roles = [RayRole.CRITIC]
            ) as pool:
                self.critic_module = pool.init_worker(critic_module)


    def fit(self):
        ...
        