import abc

import gtimer as gt
from maple.core.rl_algorithm import BaseRLAlgorithm
from maple.data_management.replay_buffer import ReplayBuffer
from maple.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            evolution_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            eval_epoch_freq=1,
            expl_epoch_freq=1,
            eval_only=False,
            no_training=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            evolution_data_collector,
            replay_buffer,
            eval_epoch_freq=eval_epoch_freq,
            expl_epoch_freq=expl_epoch_freq,
            eval_only=eval_only,
            no_training=no_training,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        gt.reset_root()

    def _train(self):
        if self.min_num_steps_before_training > 0 and not self._eval_only:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                # TODO delect 100
                self.min_num_steps_before_training/100,
                discard_incomplete_paths=True, #False,
            )
            self.evo_data_collector.population.successful_record(init_expl_paths)
            self.evo_data_collector.population.success_variation_and_crossover()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            evo_paths = self.evo_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=True,  # False,
                success_memory=True,

            )
            self.replay_buffer.add_paths(evo_paths)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs + 1),
                save_itrs=True,
        ):
            for pre_epoch_func in self.pre_epoch_funcs:
                pre_epoch_func(self, epoch)

            if epoch % self._eval_epoch_freq == 0:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            gt.stamp('evaluation sampling')
            if epoch % 1 == 0:
                self.evo_data_collector.population.success_variation_and_crossover()
                self.evo_data_collector.population.var2success()

            if not self._eval_only:
                for _ in range(self.num_train_loops_per_epoch):
                    evo_paths = self.evo_data_collector.collect_new_paths(
                        self.max_path_length,
                        # TODO delect 100
                        self.min_num_steps_before_training/100,
                        discard_incomplete_paths=True,  # False,
                        success_memory=True,

                    )
                    self.replay_buffer.add_paths(evo_paths)
                    evo_paths = self.evo_data_collector.collect_new_paths(
                        self.max_path_length,
                        # TODO delect 100
                        self.min_num_steps_before_training/100,
                        discard_incomplete_paths=True,  # False,
                        success_memory=False,

                    )
                    self.replay_buffer.add_paths(evo_paths)

                    if epoch % self._expl_epoch_freq == 0:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=True, #False,
                        )
                        gt.stamp('exploration sampling', unique=False)
                        self.evo_data_collector.population.successful_record(new_expl_paths)
                        self.replay_buffer.add_paths(new_expl_paths)
                        print(self.evo_data_collector.population.successful_memory_fitness)
                        print(sum(self.evo_data_collector.population.successful_memory_fitness) / len(
                            self.evo_data_collector.population.successful_memory_fitness))

                        gt.stamp('data storing', unique=False)



                    if not self._no_training:
                        self.training_mode(True)
                        for _ in range(self.num_trains_per_train_loop):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        self.training_mode(False)

            self._end_epoch(epoch)
