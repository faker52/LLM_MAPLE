import random
import torch
import numpy as np
from random import choice
from copy import deepcopy


VAR_NUMS = 5
class Population(object):

    def __init__(self, variant):
        self.variant = variant['population']
        self.prior_action = None
        self.successful_memory_fitness = []
        self.var_memory_fitness = []
        self.champion_index = -1
        # self.pop_initial(lens)
        self.successful_memory = []
        self.var_memory = []
        self.max_reward = variant['max_reward']
        self.min_reward = -1000
        self.successful_memory_max_size = self.variant['successful_memory_max_size']
        self.var_memory_max_size = self.variant['var_memory_max_size']
        self.success_batch_size = self.variant['success_batch_size']
        self.var_batch_size = self.variant['var_batch_size']
        self._top_var = 0
        self._size_var = 0
        self._top_success = 0
        self._size_success = 0

        # UCB
        self.ucb_var_upper = []
        self.ucb_var_chosen = []
        self.ucb_var_sum_chosen = 0

        self.successful_memory_initial()
        # self.success_variation_and_crossover()

    def successful_memory_initial(self):
        if self.prior_action == None:
            for i in range(self.successful_memory_max_size):
                self.successful_memory.append(None)
                self.successful_memory_fitness.append(self.min_reward)
            return
        for i in range(10):
            action = deepcopy(self.prior_action)
            self.successful_memory.append(action)
            self.successful_memory_fitness.append(self.max_reward)
            self._sucess_advance()

    # def pop_initial(self, lens):
    #     # TOdo 修改 0
    #     self.fitness = np.zeros(self.pop_size)
    #     for i in range(self.pop_size):
    #         a = []
    #         for t in range(lens):
    #             a.append(random.randint(0, 4))
    #         self.pop.append(a)



    #
    # def variation_and_crossover(self, p_var=0.1, p_cross=0.1):
    #     index = np.argsort(self.fitness)
    #     var_index = index[0:4]
    #     cross_index = index[-5:-3]
    #
    #     for d in var_index:
    #         if random.random() < p_var:
    #             rand = random.random()
    #             if rand < 1 / 3:
    #                 if len(self.pop[d]) > 3:
    #                     t = random.randint(0, len(self.pop[d]) - 1)
    #                     self.pop[d].pop(t)
    #             elif 1 / 3 < rand < 2 / 3:
    #                 if len(self.pop[d]) < 15:
    #                     t = random.randint(0, len(self.pop[d]))
    #                     self.pop[d].insert(t, random.randint(0, 4))
    #
    #             else:
    #                 t = random.randint(0, len(self.pop[d]) - 1)
    #                 self.pop[d][t] = random.randint(0, 4)
    #
    #     if random.random() < p_cross:
    #         a = cross_index[0]
    #         b = cross_index[1]
    #
    #         if len(self.pop[a]) > 2 and len(self.pop[b]) > 3:
    #             cross_len_a = int(len(self.pop[a]) / 4) + 1
    #             cross_len_b = int(len(self.pop[b]) / 4) + 1
    #
    #             cross_insert_a = random.randint(0, len(self.pop[a]) - cross_len_a)
    #             cross_insert_b = random.randint(0, len(self.pop[b]) - cross_len_b)
    #
    #             a_copy = self.pop[a].copy()
    #             for t in range(cross_len_a):
    #                 self.pop[a].pop(cross_insert_a)
    #             for t in range(cross_len_b):
    #                 self.pop[a].insert(cross_insert_a, self.pop[b][cross_insert_b + cross_len_b - t - 1])
    #             for t in range(cross_len_b):
    #                 self.pop[b].pop(cross_insert_b)
    #             for t in range(cross_len_a):
    #                 self.pop[b].insert(cross_insert_b, a_copy[cross_insert_a + cross_len_a - t - 1])

    #
    # def rl_to_pop(self, paths):
    #     skill_names = [path['skill_names'] for path in paths]
    #     fitiness_path = [sum(path["rewards"]) for path in paths]
    #     max_fitness_index = np.argmax(fitiness_path)
    #     skill_name = skill_names[max_fitness_index]
    #     rl_skills = []
    #     for skill in skill_name:
    #         if skill == 'Atomic':
    #             rl_skills.append(0)
    #         elif skill == 'Reach':
    #             rl_skills.append(1)
    #         elif skill == 'Grasp':
    #             rl_skills.append(2)
    #         elif skill == 'Release':
    #             rl_skills.append(3)
    #         elif skill == 'Open':
    #             rl_skills.append(4)
    #     replace_index = np.argmin(self.fitness)
    #     self.pop[replace_index] = rl_skills
    #     self.fitness[replace_index] = fitiness_path[max_fitness_index]

    #
    # def get_champion_index(self):
    #     return np.argmax(self.fitness)

    def successful_record(self, paths):
        skill_names = [path['skill_names'] for path in paths]
        fitiness_path = [sum(path["rewards"]) for path in paths]
        fitiness_path = [x for t in fitiness_path for x in t]
        min_sort = np.argsort(fitiness_path)
        for index in min_sort[::-1]:
            success_min = np.argsort(self.successful_memory_fitness)[0]
            if fitiness_path[index] > self.successful_memory_fitness[success_min]:
                skills = skill_names[index]
                rl_skills = []
                for skill in skills:
                    if skill == 'atomic':
                        rl_skills.append(0)
                    elif skill == 'reach':
                        rl_skills.append(1)
                    elif skill == 'grasp':
                        rl_skills.append(2)
                    elif skill == 'push':
                        rl_skills.append(3)
                    elif skill == 'open':
                        rl_skills.append(4)
                self.successful_memory[success_min] = deepcopy(rl_skills)
                self.successful_memory_fitness[success_min] = deepcopy(fitiness_path[index])
                self._sucess_advance()


    def success_variation_and_crossover(self, p_var=0.1, p_cross=0.1):
        if len(self.successful_memory) == 0:
            return

        max_index = np.random.randint(0, self._size_success, size=VAR_NUMS)
        if self._size_success > VAR_NUMS:
            max_sort = np.argsort(self.successful_memory_fitness)
            max_index = max_sort[-VAR_NUMS:]
        for i in max_index:
            temp = deepcopy(self.successful_memory[i])
            if random.random() < p_var:
                rand = random.random()
                if rand < 1 / 3:
                    if len(temp) > 3:
                        t = random.randint(0, len(temp) - 1)
                        temp.pop(t)
                elif 1 / 3 < rand < 2 / 3:
                    if len(temp) < 15:
                        t = random.randint(0, len(temp))
                        temp.insert(t, random.randint(0, 4))
                else:
                    t = random.randint(0, len(temp) - 1)
                    temp[t] = random.randint(0, 4)
            self.updata_var_memory(temp)

        if random.random() < p_cross:
            if self._size_success >= 2:
                max_sort = np.argsort(self.successful_memory_fitness)
                a = max_sort[-1]
                b = max_sort[-2]
            else:
                return

            if len(self.successful_memory[a]) > 2 and len(self.successful_memory[b]) > 3:
                cross_len_a = int(len(self.successful_memory[a]) / 4) + 1
                cross_len_b = int(len(self.successful_memory[b]) / 4) + 1

                cross_insert_a = random.randint(0, len(self.successful_memory[a]) - cross_len_a)
                cross_insert_b = random.randint(0, len(self.successful_memory[b]) - cross_len_b)

                a_copy = deepcopy(self.successful_memory[a])
                b_copy = deepcopy(self.successful_memory[b])
                for t in range(cross_len_a):
                    a_copy.pop(cross_insert_a)
                for t in range(cross_len_b):
                    a_copy.insert(cross_insert_a, self.successful_memory[b][cross_insert_b + cross_len_b - t - 1])
                for t in range(cross_len_b):
                    b_copy.pop(cross_insert_b)
                for t in range(cross_len_a):
                    b_copy.insert(cross_insert_b, self.successful_memory[a][cross_insert_a + cross_len_a - t - 1])
                self.updata_var_memory(a_copy)
                self.updata_var_memory(b_copy)



    def var2success(self):
        success_min = np.argsort(self.successful_memory_fitness)[0]
        var_max = np.argsort(self.var_memory_fitness)[-1]
        if self._size_success < self.successful_memory_max_size:
            self.successful_memory_fitness[self._top_success] = deepcopy(self.var_memory_fitness[var_max])
            self.successful_memory[self._top_success] = deepcopy(self.var_memory[var_max])
            self._sucess_advance()
            print("add success", self.var_memory_fitness[var_max])
        elif self.var_memory_fitness[var_max] > self.successful_memory_fitness[success_min]:
            print(self.successful_memory_fitness[success_min], "------>", self.var_memory_fitness[var_max])
            memory = deepcopy(self.successful_memory[success_min])
            fitness = deepcopy(self.successful_memory_fitness[success_min])
            self.successful_memory_fitness[success_min] = deepcopy(self.var_memory_fitness[var_max])
            self.successful_memory[success_min] = deepcopy(self.var_memory[var_max])
            self.var_memory[var_max] = memory
            self.var_memory_fitness[var_max] = fitness
            self.ucb_var_sum_chosen = self.ucb_var_sum_chosen - self.ucb_var_chosen[var_max]
            self.ucb_var_chosen[var_max] = 0
            self.ucb_var_upper[var_max] = 0



    def get_batch_from_var_memory(self):
        self.ucb_var_upper = [self.var_memory_fitness[item] + self.calculate_delta(self.ucb_var_sum_chosen, item) for item in range(len(self.var_memory))]
        min_sort = np.argsort(self.ucb_var_upper)
        if len(self.var_memory) < self.var_batch_size:
            idxs = np.random.randint(0, len(self.var_memory), size=self.var_batch_size)
        else:
            idxs = min_sort[-self.var_batch_size:]
        batch = [self.var_memory[i] for i in idxs]
        for i in idxs:
            self.ucb_var_chosen[i] = self.ucb_var_chosen[i] + 1
            self.ucb_var_sum_chosen = self.ucb_var_sum_chosen + 1
        print("ucb select index", idxs)
        print("VAR_LEN", len(self.var_memory))
        print(self.var_memory_fitness)
        print(self.ucb_var_upper)
        return batch, idxs

    def get_batch_from_success_memory(self):
        idxs = np.random.randint(0, self._size_success, size=self.success_batch_size)
        batch = [self.successful_memory[i] for i in idxs]
        return batch, idxs


    def _var_advance(self):
        self._top_var = (self._top_var + 1) % self.var_memory_max_size
        if self._size_var < self.var_memory_max_size:
            self._size_var += 1


    def _sucess_advance(self):
        self._top_success = (self._top_success + 1) % self.successful_memory_max_size
        if self._size_success < self.successful_memory_max_size:
            self._size_success += 1

    def calculate_delta(self, T, item):
        if self.ucb_var_chosen[item] == 0:
            return 10000
        else:
            return np.sqrt(4 * np.log(T) / self.ucb_var_chosen[item])



    def updata_var_memory(self,temp):
        min_sort = np.argsort(self.var_memory_fitness)
        min_index = 0

        if len(self.var_memory) < self.var_memory_max_size:
            self.var_memory.append(deepcopy(temp))
            self.var_memory_fitness.append(self.min_reward)
            self.ucb_var_chosen.append(0)
            self.ucb_var_upper.append(0)
            self._var_advance()
        else:
            index = min_sort[min_index]
            self.ucb_var_sum_chosen = self.ucb_var_sum_chosen - self.ucb_var_chosen[index]
            self.ucb_var_chosen[index] = 0
            self.ucb_var_upper[index] = 0
            self.var_memory[index] = deepcopy(temp)
            self.var_memory_fitness[index] = self.min_reward









