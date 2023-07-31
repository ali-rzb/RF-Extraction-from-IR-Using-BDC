__copyright__ = """

    Copyright 2022 Ali Roozbehi

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import Functions.GlobalUtils as g_utils
import Functions.LocalDB as db
import numpy as np, os, random
import matplotlib.pyplot as plt
from copy import deepcopy

class GeneticAlgorithm:
    def __init__(self, Fitness_Function, layers_range, len_chr = None, end_check_range=10
                 ,n_chr=100, selection_ratio=0.5, cross_ratio=0.59, mutation_ratio=0.01, abs_ranks = True):
        
        self.Fitness_Function = Fitness_Function
        self.len_chr = len_chr
        self.n_chr = n_chr
        self.end_check_range = end_check_range
        self.selection_ratio = selection_ratio
        self.cross_ratio = cross_ratio
        self.mutation_ratio = mutation_ratio
        self.layers_range = layers_range
        self.all_generations = []
        self.iteration = 0
        self.proceed = True
        self.abs_ranks = abs_ranks
                
        if self.cross_ratio + self.mutation_ratio != self.selection_ratio:
            raise(Exception('mutation plus cross ratio have to be selection ratio!'))
        
        if self.mutation_ratio > self.cross_ratio:
            raise(Exception('cross ratio must be bigger than mutation!'))

    def Procces(self, iterations=100, load_db = False, log_path = None, log_name = None):
        timer = g_utils.timer()
        Chromosomes = None
        if log_path != None:
            if log_name == None:
                log_name = ('GA.{}.{}.{}.txt').format(self.len_chr,self.n_chr, self.layers_range)
                
            self.log_full_name = log_path + '/' + log_name
            if os.path.isfile(self.log_full_name):
                if load_db:
                    try:
                        Generations = db.get_db(self.log_full_name, self.generation)
                        Chromosomes = Generations[-1].chromosomes
                        all_chr = []
                        for g in [[c[0] for c in g.chromosomes] for g in Generations]:
                            all_chr = all_chr + g
                        self.all_generations = g_utils.remove_duplicates(all_chr)
                        load_db = True
                    except:
                        load_db = False
                        Chromosomes = None
            else:    
                db.create(self.log_full_name, self.generation, True)
                load_db = False
            
                
        
        self.n_selection = int(self.n_chr * self.selection_ratio)
        self.n_cross = int(self.n_chr * self.cross_ratio)
        self.n_mutate = int(self.n_chr * self.mutation_ratio)
        
        self.n_train_count_local = 0
        if Chromosomes == None:
            timer.start()
            Chromosomes = self.make_chromosomes()
            timer.stop()
            db.insert(self.log_full_name, self.generation(
                        n_chr = self.n_chr
                        ,best_rank = best_chr[1]
                        ,best_chr = best_chr[0]
                        ,local_best_rank = best_chr[1]
                        ,chromosomes = Chromosomes
                        ,passed_time = timer.labeled_time
                    ))
        else:
            delta = len(Chromosomes) - (self.n_selection + self.n_cross + self.n_mutate)
            if delta > 0:
                Chromosomes = sorted(Chromosomes, key = lambda x:x[1])
                for _ in range(delta):
                    Chromosomes.remove(Chromosomes[0])
            elif delta < 0:
                Chromosomes = Chromosomes + self.make_chromosomes(delta)
        
        print('\n\n' + str(len(Chromosomes)) + ' Chromosomes are ready!\n\n')
        
        best_chr = self.get_best(Chromosomes)
        

        all_ranks = []
        all_ranks.append(best_chr[1])
        self.iteration = 0
        
        # Start Evolutionation
        timer = g_utils.timer()
        times = []
        
        best_rank = min([abs(t[1]) for t in Chromosomes])
        best_chr = [t for t in Chromosomes if abs(t[1]) == best_rank][0]
        
        while self.proceed:
            timer.start()
            self.iteration = self.iteration + 1
            self.n_train_count_local = 0
            if self.iteration == iterations:
                break
            
            timer.start()
            
            Selected = self.selection(self.n_selection, Chromosomes)
            Cross = self.cross(Selected, self.n_cross)
            Mutate = self.mutation(Cross, self.n_mutate)
            Chromosomes = Selected + Cross + Mutate
            timer.stop()
            del Selected, Cross, Mutate
            times.append(round(timer.exact_passed_time/1000))
            
            # check for best way
            local_best_rank = min([abs(t[1]) for t in Chromosomes])
            local_best_chr = [t for t in Chromosomes if abs(t[1]) == local_best_rank][0]
            
            all_ranks.append(local_best_rank)
            print('\niteration : {} , best in this generation → Chromosome : {} , Fittness : {}, time : {}\n'.format(
                self.iteration, local_best_chr, local_best_rank, times[-1]))
            if local_best_rank < best_rank:
                best_rank = local_best_rank
                best_chr = local_best_chr
                print('\nBest Guess → {}th Generation, Chromosome : {} , Fittness : {}'.format(self.iteration, best_chr[0], best_chr[1]))
            
            if log_path != None:
                db.insert(self.log_full_name, self.generation(
                    n_chr = self.n_chr
                    ,best_rank = best_rank
                    ,best_chr = best_chr
                    ,local_best_rank = local_best_rank
                    ,chromosomes = Chromosomes
                    ,passed_time = timer.labeled_time
                ))
            
            # Check for Termination
            if self.end_condition(all_ranks):
                print('\nno improvements in {} cycles.'.format(self.end_check_range, best_rank))
                print('Best Guess → Chromosome : {} , Fittness : {}'.format(best_chr[0], best_chr[1]))
                break
            
        print('\ntotal passed time : {}'.format(sum(times)))
               
    def cross(self, selected, n):
        _temp_list = [c[0] for c in selected].copy()
        output = []
        i = 0
        while True:
            counter = 0
            while True:
                _random = random.sample(_temp_list, 2)
                target_a = _random[0]
                target_b = _random[1]
                temp_a = []
                temp_b = []
                gen = random.randint(1, self.len_chr - 1)
                for j, l in enumerate(self.layers_range):
                    temp_a.append(target_a[j][:gen] + target_b[j][gen:])
                    temp_b.append(target_b[j][:gen] + target_a[j][gen:])
                    
                if counter > 500:
                    temp_a ,temp_b = [], []
                    for l in self.layers_range:
                        temp_a.append([random.choice(range(l[0], l[1]+1)) for _ in range(self.len_chr)])
                    for l in self.layers_range:
                        temp_b.append([random.choice(range(l[0], l[1]+1)) for _ in range(self.len_chr)])
                
                if counter > 10000:
                    print('out of choices!')
                    self.proceed = False
                    break
                    
                if (temp_a not in self.all_generations) and (temp_b not in self.all_generations):        
                    break
                else:
                    counter = counter + 1
            
            if not self.proceed:
                break
                
            _temp_list.remove(target_a)
            _temp_list.remove(target_b)
            
            self.all_generations.append(temp_a)
            self.all_generations.append(temp_b)
            
            self.n_train_count_local = self.n_train_count_local + 1
            # self.pBar.update()
            print('generation : {}, train count : {}/{}, Crossing'.format(self.iteration, self.n_train_count_local, self.n_cross + self.n_mutate))
            output.append([temp_a, self.Fitness_Function(temp_a)])
            
            self.n_train_count_local = self.n_train_count_local + 1
            # self.pBar.update()
            print('generation : {}, train count : {}/{}, Crossing'.format(self.iteration, self.n_train_count_local, self.n_cross + self.n_mutate))
            output.append([temp_b, self.Fitness_Function(temp_b)])
            
            i = i + 2
            if i == n:
                break
            elif i == n - 1:
                counter = 0
                while True:
                    target = random.sample(_temp_list, 1)
                    gen = random.randint(1, self.len_chr - 2)
                    for j, l in enumerate(self.layers_range):
                        temp = target[j][:gen] + list(reversed(target[j][gen:]))
                    
                    if counter > 500:
                        temp = []
                        for l in self.layers_range:
                            temp.append([random.choice(range(l[0], l[1]+1)) for _ in range(self.len_chr)])
                        
                    if counter > 1000:
                        print('out of choices!')
                        self.proceed = False
                        break
    
                    if temp not in self.all_generations:
                        break
                    else:
                        counter = counter + 1
                
                if not self.Procces:
                    break
                
                _temp_list.remove(target)
                output.append([temp, self.Fitness_Function(temp)])
                self.n_train_count_local = self.n_train_count_local + 1
                # self.pBar.update()
                print('generation : {}, train count : {}/{}, Crossing'.format(self.iteration, self.n_train_count_local, self.n_cross + self.n_mutate))
                self.all_generations.append(temp)
                break
        
        del _temp_list, i, _random, target_a, target_b, temp_a, temp_b, gen, counter
        return output

    def mutation(self, parents, n):
        _temp_list = [c[0] for c in parents].copy()
        output = []
        for i in range(n):
            counter = 0
            while True:
                target_chr = random.sample(_temp_list, 1)[0]
                mutated = deepcopy(target_chr)
                random_indexes = random.sample(range(self.len_chr), 2)
                for j, l in enumerate(self.layers_range):
                    t = mutated[j][random_indexes[0]]
                    mutated[j][random_indexes[0]] = mutated[j][random_indexes[1]]
                    mutated[j][random_indexes[1]] = t
                
                if counter > 500:
                    mutated = []
                    for l in self.layers_range:
                        mutated.append([random.choice(range(l[0], l[1]+1)) for _ in range(self.len_chr)])
                
                if counter > 1000:
                    print('out of choices!')
                    self.proceed = False
                    break
                
                if mutated not in self.all_generations:
                    break
                else:
                    counter = counter + 1
            if not self.proceed:
                break
            self.n_train_count_local = self.n_train_count_local + 1
            # self.pBar.update()
            print('generation : {}, train count : {}/{}, Mutating'.format(self.iteration, self.n_train_count_local, self.n_cross + self.n_mutate))
            self.all_generations.append(mutated)
            output.append([mutated, self.Fitness_Function(mutated)])
            _temp_list.remove(target_chr)
            
        del _temp_list, counter, target_chr, mutated, random_indexes
        return output

    def selection(self, n_childs, parents):
        if self.abs_ranks:
            sorted_list = sorted(parents, key=lambda x: abs(x[1]))
        else:
            sorted_list = sorted(parents, key=lambda x: x[1])
        return sorted_list[:n_childs]

    def get_best(self, chromosomes):
        if self.abs_ranks:
            return [c for c in chromosomes if c[1] == min([abs(t[1]) for t in chromosomes])][0]
    
    def end_condition(self, all_ranks):
        length = len(all_ranks)
        result = False
        if length > self.end_check_range:
            if min(all_ranks[:length - self.end_check_range]) == min(all_ranks[length - self.end_check_range: ]):
                result = True
        return result

    def make_chromosomes(self, n = None):
        data = []
        if n == None:
            n = self.n_chr
        for _ in range(n):
            while True:
                chr = []
                for l in self.layers_range:
                    chr.append([random.choice(range(l[0], l[1]+1)) for _ in range(self.len_chr)])    
                if chr not in self.all_generations:
                    break
            
            self.n_train_count_local = self.n_train_count_local + 1
            print('generation : {}, train count : {}/{}, Making Chromosomes'.format(self.iteration, self.n_train_count_local, self.n_chr))
            # self.pBar.update()
            fitness = self.Fitness_Function(chr)
            self.all_generations.append(chr)
            data.append([chr, fitness])
        return data
    
    class generation:
        def __init__(self, n_chr: int
                     ,best_rank: float, best_chr: list, local_best_rank : float,chromosomes: list, passed_time : str, id=None):
            self.n_chr = n_chr
            self.best_rank = best_rank
            self.best_chr = best_chr
            self.local_best_rank = local_best_rank
            self.chromosomes = chromosomes
            self.passed_time = passed_time
            self.id = id
    
    @staticmethod
    def plot_hist(generations : list, inverse_ranks = False, normalize = False):
        data_max = [g.best_rank for g in generations]
        # data_min = [max([c[1] for c in g.chromosomes]) for g in generations]
        
        if inverse_ranks:
            # data_min = [1/r for r in data_min]
            data_max = [1/r for r in data_max]

        
        # plt.plot(range(len(data_min)), data_min)
        plt.plot(range(len(data_max)), data_max)
        plt.grid(True)
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(len(data_max)), np.arange(len(data_max)))
        plt.tight_layout()
        if normalize:
            plt.ylim([0, 1])
        plt.xlim([0, len(data_max)-1])
        plt.show()
