import random
import os,sys
import numpy as np
import pickle
import multiprocessing

from mindmappings.parameters import Parameters
from mindmappings.utils.parallelProcess import parallelProcess

class SingleDataGen:
    def __init__(self, model, parameters=Parameters(), path=None, num_files=None, samples_per_file=None, samples_per_problem=None):
        self.model = model
        self.parameters = parameters
        self.path = parameters.DATASET_UNPROCESSED_PATH if(path==None) else path
        self.num_files = parameters.DATASET_NUMFILES if(num_files==None) else num_files
        self.samples_per_file = parameters.DATASET_NUMSAMPLES_FILE if(samples_per_file==None) else samples_per_file
        self.samples_per_problem = parameters.DATASET_MAPPINGS_PER_PROBLEM if(samples_per_problem==None) else samples_per_problem
        self.bounds = self.parameters.random_problem_gen()
        self.costmodel = self.model(problem=self.bounds, parameters=self.parameters)
        self.oracle_cost = self.costmodel.getOracleCost(metric='RAW')
    
    def getDataset(self, index):
        """
            1. Creates a random problem.
            2. Samples a mapping from that.
            3. Generates data from that.

            Above steps are iterated until the number of samples requested are reached.
        """

        data_arr = []
        threadID = str(multiprocessing.current_process()._identity[0])

        for n in range(self.samples_per_file):
            success = False
            while not success:
                try:
                    mapping, cost = self.costmodel.getMapCost(metric='RAW', threadID=threadID)
                    success = True
                except Exception as e:
                    print(e)
                    success = False
            
            # Generate input vector
            input_vector = self.costmodel.getInputVector(mapping)
            
            # Cost vector is normalized to the oracle cost
            cost = [cost[i]/float(self.oracle_cost[i]) for i in range(len(cost))]

            data_arr.append([input_vector, cost])

            # Print Progress
            print("{0} mappings, completed for {1}".format(n, threadID))

        # name = self.path + 'data_' + str(index) + '.npy'
        # np.save(name, data_arr)
        name = self.path + 'data_' + str(index) + '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(data_arr, f)
        print('Wrote to ' + name)

        return None

    def run(self):
        """
            Main File to generate data.
        """
        # Setup Path
        if(not os.path.isdir(self.path)):
            print("Creating the dataset path at {0}".format(self.path))
            os.makedirs(self.path)

        # Call threads in parallel to write the data
        # Processed = Parallel(n_jobs=-1)(delayed(getDataset)(path, ind, samplesperFile) for ind in range(numFiles))
        print("We will run 1 problems, with {0} mappings in it.".format(self.samples_per_problem))
        work = [ind for ind in range(self.num_files)]
        parallelProcess(self.getDataset, work, num_cores=None)

        print("All Done!")

        with open(os.path.join(self.path, 'problem.log'), 'w') as f:
            f.write(str(self.bounds))

        return None
            