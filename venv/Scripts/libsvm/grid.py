class Grid(object):
    def __int__(self):
        pass
    def grid(self, grid_file, ori_file, tar_file):
        command = grid_file + r"/tools/grid.py -v 5 " + ori_file + ' > ' + tar_file
        return command
if __name__ == '__main__':
    import os
    os.system("chmod 755 ~/circRna-disease-association/libsvm-3.24/libsvm-3.24/tools/grid.py")
    grider = Grid()
    for i in ['60', '2000']:
        for feature_strategy in [1, 2]:
            for seed in [0, 1, 2, 3, 4]:
                command = grider.grid("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                                              "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_train_scale.txt".format(str(i), str(feature_strategy), str(seed)),
                                              "~/circRna-disease-association/data/libsvm/grid/libsvm_samples_{}_strategy{}_seed{}_train_grid.txt".format(str(i), str(feature_strategy), str(seed)))
                os.system(command)
                print("libsvm_samples_{}_strategy{}_seed{}_train_scale.txt completed".format(str(i), str(feature_strategy), str(seed)))
