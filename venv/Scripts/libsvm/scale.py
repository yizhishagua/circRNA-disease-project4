class Scale(object):
    def __int__(self):
        pass
    def scale(self, scale_file, ori_file, tar_file):
        '''
        缩放函数
        :param scale_file:libsvm所在的文件夹
        :param ori_file: 需要缩放的文件
        :param tar_file: 将缩放后的文件放在哪个位置用什么名字保存
        :return:
        '''
        command = scale_file + r"/svm-scale -l 0 -u 1 " + ori_file + " > " + tar_file
        return command
    def train_scale(self, scale_file, ori_file, tar_file, save_path):
        '''
        缩放函数并保存缩放规则
        :param scale_file: libsvm所在的文件夹
        :param ori_file: 需要缩放的文件
        :param tar_file: 将缩放后的文件放在哪个位置用什么名字保存
        :param save_path: 缩放规则所保存的位置
        :return:
        '''
        command = scale_file + r"/svm-scale"+ " -s "+save_path+ " -l 0 -u 1 " + ori_file + " > " + tar_file
        return command
    def test_scale(self, scale_file, ori_file, scale_rule, tar_file):
        '''
        缩放函数并使用-s对输入文件用某个scale规则恢复
        :param scale_file: libsvm所在的文件夹
        :param ori_file:需要缩放的文件
        :param scale_rule:缩放规则
        :param tar_file:缩放后的文件放的位置
        :param is_test: 是否使用缩放规则
        :return:
        '''
        command = scale_file + r"/svm-scale"+ " -r "+ scale_rule + " " + ori_file +  " > " + tar_file
        return command
if __name__ == '__main__':
    import os
    scaler = Scale()
    # 缩放训练集特征到[0, 1]之间, 并保存缩放规则

    for i in ['60', '2000']:
        for feature_strategy in [1, 2]:
            for seed in [0, 1, 2, 3, 4]:
                command1 = scaler.train_scale("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                                              "~/circRna-disease-association/data/libsvm/libsvm_samples_{}_strategy{}_seed{}_train.txt".format(str(i), str(feature_strategy), str(seed)),
                                              "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_train_scale.txt".format(str(i), str(feature_strategy), str(seed)),
                                              "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_scale_rule.txt".format(str(i), str(feature_strategy), str(seed)))

                command2 = scaler.test_scale("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                                             "~/circRna-disease-association/data/libsvm/libsvm_samples_{}_strategy{}_seed{}_test.txt".format(str(i), str(feature_strategy), str(seed)),
                                             "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_scale_rule.txt".format(str(i), str(feature_strategy), str(seed)),
                                             "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_test_scale.txt".format(str(i), str(feature_strategy), str(seed)))

                command3 = scaler.test_scale("~/circRna-disease-association/libsvm-3.24/libsvm-3.24",
                                             "~/circRna-disease-association/data/libsvm/not_selected_negative_samples_{}_strategy{}_seed{}.txt".format(str(i), str(feature_strategy), str(seed)),
                                             "~/circRna-disease-association/data/libsvm/scale/libsvm_samples_{}_strategy{}_seed{}_scale_rule.txt".format(str(i), str(feature_strategy), str(seed)),
                                             "~/circRna-disease-association/data/libsvm/scale/not_selected_negative_samples_{}_strategy{}_seed{}_scale.txt".format(str(i), str(feature_strategy), str(seed)))
                os.system(command1)
                os.system(command2)
                os.system(command3)