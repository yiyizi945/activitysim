import pandas as pd
import collections.abc
import warnings
# hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import choicemodels
# Now import hyper
warnings.filterwarnings("ignore")


class Non_Mandatory_Tour_Frequency(object):
    def __init__(self, person_data_path=None, household_data_path=None, alternatives_path=None, chooser_data_path=None, model_expression=None):
        if person_data_path is None:        # 标定数据
            person_data_path = "../data/standard_non_mantatory_tour_frequency_data.csv"
        person_data = pd.read_csv(person_data_path)
        if household_data_path is None:     # 家庭数据
            household_data_path = "../data/override_households.csv"
        household_data = pd.read_csv(household_data_path)
        if alternatives_path is None:       # 备选集
            alternatives_path = "../data/alternatives_non_mandatory_tour_frequency.csv"
        alternatives = pd.read_csv(alternatives_path, index_col=0)
        if chooser_data_path is None:       # 选择者
            chooser_data_path = "../data/non_mandatory_frequency_choosers1.csv"
        chooser_data = pd.read_csv(chooser_data_path, index_col=0)
        if model_expression is None:        # 模型表达式
            # c tot_tours == 1&is_famale + is_famale*_shopping + is_famale*_othmaint + is_famale*_eatout + auto_ownership&tot_tours == 1 + auto_ownership&tot_tours == 2
            # model_expression = 'tot_tours + shopping + eatout + social'
            model_expression = 'tot_tours == 1 + tot_tours == 2 + tot_tours == 3 + tot_tours == 4 + tot_tours == 5 + shopping + eatout + social '
                                # is_female&(tot_tours == 1) + is_female&(tot_tours == 2) + is_female&(tot_tours == 3) + \
                                # is_female&(tot_tours == 4) + is_female&(tot_tours == 5)'

        self.alternatives = alternatives
        self.person_data = person_data
        self.household_data = household_data
        self.chooser_data = chooser_data
        self.model_expression = model_expression

    def person_data_process(self):
        # 个人表根据原数据添加新列
        # 是否有强制活动
        have_mandatory_tour = self.person_data['mandatory_tour_frequency'].notna()
        self.person_data['have_mandatory_tour'] = have_mandatory_tour
        # 是否为男性
        is_male = self.person_data['sex'].apply(lambda x: True if x == 1 else False)
        self.person_data['is_male'] = is_male
        # 是否为女性
        is_female = self.person_data['sex'] == 2
        self.person_data['is_female'] = is_female
        # 将布尔值转化为0和1
        self.person_data.replace({True: 1, False: 0}, inplace=True)
        return self.person_data

    def household_data_process(self):
        # 家庭表根据原数据添加新列
        # 家庭有无工作者
        have_worker = ~(self.household_data['num_workers'] == 0)
        self.household_data['have_worker'] = have_worker
        # 家庭收入分类

        # 将布尔值转化为0和1
        self.household_data.replace({True: 1, False: 0}, inplace=True)
        return self.household_data

    def alternatives_process(self):
        # 备选集根据原数据添加新列
        # 强制活动次数
        for i in range(0, 6):
            condition = self.alternatives['tot_tours'] == i
            self.alternatives[f'tot_tours == {i}'] = condition
        bool_series = (self.alternatives['_shopping'] >= 1)
        self.alternatives['shopping'] = bool_series
        bool_series = (self.alternatives['_eatout'] >= 1)
        self.alternatives['eatout'] = bool_series
        bool_series = (self.alternatives['_social'] >= 1)
        self.alternatives['social'] = bool_series
        # 将布尔值转化为0和1
        self.alternatives.replace({True: 1, False: 0}, inplace=True)
        self.alternatives.set_index(['alternatives_id'], inplace=True)
        return self.alternatives

    def person_merged(self):
        # 合并个人表及家庭表
        person_merged_data = pd.merge(self.person_data, self.household_data, on='household_id')
        self.person_merged_data = person_merged_data.set_index(['person_id'])
        return self.person_merged_data

    def df_split(self, df, split_col):
        groups = df.groupby(split_col)
        num_level = len(df[split_col].unique().tolist())  # 判断等级数（有时间分4类，有时候分6类）
        return groups, num_level

    def segment_fit(self, name, mct, model_expression):
        print('SPEC_SEGMENTS:', name)
        results = choicemodels.MultinomialLogit(data=mct,
                                                model_expression=model_expression,
                                                observation_id_col='person_id',
                                                choice_col='chosen',
                                                alternative_id_col='alternatives_id'
                                                )
        return results

    def segment_cal_pop(self, name, m, mct_simu):
        print('segment:', name)
        fitted_model = m.fit()   # 标定的模型
        prob = fitted_model.probabilities(mct_simu)  # 用标定模型去预测选择概率
        return prob

    def calibration_process(self):
        # 标定结果
        # 选择人员类型作为标定数据分段
        split_col = 'ptype'
        spec_segments, num_level = self.df_split(self.person_merged_data, split_col)
        print(spec_segments)
        for i in range(1, num_level+1):
            # 根据变量i来设置choosers分组变量名
            exec('spec_segments_{} = spec_segments.get_group({})'.format(i, i))
        # 首先要对选择集和备选集进行数据处理
        groups = spec_segments
        alternatives = self.alternatives
        simulated_data_dict = {}
        for name, group in groups:
            print(name, len(group))
        # group = group.get_group(name)#将group转换成df
            choosers = group   # 选择者
            choosers = choosers.fillna(0)
            # choosers = choosers.set_index('person_id')
            # alternatives = alternatives.set_index('alternatives_id')
            # print('choosers', choosers)
            # print('alternatives', alternatives)
            model_expression = self.model_expression
            merged_tb = choicemodels.tools.MergedChoiceTable(choosers, alternatives, chosen_alternatives='alternatives_id', sample_size=len(alternatives))
            # print(merged_tb.to_frame())
            results = self.segment_fit(name, merged_tb, model_expression)
            i = int(name)
            # 用字典接收结果
            simulated_data_dict[i] = results
            print(results.fit())
        self.simulated_data_dict = simulated_data_dict

    def simulated_selection(self):
        chooser_data = self.chooser_data
        df = chooser_data.fillna(0)
        split_col = 'ptype'
        spec_segments, num_level = self.df_split(df, split_col)
        for i in range(1, num_level+1):
            exec('spec_segments_{} = spec_segments.get_group({})'.format(i, i))
        groups = spec_segments
        # df = self.choosers.fillna(0)
        choice_list = {}
        alternatives = self.alternatives
        for name, group in groups:
            # print(name, len(group))
            choosers = group   # 选择者
            choosers = choosers.fillna(0)
            choosers = choosers.set_index('person_id')
            mct_simu = choicemodels.tools.MergedChoiceTable(choosers, alternatives, sample_size=82)
            i = int(name)
            m = self.simulated_data_dict[i]
            prob = self.segment_cal_pop(name, m, mct_simu)
            choice_list[i] = choicemodels.tools.simulation.monte_carlo_choices(prob)
            # choice_list = {key: value.to_frame() for key, value in choice_list.items()}
            # choice_list = pd.concat(choice_list.values(), keys=choice_list.keys())
            result_df = pd.concat([choice_list[i] for i in choice_list.keys()])
            print(result_df)
        result_df = pd.DataFrame(result_df)
        result_df.reset_index(inplace=True)
        non_mandatory_tour_person = pd.merge(result_df, self.alternatives, on='alternatives_id')
        non_mandatory_tour_person = pd.merge(non_mandatory_tour_person, chooser_data, on='person_id')
        non_mandatory_tour_person.to_csv('../output/non_mandatory_tour_frequency_person.csv')


if __name__ == '__main__':
    nmtf = Non_Mandatory_Tour_Frequency()
    nmtf.person_data_process()
    nmtf.household_data_process()
    nmtf.person_merged()
    nmtf.alternatives_process()
    # print('person_data\n', nmtf.person_data)
    # print('household_data\n', nmtf.household_data)
    # print('person_merged_data\n', nmtf.person_merged_data)
    # print('alternatives\n', nmtf.alternatives)
    nmtf.calibration_process()
    nmtf.simulated_selection()
