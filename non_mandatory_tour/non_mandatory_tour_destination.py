import pandas as pd
import collections.abc
import warnings
import numpy as np
import openmatrix as omx
from patsy import dmatrix
# hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import choicemodels
# Now import hyper
warnings.filterwarnings("ignore")


class Non_Mandatory_Tour_Destination(object):
    def __init__(self, tour_data_path=None,  alternatives_path=None, jinkai_alternatives_path=None, dist_data_path=None, jinkai_dist_data_path=None, model_expression=None, chooser_data_path=None):
        if tour_data_path is None:      # 标定数据
            tour_data_path = "../data/standard_non_mantatory_tour_destination_data.csv"
        tour_data = pd.read_csv(tour_data_path)
        if alternatives_path is None:   # 备选集
            alternatives_path = "../data/PARCEL_AI.csv"
        alternatives = pd.read_csv(alternatives_path, index_col=0)
        if jinkai_alternatives_path is None:   # 备选集
            jinkai_alternatives_path = '../data/alternatives_non_mandatory_tour_destination.csv'
        jinkai_alternatives = pd.read_csv(jinkai_alternatives_path)
        if dist_data_path is None:      # 计算系数距离矩阵
            skims = omx.open_file("../data/skims.omx")
        dist_data = skims['/DIST'][:]
        dist_data = pd.DataFrame(dist_data, index=pd.RangeIndex(start=1, stop=dist_data.shape[0] + 1, name='origin'),
                                 columns=pd.RangeIndex(start=1, stop=dist_data.shape[1] + 1, name='TAZ'))
        if jinkai_dist_data_path is None:      # 计算系数距离矩阵
            jinkai_dist_data_path = '../distance/distance_matrix.csv'
        jinkai_dist_data = pd.read_csv(jinkai_dist_data_path, index_col=0)
        jinkai_dist_data.columns.name = 'parcel_id'
        if chooser_data_path is None:   # 选择者
            chooser_data_path = "../output/non_mandatory_tour_frequency_person.csv"
        chooser_data = pd.read_csv(chooser_data_path, index_col=0)
        if model_expression is None:    # 模型表达式
            model_expression = 'des_0_1 + des_1_2 + AI_STAND'
            # model_expression= ' des_0_1 + des_1_2 + des>2 +AI_STAND '
        self.alternatives = alternatives
        self.tour_data = tour_data
        self.dist_data = dist_data
        self.jinkai_alternatives = jinkai_alternatives
        self.jinkai_dist_data = jinkai_dist_data
        self.chooser_data = chooser_data
        self.model_expression = model_expression

    def chooser_data_process(self):
        chooser_data = self.chooser_data
        new_columns = ['tour_id', 'person_id', 'tour_type']
        new_data = {'tour_id': [], 'person_id': [], '_shopping': [], '_medical': [], '_other': [], '_eatout': [], '_social': [], 'tot_tours': [], 'tour_type': []}
        # 遍历每行数据，生成新的列
        for index, row in chooser_data.iterrows():
            person_id = row['person_id']
            tour_counter = 0
            for col in ['_shopping', '_medical', '_other', '_eatout', '_social']:
                for _ in range(int(row[col])):    
                    tour_counter += 1
                    tour_id = f"{person_id}{tour_counter}"
                    tour_type = col[1:]  # 获取活动类型，去除下划线前缀
                    new_data['tour_id'].append(tour_id)
                    new_data['person_id'].append(person_id)
                    new_data['tour_type'].append(tour_type)
        new_df = pd.DataFrame(new_data, columns=new_columns)
        chooser_data = pd.merge(new_df, chooser_data, on=['person_id'])
        self.chooser_data = chooser_data
        return self.chooser_data

    def find_dis(dist_data, o, d):    # 从距离矩阵中选取值
        o = int(o)
        d = int(d)
        if dist_data.loc[dist_data.index == o, d].values[0]:
            return dist_data.loc[dist_data.index == o, d].values[0]
        else:
            return None

    def df_split(self, df, split_col):
        groups = df.groupby(split_col)
        num_level = len(df[split_col].unique().tolist())  # 判断等级数
        return groups, num_level

    def segment_fit(self, name, mct, model_expression):
        print('SPEC_SEGMENTS:', name)
        results = choicemodels.MultinomialLogit(data=mct,
                                                model_expression=model_expression,
                                                observation_id_col='tour_id',
                                                choice_col='chosen',
                                                alternative_id_col='TAZ')
        return results

    def segment_cal_pop(self, name, m, mct_simu):
        print('segment:', name),
        fitted_model = m.fit()   # 标定的模型
        raw_results = fitted_model.get_raw_results()
        # prop = fitted_model.probabilities(mct_simu)  # 用标定模型去预测选择概率
        data = mct_simu
        df = data.to_frame()
        numalts = len(self.alternatives['parcel_id'])  # TO DO - make this an official MCT param
        dm = dmatrix(self.model_expression, data=df)
        # utility is sum of data values times fitted betas
        fitted_parameters = raw_results['fit_parameters']['Coefficient'].tolist()
        u = np.dot(fitted_parameters, np.transpose(dm))
        # reshape so axis 0 lists alternatives and axis 1 lists choosers
        u = np.reshape(u, (numalts, u.size // numalts), order='F')
        # scale the utilities to make exponentiation easier
        u = u - u.max(axis=0)
        exponentiated_utility = np.exp(u)
        sum_exponentiated_utility = np.sum(exponentiated_utility, axis=0)
        probs = exponentiated_utility / sum_exponentiated_utility
        # convert back to ordering of the input data
        prob = probs.flatten(order='F')
        df['prob'] = prob  # adds indexes
        prob = df.prob
        return prob

    def calibration_process(self):
        # 标定结果
        # 选择活动类型作为标定数据分段
        split_col = 'tour_type'
        spec_segments, num_level = self.df_split(self.tour_data, split_col)
        print(spec_segments)
        print(num_level)
        grouped_list = [(key) for key, group in spec_segments]
        # print(spec_segments.get_group('eatout'))
        # print(grouped_list[0].strip('\''))
        print(grouped_list)
        for i in range(1, num_level+1):
            # 根据变量i来设置choosers分组变量名
            # 首先要对选择集和备选集进行数据处理
            # exec('spec_segments_{} = spec_segments.get_group({})'.format(i, i))
            exec('spec_segments_{} = spec_segments.get_group("{}")'.format(i, grouped_list[i-1]))  #字符串类型的活动类型为分段依据
        groups = spec_segments
        alternatives = self.alternatives
        alternatives = alternatives.set_index('parcel_id')
        simulated_data_dict = {}
        for name, group in groups:
            # print(name, len(group))
            choosers = group   # 选择者
            choosers = choosers.fillna(0)
            choosers.rename(columns={'destination': 'parcel_id'}, inplace=True)
            model_expression = self.model_expression
            merged_tb = choicemodels.tools.MergedChoiceTable(choosers, alternatives, chosen_alternatives='parcel_id', sample_size=None)
            merged_tb = merged_tb.to_frame()
            merged_tb = merged_tb.reset_index()  # (长表的双重索引暂时设置为普通列索引)
            # 从距离矩阵中给长表匹配距离
            merged_tb['des'] = merged_tb[['origin', 'parcel_id']].apply(lambda x: Non_Mandatory_Tour_Destination.find_dis(self.dist_data, x['origin'], x['parcel_id']), axis=1) #从距离矩阵中匹配距离值
            merged_tb['des_0_1'] = (merged_tb['des'] >= 0) & (merged_tb['des'] <= 1)
            merged_tb['des_1_2'] = (merged_tb['des'] > 1) & (merged_tb['des'] <= 2)
            merged_tb['des>2'] = (merged_tb['des'] >= 2)
            merged_tb.set_index(['tour_id', 'parcel_id'], inplace=True)   # (设置双重索引)
            # print(merged_tb.to_frame()
            print(merged_tb)
            results = self.segment_fit(name, merged_tb, model_expression)
            i = name
            # 用字典接收结果
            simulated_data_dict[i] = results
            print(results.fit())
        self.simulated_data_dict = simulated_data_dict

    def simulated_selection(self):
        choosers = self.chooser_data
        # choosers = choosers.iloc[:10000, :]
        df = choosers.fillna(0)
        split_col = 'tour_type'
        spec_segments, num_level = self.df_split(df, split_col)
        grouped_list = [(key) for key, group in spec_segments]
        # print(spec_segments.get_group('eatout'))
        # print(grouped_list[0].strip('\''))
        print(grouped_list)
        for i in range(1, num_level+1):
            exec('spec_segments_{} = spec_segments.get_group("{}")'.format(i,  grouped_list[i-1]))
        groups = spec_segments
        choice_list = {}
        choosers_list = {}
        alternatives = self.jinkai_alternatives
        alternatives = alternatives.set_index('parcel_id')
        for name, group in groups:
            choosers = group   # 选择者
            choosers = choosers.fillna(0)
            choosers = choosers.set_index('tour_id')
            mct_simu = choicemodels.tools.MergedChoiceTable(choosers, alternatives, sample_size=100)
            # print(f'mct_simu的类型: {type(mct_simu)}')
            mct_simu = mct_simu.to_frame()
            mct_simu = mct_simu.reset_index()  # (长表暂时移除双重索引)
            jinkai_dist_data = self.jinkai_dist_data
            # print(jinkai_dist_data.index)
            # print(jinkai_dist_data.columns)
            jinkai_dist_data.columns = jinkai_dist_data.columns.astype(int)
            # print(jinkai_dist_data.columns)
            # 取前4000个数据测试
            # mct_simu = mct_simu.iloc[:4000, :]
        # 从距离矩阵中匹配距离值
            mct_simu['des'] = mct_simu[['origin', 'parcel_id']].apply(lambda x: Non_Mandatory_Tour_Destination.find_dis(jinkai_dist_data, x['origin'], x['parcel_id']), axis=1)
            # print(mct_simu)
            mct_simu['des_0_1'] = (mct_simu['des'] >= 0) & (mct_simu['des'] <= 1)
            mct_simu['des_1_2'] = (mct_simu['des'] > 1) & (mct_simu['des'] <= 2)
            mct_simu['des>2'] = (mct_simu['des'] >= 2)
            mct_simu.set_index(['tour_id', 'parcel_id'], inplace=True)
            mct_simu = choicemodels.tools.MergedChoiceTable.from_df(mct_simu)  # 将数据表转换为ChoiceTable格式
            i = name
            m = self.simulated_data_dict[i]
            prob = self.segment_cal_pop(name, m, mct_simu)
            print(prob)
            choice_list[i] = choicemodels.tools.simulation.monte_carlo_choices(prob)
            print(choice_list[i])
            choosers_list[i] = choosers
        choosers_df = pd.concat([choosers_list[i] for i in choosers_list.keys()])
        result_df = pd.concat([choice_list[i] for i in choice_list.keys()])
        result_df = pd.merge(result_df, choosers_df, on=['tour_id'])
        print(result_df)
        result_df.to_csv('../output/non_mandatory_tour_destination_tour.csv')


if __name__ == '__main__':
    nmtd = Non_Mandatory_Tour_Destination()
    nmtd.chooser_data_process()
    print('chooser_data\n', nmtd.chooser_data)
    nmtd.calibration_process()
    nmtd.simulated_selection()
