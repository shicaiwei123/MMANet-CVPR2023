"""
Function: Protocol rules for face anti-spoofing datasets
Author: AJ
Date: 2019/10/8
"""
import os, copy
# four protocols and for evey protocol, there are different sub protocols
class CASIA_Race_RDI:
    def __init__(self, protocol, mode):
        """
        Experiments based on RGB Depth and IR modal
        :param protocol:
            race_prot_rdi_1@Race(6): AF&AF-CA AF&AF-EA...
            race_prot_rdi_2@PAI(2): 12&12-14 14&14-12
            race_prot_rdi_3@Modal(6): R&R-D R&R-I...
            race_prot_rdi_4@(Race & PAI) for challenge
        :param mode: train, dev, test
        """
        protocol_dict = {}
        protocol_dict['race_prot_rdi_1'] = {
            'train': {
                'Race': [],
                'ID': list(range(0, 200)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
            'dev': {
                'Race': [],
                'ID': list(range(200, 300)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
            'test': {
                'Race': [],
                'ID': list(range(300, 500)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
        }
        for i in range(6):
            protocol_dict['race_prot_rdi_1@%d'%(i+1)] = copy.deepcopy(protocol_dict['race_prot_rdi_1'])
            protocol_dict['race_prot_rdi_1@%d'%(i+1)]['train']['Race'] = []
            protocol_dict['race_prot_rdi_1@%d'%(i+1)]['dev']['Race'] = []
            protocol_dict['race_prot_rdi_1@%d'%(i+1)]['test']['Race'] = []
            if i == 0:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(1)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(1)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(2)
            elif i == 1:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(1)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(1)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(3)
            elif i == 2:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(2)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(2)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(1)
            elif i == 3:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(2)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(2)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(3)
            elif i == 4:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(3)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(3)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(1)
            elif i == 5:
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['train']['Race'].append(3)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['dev']['Race'].append(3)
                protocol_dict['race_prot_rdi_1@%d' % (i + 1)]['test']['Race'].append(2)

        protocol_dict['race_prot_rdi_2'] = {
            'train': {
                'Race': [1, 2, 3],
                'ID': list(range(0, 200)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': []
            },
            'dev': {
                'Race': [1, 2, 3],
                'ID': list(range(200, 300)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': []
            },
            'test': {
                'Race': [1, 2, 3],
                'ID': list(range(300, 500)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': []
            },
        }
        for i in range(2):
            protocol_dict['race_prot_rdi_2@%d' % (i + 1)] = copy.deepcopy(protocol_dict['race_prot_rdi_2'])
            protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['train']['PAI'] = [1]
            protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['dev']['PAI'] = [1]
            protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['test']['PAI'] = [1]
            if i == 0:
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['train']['PAI'].append(2)
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['dev']['PAI'].append(2)
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['test']['PAI'].append(4)
            elif i == 1:
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['train']['PAI'].append(4)
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['dev']['PAI'].append(4)
                protocol_dict['race_prot_rdi_2@%d' % (i + 1)]['test']['PAI'].append(2)

        protocol_dict['race_prot_rdi_3'] = {
            'train': {
                'Race': [1, 2, 3],
                'ID': list(range(0, 200)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
            'dev': {
                'Race': [1, 2, 3],
                'ID': list(range(200, 300)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
            'test': {
                'Race': [1, 2, 3],
                'ID': list(range(300, 500)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2, 4]
            },
        }

        protocol_dict['race_prot_rdi_4'] = {
            'train': {
                'Race': [],
                'ID': list(range(0, 200)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 4]
            },
            'dev': {
                'Race': [],
                'ID': list(range(200, 300)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 4]
            },
            'test': {
                'Race': [],
                'ID': list(range(300, 500)),
                'AcqDevice': [1, 3],
                'Session': [1, 2],
                'PAI': [1, 2]
            },
        }
        for i in range(6):
            protocol_dict['race_prot_rdi_4@%d' % (i + 1)] = copy.deepcopy(protocol_dict['race_prot_rdi_4'])
            protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'] = []
            protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'] = []
            protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'] = []
            if i == 0:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(1)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(1)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(2)
            elif i == 1:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(1)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(1)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(3)
            elif i == 2:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(2)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(2)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(1)
            elif i == 3:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(2)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(2)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(3)
            elif i == 4:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(3)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(3)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(1)
            elif i == 5:
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['train']['Race'].append(3)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['dev']['Race'].append(3)
                protocol_dict['race_prot_rdi_4@%d' % (i + 1)]['test']['Race'].append(2)

        self.protocol_dict = protocol_dict
        if not (protocol in self.protocol_dict.keys()):
            print('error: Protocal should be ', list(self.protocol_dict.keys()))
            exit(1)
        self.protocol = protocol
        self.mode = mode
        self.protocol_info = protocol_dict[protocol][mode]

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_') ### 2_008_4_1_3
        if not len(name_split) == 5:
            return False
        [race_, id_, acqDevice_, session_, pai_] = [int(x) for x in name_split]
        if (race_ in self.protocol_info['Race']) and (id_ in self.protocol_info['ID']) and \
                (acqDevice_ in self.protocol_info['AcqDevice']) and (session_ in self.protocol_info['Session']) and \
                (pai_ in self.protocol_info['PAI']):
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        # print('********** Dataset Info **********')
        # print('Data:CASIA_Race, protocol:{}, Mode:{}'.format(self.protocol, self.mode))
        # print('All counts={} vs Protocal counts={}'.format(len(file_list), len(res_list)))
        # print('********** Protocol Info **********')
        # print('protocol_info={}'.format(self.protocol_info))
        # print('**********************************')
        return res_list

class CASIA_Mask_RDI:
    def __init__(self, protocol, mode):
        """
        Experiments based on RGB Depth and IR modal
        :param protocol:
            mask_prot_rdi_3@Modal(6): R&R-D R&R-I...
        :param mode: train, dev, test
        """
        protocol_dict = {}
        protocol_dict['mask_prot_rdi_3'] = {
            'train': {
                'Mask': [1, 2],
                'ID': list(range(81, 100)),
                'Dresses': [0, 1],
                'Glasses': [0, 1],
                'AcqDevice': [2],
                'Session': [1, 2],
                'PAI': [1]

            },
            'dev': {
                'Mask': [1, 2],
                'ID': list(range(51, 81)),
                'Dresses': [0, 1],
                'Glasses': [0, 1],
                'AcqDevice': [2],
                'Session': [1, 2],
                'PAI': [1]
            },
            'test': {
                'Mask': [1, 2],
                'ID': list(range(1, 100)),
                'Dresses': [0, 1],
                'Glasses': [0, 1],
                'AcqDevice': [2],
                'Session': [1, 2],
                'PAI': [2, 3, 4]
            },
        }
        self.protocol_dict = protocol_dict
        if not (protocol in self.protocol_dict.keys()):
            print('error: Protocal should be ', list(self.protocol_dict.keys()))
            exit(1)
        self.protocol = protocol
        self.mode = mode
        self.protocol_info = protocol_dict[protocol][mode]

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_') ### 2_007_1_1_2_1_1
        if not len(name_split) == 7:
            return False
        [mask_, id_, dress_,  glass_, acqDevice_, session_, pai_] = [int(x) for x in name_split]
        if (mask_ in self.protocol_info['Mask']) and (id_ in self.protocol_info['ID']) \
                and (dress_ in self.protocol_info['Dresses']) and (glass_ in self.protocol_info['Glasses']) and \
                (acqDevice_ in self.protocol_info['AcqDevice']) and (session_ in self.protocol_info['Session']) and \
                (pai_ in self.protocol_info['PAI']):
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('********** Dataset Info **********')
        print('Data:CASIA_Mask, protocol:{}, Mode:{}'.format(self.protocol, self.mode))
        print('All counts={} vs Protocal counts={}'.format(len(file_list), len(res_list)))
        print('********** Protocol Info **********')
        print('protocol_info={}'.format(self.protocol_info))
        print('**********************************')
        return res_list

# p = '3'
# CASIA_Mask_Data = CASIA_Mask_RDI(protocol='mask_prot_rdi_{}'.format(p), mode='test')
# print(CASIA_Mask_Data.protocol_info)