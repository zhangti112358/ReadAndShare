import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from collections import OrderedDict

dir_this = os.path.dirname(os.path.abspath(__file__))

# TSMC 晶圆价格
wafer_price_process = OrderedDict(
    {
        '90':1650,
        '65':1937,
        '40':2275,
        '28':2891,
        '20':3677,
        '12':3984,
        '10':5992,
        '7' :9346,
        '5' :16988,
    }
)

# 芯片信息
# https://en.wikipedia.org/wiki/Transistor_count#GPUs
chip_info_process = OrderedDict(
    {
        # https://en.wikipedia.org/wiki/GeForce_8_series#GeForce_8800_series
        '90':{
            'company':'NVIDIA',
            'name':'G80',
            'area':480,
            'transistor':681e6,
            'power':146,
            'year': 2006,
        },
        # https://en.wikipedia.org/wiki/GeForce_200_series#GeForce_200_series
        '65':{
            'company':'NVIDIA',
            'name':'GT280',
            'area':576,
            'transistor':1400e6,
            'power':236,
            'year':2008,
        },
        # https://en.wikipedia.org/wiki/GeForce_400_series#Products
        '40':{
            'company':'NVIDIA',
            'name':'GTX480',
            'area':529,
            'transistor':3e9,
            'power':250,
            'year':2010,
        },
        # https://en.wikipedia.org/wiki/Nvidia_Tesla
        # https://www.techpowerup.com/gpu-specs/tesla-k20c.c564
        '28':{
            'company':'NVIDIA',
            'name':'K20',
            # 微架构 GK110
            'area':561,
            'transistor':7.08e9,
            'power':225,
            'year':2012,
        },
        # https://en.wikipedia.org/wiki/Apple_A8
        # https://www.cpu-monkey.com/en/cpu-apple_a8
        # '20':{
        #     'company':'Apple',
        #     'name':'A8',
        #     'area':89,
        #     'transistor':2e9,
        #     'power':5,
        # },
        # https://en.wikipedia.org/wiki/GeForce_16_series#Architecture
        '12':{
            'company':'NVIDIA',
            'name': 'GTX1650',
            'area':284,
            'transistor':6.6e9,
            'power':80,
            'year':2019,
        },
        # https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623
        '7':{
            'company':'NVIDIA',
            'name':'A100',
            'area':826,
            'transistor':54.2e9,
            'power':250,
            'year':2020,
        },
        # '8':{
        #     'company':'NVIDIA',
        #     'name':'RTX3090',
        #     'area':826,
        #     'transistor':54.2e9,
        #     'power':250,
        # },
        # https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899
        '5':{
            'company':'NVIDIA',
            'name':'H100',
            'area':814,
            'transistor':80e9,
            'power':350,
            'year':2022,
        },
        # '5':{
        #     'company':'NVIDIA',
        #     'name':'4090',
        #     'area':608.5,
        #     'transistor':76.3e9,
        #     'power':450,
        # },
        
    }
)

# 计算晶体管价格和功耗
def t_price_power():
    
    # 晶圆面积
    wafer_area = np.pi * (300/2)**2
    
    process_list = []
    transistor_price_list = []
    transistor_power_list = []
    name_list = []
    
    for key in wafer_price_process:
        if key in chip_info_process:
            wafer_price = wafer_price_process[key]
            chip_info = chip_info_process[key]
            area = chip_info['area']
            transistor = chip_info['transistor']
            power = chip_info['power']
            
            # 晶体管价格
            transistor_price = wafer_price / wafer_area * area / transistor
            transistor_power = power / transistor
            
            print(f'制程：{key} 晶体管价格：{transistor_price:.2e} 晶体管功耗：{transistor_power:.2e}')
            
            process_list.append(key)
            transistor_price_list.append(1/transistor_price)
            transistor_power_list.append(1/transistor_power)
            
            if chip_info['name'] == 'H100':
                name_list.append(f'{chip_info["name"]}\n4nm {chip_info["year"]}')
            else:
                name_list.append(f'{chip_info["name"]}\n{key}nm {chip_info["year"]}')
                

    # 绘制晶体管价格和功耗 横轴是功耗 纵轴是价格 使用对数坐标
    plt.figure()
    plt.plot(transistor_power_list, transistor_price_list, 'o-', markersize=10)
    for i in range(len(process_list)):
        plt.text(transistor_power_list[i]*1.2, transistor_price_list[i]/1.1, name_list[i])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(transistor_power_list[0]/1.5, transistor_power_list[-1]*2.4) # x轴范围
    plt.xlabel('晶体管/W')
    plt.ylabel('晶体管/$')
    plt.title('晶体管价格和功耗 随制程变化')
    # plt.show()
    
    plt.savefig(os.path.join(dir_this, 'transistor_price_power.png'), dpi=300)
    
    

if __name__ == '__main__':
    
    t_price_power()
    
    
    pass