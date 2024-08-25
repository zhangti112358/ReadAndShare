import os
from openai import OpenAI
import tiktoken
import json
import time
import numpy as np
import requests
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
from collections import OrderedDict

dir_this = os.path.dirname(os.path.abspath(__file__))

def siliconcloud_key():
    key_file = './.private/key.json'
    with open (key_file, 'r') as f:
        data = json.load(f)
    key = data['siliconflow']
    return key

def volcengine_key():
    key_file = './.private/key.json'
    with open (key_file, 'r') as f:
        data = json.load(f)
    key = data['volcengine']['key']
    return key

class TokensUtil(object):
    encoding_list = [
        'o200k_base',
        'cl100k_base',
    ]
    
    def __init__(self, encoding_type=encoding_list[1]):
        enc = tiktoken.get_encoding(encoding_type)
        assert enc.decode(enc.encode("hello world")) == "hello world"
        self.enc = enc
        
    def string2num(self, string):
        '''
        统计字符串的token数量
        '''
        return len(self.enc.encode(string))
    
    def str_split(self, string):
        '''
        字符串分割 查看分割后的各个字符
        '''
        tokens = self.enc.encode(string)
        for token in tokens:
            print(self.enc.decode([token]))

class SiliconCloud(object):
    '''
    硅基流动云服务
    '''
    @staticmethod
    def simple():
        client = OpenAI(api_key=siliconcloud_key(), base_url="https://api.siliconflow.cn/v1")

        response = client.chat.completions.create(
            model='meta-llama/Meta-Llama-3.1-8B-Instruct',
            messages=[
                {'role': 'user', 'content': "what is moore's law"},
            ],
            stream=True
        )

        for chunk in response:
            print(chunk.choices[0].delta.content, end='')

    def get_model_list(self, save_file=os.path.join(dir_this, 'model_list.json')):
        '''
        获取模型列表 并保存到文件
        '''
        url = "https://api.siliconflow.cn/v1/models"

        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {siliconcloud_key()}"
            }

        response = requests.get(url, headers=headers)

        print(response.text)
        
        model_list_dict = json.loads(response.text)
        with open(save_file, 'w') as f:
            json.dump(model_list_dict, f, indent=4)

class CloudServerTest(object):
    '''
    测试云服务
    '''
    def __init__(self) -> None:
        self.tokens_util = TokensUtil()
        pass
    
    def test_simple(self, model='Qwen/Qwen1.5-7B-Chat', content="what is moore's law?"):
        '''
        测试反应时间 即生成第一个tokens的时间
        '''
        
        client = OpenAI(api_key=siliconcloud_key(), base_url="https://api.siliconflow.cn/v1")
        
        result = []
        time_list = []
        time_start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'user', 'content': content},
            ],
            stream=True,
            max_tokens=100
        )

        for i, chunk in enumerate(response):
            time_list.append(time.time())
            result.append(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content, end='')
        print('')
        result_str = ''.join(result[1:-1])
        
        tokens_num = self.tokens_util.string2num(result_str)
        
        t_first = time_list[0] - time_start
        t_tps = tokens_num / (time_list[-1] - time_start)
        print(f'first:{t_first}, tps:{t_tps}, tokens:{tokens_num}')
        
        return t_first, t_tps, tokens_num
        
    def test_one_day(self):
        '''
        测试一天内耗时波动 每分钟请求一次
        '''
        
        save_file = os.path.join(dir_this, 'test_one_day.json')
        
        client = OpenAI(api_key=siliconcloud_key(), base_url="https://api.siliconflow.cn/v1")
        
        
        time_dict = OrderedDict()
        time_test_start = time.time()
        # 每分钟循环一次
        i_run = 0
        while (time.time() - time_test_start < 60*60*24):
            # 运行一次
            try:
                result = []
                time_list = []
                time_local = time.localtime()
                time_stamp = f'{time_local.tm_yday}_{time_local.tm_hour:02d}{time_local.tm_min:02d}{time_local.tm_sec:02d}'
                time_start = time.time()
                response = client.chat.completions.create(
                    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
                    messages=[
                        {'role': 'user', 'content': "what is moore's law?"},
                    ],
                    stream=True,
                    max_tokens=100
                )

                for i, chunk in enumerate(response):
                    time_list.append(time.time())
                    result.append(chunk.choices[0].delta.content)
                result_str = ''.join(result[1:-1])

                tokens_num = self.tokens_util.string2num(result_str)
                t_first = time_list[0] - time_start
                print(f'i_run:{i_run}, time_stamp:{time_stamp}, first:{t_first}, tokens:{tokens_num}')
                time_dict[time_stamp] = t_first
                # 写文件
                with open(save_file, 'w') as f:
                    json.dump(time_dict, f, indent=4)
            except:
                print('error')
                pass
            # 等待一分钟
            if time.time() - time_test_start < 60 * i_run:
                time.sleep(time_test_start + 60 * i_run - time.time())
            
            i_run += 1
    
    def test_one_day_plot(self):
        '''
        画出一天内的耗时波动
        '''
        save_file = os.path.join(dir_this, 'test_one_day.json')
        with open(save_file, 'r') as f:
            time_dict = json.load(f)
        
        time_value = list(time_dict.values())
        
        # 画图 time_value 的分布
        time_value_ms = [x * 1000 for x in time_value]
        plt.hist(time_value_ms, bins=3000)
        plt.title('第一个token的输出时间')
        plt.xlabel('时间（ms）')
        plt.ylabel('频率')
        # plt.show()
        plt.savefig(os.path.join(dir_this, 'test_one_day.png'), dpi=300)
        
        # 统计超过0.5s的比例
        time_value = np.array(time_value)
        print(f'超过0.5s的比例:{np.sum(time_value > 0.5) / len(time_value)}')
        print(f'最大值:{np.max(time_value)}')
        
        pass
    
    def test_time(self):
        '''
        测试不同模型的速度
        '''
        model_list = [
            'Pro/Qwen/Qwen2-1.5B-Instruct',
            'Pro/Qwen/Qwen2-7B-Instruct',
            '01-ai/Yi-1.5-34B-Chat-16K',
            'deepseek-ai/deepseek-llm-67b-chat',
            'meta-llama/Meta-Llama-3.1-405B-Instruct',
        ]
        
        time_dict = OrderedDict()
        
        for model in model_list:
            t_first, t_tps, tokens_num = self.test_simple(model=model)
            time_dict[model] = {
                'first': t_first,
                'tps': t_tps,
                'tokens': tokens_num
            }
        
        with open(os.path.join(dir_this, 'test_time.json'), 'w') as f:
            json.dump(time_dict, f, indent=4)

    def test_volcengine(self):
        '''
        测试Volcengine
        '''
        def simple_chat(model='skylark-chat', content="what is moore's law?"):
            client = OpenAI(api_key = volcengine_key(), base_url = "https://ark.cn-beijing.volces.com/api/v3",)
            time_start = time.time()
            response = client.chat.completions.create(
                model = model,  # your model endpoint ID
                messages = [
                    {"role": "user", "content": content},
                ],
                stream=True,
                max_tokens=100,
            )
            time_list = []
            result = []
            for i, chunk in enumerate(response):
                result.append(chunk.choices[0].delta.content)
                time_list.append(time.time())
                print(chunk.choices[0].delta.content, end='')
            print('')
            
            result_str = ''.join(result)

            tokens_num = self.tokens_util.string2num(result_str)

            t_first = time_list[0] - time_start
            t_tps = tokens_num / (time_list[-1] - time_start)
            print(f'first:{t_first}, tps:{t_tps}, tokens:{tokens_num}')

            return t_first, t_tps, tokens_num
        
        model_dict = {
            "Mistral-7B": "ep-20240826002045-4zppn",
            "Doubao-lite-4k": "ep-20240826002316-dpvbq",
            "Doubao-pro-32k": "ep-20240826001841-lx4v7",
        }
        
        time_dict = OrderedDict()
        for model_name, model in model_dict.items():
            t_first, t_tps, tokens_num = simple_chat(model=model)
            time_dict[model_name] = {
                'first': t_first,
                'tps': t_tps,
                'tokens': tokens_num
            }
        with open(os.path.join(dir_this, 'test_volcengine.json'), 'w') as f:
            json.dump(time_dict, f, indent=4)

if __name__ == '__main__':
    Ctest = CloudServerTest()
    # Ctest.test_one_day()
    # Ctest.test_one_day_plot()
    # Ctest.test_time()
    
    Ctest.test_volcengine()

    pass