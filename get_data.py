# -*- coding:utf-8 -*-
"""
数据获取模块 - 从彩票网站爬取历史数据
Updated Version
"""
import argparse
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from config import name_path, data_file_name

# 配置请求超时
requests.packages.urllib3.disable_warnings()

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择爬取数据: 双色球(ssq)/大乐透(dlt)")
parser.add_argument('--output', default="data", type=str, help="数据输出目录")
args = parser.parse_args()


def get_url(name):
    """
    获取数据源 URL
    :param name: 玩法名称 (ssq 或 dlt)
    :return: 基础 URL 和路径模板
    """
    url = f"https://datachart.500.com/{name}/history/"
    path = "newinc/history.php?start={}&end="
    return url, path


def get_current_number(name):
    """
    获取最新一期号码
    :param name: 玩法名称
    :return: 最新期号
    """
    try:
        url, _ = get_url(name)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(f"{url}history.shtml", headers=headers, verify=False, timeout=10)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
        logger.info(f"获取最新期号：{current_num}")
        return current_num
    except Exception as e:
        logger.error(f"获取期号失败: {e}")
        return None


def spider(name, start, end, mode):
    """
    爬取历史数据
    :param name: 玩法名称 (ssq 或 dlt)
    :param start: 开始期号
    :param end: 结束期号
    :param mode: 模式 ('train' 或 'predict')
    :return: DataFrame
    """
    try:
        url, path = get_url(name)
        full_url = f"{url}{path.format(start)}{end}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"正在爬取数据: {full_url}")
        r = requests.get(url=full_url, headers=headers, verify=False, timeout=10)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        
        trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        data = []
        
        for tr in trs:
            item = {}
            try:
                tds = tr.find_all("td")
                item["期数"] = tds[0].get_text().strip()
                
                if name == "ssq":
                    # 双色球：6个红球 + 1个蓝球
                    for i in range(6):
                        item[f"红球_{i+1}"] = int(tds[i+1].get_text().strip())
                    item["蓝球"] = int(tds[7].get_text().strip())
                    
                elif name == "dlt":
                    # 大乐透：5个红球 + 2个蓝球
                    for i in range(5):
                        item[f"红球_{i+1}"] = int(tds[i+1].get_text().strip())
                    for j in range(2):
                        item[f"蓝球_{j+1}"] = int(tds[6+j].get_text().strip())
                
                data.append(item)
            except Exception as e:
                logger.warning(f"解析行数据失败: {e}")
                continue
        
        if len(data) == 0:
            logger.warning("未获取到有效数据！")
            return None
        
        df = pd.DataFrame(data)
        
        if mode == "train":
            # 训练模式：保存数据到文件
            data_path = name_path[name]["path"]
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            
            output_file = os.path.join(data_path, data_file_name)
            df.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"数据已保存到: {output_file}")
        
        logger.info(f"成功获取 {len(df)} 期数据")
        return df
        
    except Exception as e:
        logger.error(f"爬取数据失败: {e}")
        return None


def run(name):
    """
    主函数：获取并保存数据
    :param name: 玩法名称
    """
    logger.info(f"【{name_path[name]['name']}】开始获取数据...")
    
    current_number = get_current_number(name)
    if current_number is None:
        logger.error("无法获取最新期号，请检查网络连接")
        return
    
    logger.info(f"【{name_path[name]['name']}】最新一期期号：{current_number}")
    logger.info(f"正在获取【{name_path[name]['name']}】数据。。。")
    
    data = spider(name, 1, current_number, "train")
    
    if data is not None and len(data) > 0:
        logger.info(f"【{name_path[name]['name']}】数据准备就绪，共 {len(data)} 期")
        logger.info("下一步可执行: python run_train_model.py --name {}".format(name))
    else:
        logger.error(f"【{name_path[name]['name']}】数据获取失败！")


if __name__ == "__main__":
    if not args.name or args.name not in ["ssq", "dlt"]:
        raise Exception("玩法名称无效！请选择 'ssq' (双色球) 或 'dlt' (大乐透)")
    run(name=args.name)
