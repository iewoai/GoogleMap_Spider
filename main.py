import requests
import random
import json
import numpy as np
import math

######## 谷歌地图爬虫（requests版）
######## 目标：搜素company，改坐标，获取全球所有公司信息
###### 思路：从一点出发，选取该点最优Z（以整数调整），以该点的切片边长或1d表达式做距离，经纬等长移动度差，移动后的每一点重复该步骤。
#### 注（纯个人理解）：
# 1. 同一经纬度，不同缩放倍数的公司数目不一样，很难选最优值
# 2. 当没有最优值，即该坐标附近无公司的情况，需设置默认度幅（经纬度增减幅度）
# 3. 移动时为前后左右移动，即固定经度或纬度，正负移动另一个
# 4. 固纬度动经度，度差受纬度和距离影响；固经度动纬度，度差仅受距离影响
# 5. 最优值选缩放倍数大于等于12的（纬度幅0.82739）小于等于18的（纬度幅0.00899）中公司数目中最多的倍数，以其1d（见url解析）做距离，上下左右移动。当12-18公司数都为0时，取默认度幅

##### 瓦片地图原理参考资料：
# https://segmentfault.com/a/1190000011276788
# https://blog.csdn.net/mygisforum/article/details/8162751
# https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/
# https://blog.csdn.net/mygisforum/article/details/13295223

#### url解析1
## url = 'https://www.google.com/maps/search/company/@43.8650268,-124.6372106,3.79z'
## https://www.google.com/maps/search/company/@X,Y,Z
## X 纬度，范围[-85.05112877980659, 85.05112877980659]
## Y 经度，范围 [-180, 180]
## Z 缩放倍数，范围[2, 21]
## Z=2 切片正方形边长为20037508.3427892
#### url解析2
## https://www.google.com/search?tbm=map&authuser=0&hl={1}&gl={2}&pb=!4m9!1m3!1d{3}!2d{4}!3d{5}!2m0!3m2!1i784!2i644!4f13.1!7i20{6}!10b1!12m8!1m1!18b1!2m3!5m1!6e2!20e3!10b1!16b1!19m4!2m3!1i360!2i120!4i8!20m57!2m2!1i203!2i100!3m2!2i4!5b1!6m6!1m2!1i86!2i86!1m2!1i408!2i240!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e4!2b1!4b1!9b0!22m6!1sg3qzXsG-JpeGoATHyYKQBw%3A1!2zMWk6MSx0OjExODg3LGU6MCxwOmczcXpYc0ctSnBlR29BVEh5WUtRQnc6MQ!7e81!12e3!17sg3qzXsG-JpeGoATHyYKQBw%3A110!18e15!24m46!1m12!13m6!2b1!3b1!4b1!6i1!8b1!9b1!18m4!3b1!4b1!5b1!6b1!2b1!5m5!2b1!3b1!5b1!6b1!7b1!10m1!8e3!14m1!3b1!17b1!20m2!1e3!1e6!24b1!25b1!26b1!30m1!2b1!36b1!43b1!52b1!55b1!56m2!1b1!3b1!65m5!3m4!1m3!1m2!1i224!2i298!26m4!2m3!1i80!2i92!4i8!30m28!1m6!1m2!1i0!2i0!2m2!1i458!2i644!1m6!1m2!1i734!2i0!2m2!1i784!2i644!1m6!1m2!1i0!2i0!2m2!1i784!2i20!1m6!1m2!1i0!2i624!2m2!1i784!2i644!34m13!2b1!3b1!4b1!6b1!8m3!1b1!3b1!4b1!9b1!12b1!14b1!20b1!23b1!37m1!1e81!42b1!47m0!49m1!3b1!50m4!2e2!3m2!1b1!3b0!65m0&q={7}&oq={8}&gs_l=maps.12...0.0.1.12357296.1.1.0.0.0.0.0.0..0.0....0...1ac..64.maps..1.0.0.0...3041.&tch=1&ech=1&psi=g3qzXsG-JpeGoATHyYKQBw.1588820611303.1
## hl={1}为语言，常用zh-CN或en
## g1={2}为当前所在国家地区缩写
## !1d{3}为1d，与缩放倍数有关，相邻两整数缩放倍数之间1d成1/2关系，缩放倍数越高值越小，最大为94618532.08008283，猜测为当前所在地图切片边长或周长等边长正比关系
## !2d{4}为2d，经度
## !3d{5}为3d，纬度
## {6}为搜素结果页数，格式为!8i+page，其中page默认为20的倍数，且page为0时（即第一页时），url中无!8i字段
## q={7}&oq={8}，基本都是搜素词

##### 经纬度和距离互转
# 来源：https://blog.csdn.net/qq_37742059/article/details/101554565
earth_radius = 6370.856  # 地球平均半径，单位km，最简单的模型往往把地球当做完美的球形，这个值就是常说的RE
math_2pi = math.pi * 2
pis_per_degree = math_2pi / 360  # 角度一度所对应的弧度数，360对应2*pi

# 纯纬度上，度数差转距离
def lat_degree2km(dif_degree=.0001, radius=earth_radius):
    """
    通过圆环求法，纯纬度上，度数差转距离(km)，与中间点所处的地球上的位置关系不大
    :param dif_degree: 度数差, 经验值0.0001对应11.1米的距离
    :param radius: 圆环求法的等效半径，纬度距离的等效圆环是经线环，所以默认都是earth_radius
    :return: 这个度数差dif_degree对应的距离，单位km
    """
    return radius * dif_degree * pis_per_degree

# 纯纬度上，距离值转度数
def lat_km2degree(dis_km=111, radius=earth_radius):
    """
    通过圆环求法，纯纬度上，距离值转度数(diff)，与中间点所处的地球上的位置关系不大
    :param dis_km: 输入的距离，单位km，经验值111km相差约(接近)1度
    :param radius: 圆环求法的等效半径，纬度距离的等效圆环是经线环，所以默认都是earth_radius
    :return: 这个距离dis_km对应在纯纬度上差多少度
    """
    return dis_km / radius / pis_per_degree

def lng_degree2km(dif_degree=.0001, center_lat=22):
    """
    通过圆环求法，纯经度上，度数差转距离(km)，纬度的高低会影响距离对应的经度角度差，具体表达式为：
    :param dif_degree: 度数差
    :param center_lat: 中心点的纬度，默认22为深圳附近的纬度值；为0时表示赤道，赤道的纬线环半径使得经度计算和上面的纬度计算基本一致
    :return: 这个度数差dif_degree对应的距离，单位km
    """
    # 修正后，中心点所在纬度的地表圆环半径
    real_radius = earth_radius * math.cos(center_lat * pis_per_degree)
    return lat_degree2km(dif_degree, real_radius)

def lng_km2degree(dis_km=1, center_lat=22):
    """
    纯经度上，距离值转角度差(diff)，单位度数。
    :param dis_km: 输入的距离，单位km
    :param center_lat: 中心点的纬度，默认22为深圳附近的纬度值；为0时表示赤道。赤道、中国深圳、中国北京、对应的修正系数分别约为： 1  0.927  0.766
    :return: 这个距离dis_km对应在纯经度上差多少度
    """
    # 修正后，中心点所在纬度的地表圆环半径
    real_radius = earth_radius * math.cos(center_lat * pis_per_degree)
    return lat_km2degree(dis_km, real_radius)

agents = [
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/532.5 (KHTML, like Gecko) Chrome/4.0.249.0 Safari/532.5',
    'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.9 (KHTML, like Gecko) Chrome/5.0.310.0 Safari/532.9',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/534.7 (KHTML, like Gecko) Chrome/7.0.514.0 Safari/534.7',
    'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/9.0.601.0 Safari/534.14',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/10.0.601.0 Safari/534.14',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.20 (KHTML, like Gecko) Chrome/11.0.672.2 Safari/534.20", "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1']

# 获得一页的所有公司信息
def get_allcom(response):
    page_source = response.text
    big_dict = json.loads(page_source.replace('/*""*/', ''))
    d_str = big_dict['d'].replace(")]}'", '').strip()
    d_list = json.loads(d_str)
    company_list = d_list[0][1]
    result_list = list()
    if company_list:
        for company in company_list:
            try:
                temp_dict = dict()
                temp_dict['companyName'] = company[14][11] if company and company[14] and company[14][11] else None
                temp_dict['url'] = company[14][7][0] if company and company[14] and company[14][7] and company[14][7][
                    0] else None
                temp_dict['address'] = company[14][18] if company and company[14] and company[14][18] else None
                temp_dict['phone'] = company[14][3][0] if company and company[14] and company[14][3] and company[14][3][
                    0] else None
                temp_dict['category'] = '>'.join(company[14][13]) if company and company[14] and company[14][13] else None
                if temp_dict['companyName']:
                    result_list.append(temp_dict)
            except Exception as e:
                continue
    return result_list

# 调整1d，moudle1：步幅0.01，moudle0：步幅1
def get_1d(module = 1, offset = 0.01):
    a = []
    # z=2 1d值
    ori = 94618532.08008283
    a.append([2,ori])
    for i in range(2,22):
        if i > 2:
            ori = (ori / 2)
        else:
            ori = ori
        if module == 1:
            for j in np.arange(0,1,offset):
                if (i+j) > 2 and (i+j) <= 21:
                    # print((i+j),ori - ori*j/2)
                    a.append([(i+j),ori - ori*j/2])
        elif module == 0:
            if [i,ori] not in a:
                a.append([i,ori])
    return dict(a)

# 调整经纬度
# d3为纬度，范围[-85.05112877980659, 85.05112877980659]
# d2为经度，范围 [-180, 180]
def get_23d(d2, d3, dis = 5775.056889653493):
    # 默认经纬度步幅，取缩放倍数16的1d
    # 取值范围
    lat_range = [-85.05112877980659, 85.05112877980659]
    lon_range = [-180, 180]

    lat = lat_km2degree(int(dis/1000))
    lon = lng_km2degree(int(dis/1000),d3)
    # print(lat, lon)
    up = (d2, d3+lat) if d3+lat > -85.05112877980659 and d3+lat < 85.05112877980659 else None
    down = (d2, d3-lat) if d3-lat > -85.05112877980659 and d3-lat < 85.05112877980659 else None
    left = (d2-lon, d3) if d2-lon > -180 and d2-lon < 180 else None
    right = (d2+lon, d3) if d2+lon > -180 and d2+lon < 180 else None

    return[up, down, left, right]

def get_com(d2, d3):

    # 记录不同缩放倍数的公司数目
    com_num = {}
    # 倍数字典
    d1_dict = get_1d(0)
    for d1_multiple in d1_dict:
        d1 = d1_dict[d1_multiple]
        # print('目前倍数%d：'%d1_multiple)
        url = 'https://www.google.com/search?tbm=map&authuser=0&hl=en&pb=!4m12!1m3!1d{}!2d{}!3d{}'
        page = 0
        all_result = []
        while True:
            pb = '!2m3!1f0!2f0!3f0!3m2!1i784!2i644!4f13.1!7i20{}!10b1!12m8!1m1!18b1!2m3!5m1!6e2!20e3!10b1!16b1!19m4!2m3!1i360!2i120!4i8!20m57!2m2!1i203!2i100!3m2!2i4!5b1!6m6!1m2!1i86!2i86!1m2!1i408!2i240!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e4!2b1!4b1!9b0!22m5!1s5mKzXrHAJNSXr7wP5u-akAQ!4m1!2i5600!7e81!12e30!24m46!1m12!13m6!2b1!3b1!4b1!6i1!8b1!9b1!18m4!3b1!4b1!5b1!6b1!2b1!5m5!2b1!3b1!5b1!6b1!7b1!10m1!8e3!14m1!3b1!17b1!20m2!1e3!1e6!24b1!25b1!26b1!30m1!2b1!36b1!43b1!52b1!55b1!56m2!1b1!3b1!65m5!3m4!1m3!1m2!1i224!2i298!26m4!2m3!1i80!2i92!4i8!30m28!1m6!1m2!1i0!2i0!2m2!1i458!2i644!1m6!1m2!1i734!2i0!2m2!1i784!2i644!1m6!1m2!1i0!2i0!2m2!1i784!2i20!1m6!1m2!1i0!2i624!2m2!1i784!2i644!31b1!34m13!2b1!3b1!4b1!6b1!8m3!1b1!3b1!4b1!9b1!12b1!14b1!20b1!23b1!37m1!1e81!42b1!46m1!1e2!47m0!49m1!3b1!50m13!1m8!3m6!1u17!2m4!1m2!17m1!1e2!2z6Led56a7!4BIAE!2e2!3m2!1b1!3b0!59BQ2dBd0Fn!65m0&q=company&tch=1&ech=4&psi=5mKzXrHAJNSXr7wP5u-akAQ.1588814569168.1'
            page_id = '!8i%d'%page
            headers = {'User-Agent':random.choice(agents)}
            response = requests.get(url.format(d1,d2,d3)+pb.format(page_id), headers = headers)
            all_result = all_result + get_allcom(response)
            if len(get_allcom(response)) != 0:
                page+=20
            else:
                break
        # print(d1, len(all_result))
        com_num[d1_multiple] = len(all_result)

    max_num = max(list(com_num.values())[12:19])
    if max_num == 0:
        # 新的四个坐标
        new_list = get_23d(d2, d3)
        print('没有最优倍数，取默认值')
    else:
        best_d1_multiple = [i for i in list(com_num.keys())[12:19] if com_num[i] == max_num]
        print('最优倍数为：', best_d1_multiple)
        new_list = get_23d(d2, d3, dis = d1_dict[best_d1_multiple[0]])
    print('移动后的四个坐标',new_list)
    for i in new_list:
        if i:
            get_com(i[0],i[1])

if __name__ == '__main__':
    # d = 20037508.3427892
    # 起始点
    d3 = 22.3527234
    d2 = 114.1277
    get_com(d2, d3)
