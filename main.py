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
# 谷歌地图英文国家
google_country_dict = {'afghanistan': 'Afghanistan', 'åland islands': 'Åland Islands', 'albania': 'Albania', 'algeria': 'Algeria', 'american samoa': 'American Samoa', 'andorra': 'Andorra', 'angola': 'Angola', 'anguilla': 'Anguilla', 'antarctica': 'Antarctica', 'antigua and barbuda': 'Antigua and Barbuda', 'argentina': 'Argentina', 'armenia': 'Armenia', 'aruba': 'Aruba', 'australia': 'Australia', 'austria': 'Austria', 'azerbaijan': 'Azerbaijan', 'bahamas': 'The Bahamas', 'bahrain': 'Bahrain', 'bangladesh': 'Bangladesh', 'barbados': 'Barbados', 'belarus': 'Belarus', 'belgium': 'Belgium', 'belize': 'Belize', 'benin': 'Benin', 'bermuda': 'Bermuda', 'bhutan': 'Bhutan', 'bolivia': 'Bolivia', 'bonaire, sint eustatius and saba (netherlands)': 'Sint Eustatius', 'bosnia and herzegovina': 'Bosnia and Herzegovina', 'botswana': 'Botswana', 'bouvet island': 'Bouvet Island', 'brazil': 'Brazil', 'british indian ocean territory': 'British Indian Ocean Territory', 'brunei darussalam': 'Brunei', 'brunei': 'Brunei', 'bulgaria': 'Bulgaria', 'burkina faso': 'Burkina Faso', 'burundi': 'Burundi', 'cape verde': 'Cape Verde', 'cambodia': 'Cambodia', 'cameroon': 'Cameroon', 'canada': 'Canada', 'cayman islands': 'Cayman Islands', 'central african': 'Central African Republic', 'central african republic': 'Central African Republic', 'chad': 'Chad', 'chile': 'Chile', 'china': 'China', 'christmas island': 'Christmas Island', 'colombia': 'Colombia', 'comoros': 'Comoros', 'democratic republic of congo': 'Democratic Republic of the Congo', 'congo dr': 'Democratic Republic of the Congo', 'congo': 'Democratic Republic of the Congo', 'cook islands': 'Cook Islands', 'costa rica': 'Costa Rica', 'croatia': 'Croatia', 'cuba': 'Cuba', 'curaçao': 'Curaçao', 'cyprus': 'Cyprus', 'czech republic': 'Czechia', 'denmark': 'Denmark', 'djibouti': 'Djibouti', 'dominica': 'Dominica', 'dominican rep': 'Dominican Republic', 'dominican republic': 'Dominican Republic', 'ecuador': 'Ecuador', 'egypt': 'Egypt', 'el salvador': 'El Salvador', 'equatorial guinea': 'Equatorial Guinea', 'eritrea': 'Eritrea', 'estonia': 'Estonia', 'ethiopia': 'Ethiopia', 'falkland islands  [malvinas]': 'Falkland Islands (Islas Malvinas)', 'faroe islands (denmark)': 'Faroe Islands', 'fiji': 'Fiji', 'finland': 'Finland', 'france': 'France', 'french guiana (france)': 'French Guiana', 'french polynesia (france)': 'French Polynesia', 'french southern territories': 'French Southern and Antarctic Lands', 'gabon': 'Gabon', 'gambia': 'The Gambia', 'georgia': 'Georgia', 'germany': 'Germany', 'ghana': 'Ghana', 'gibraltar': 'Gibraltar', 'greece': 'Greece', 'greenland (denmark)': 'Greenland', 'grenada': 'Grenada', 'guadeloupe (france)': 'Guadeloupe', 'guam (united states of america)': 'Guam', 'guatemala': 'Guatemala', 'guernsey (united kingdom)': 'Guernsey', 'guinea': 'Guinea', 'guinea bissau': 'Guinea-Bissau', 'guyana': 'Guyana', 'haiti': 'Haiti', 'heard island and mcdonald islands': 'Heard Island and McDonald Islands', 'holy see': 'Vatican City', 'honduras': 'Honduras', 'hong kong': 'Hong Kong', 'hungary': 'Hungary', 'iceland': 'Iceland', 'india': 'India', 'indonesia': 'Indonesia', 'islamic republic of iran': 'Iran', 'iraq': 'Iraq', 'ireland': 'Ireland', 'isle of man (united kingdom)': 'Isle of Man', 'israel': 'Israel', 'italy': 'Italy', 'jamaica': 'Jamaica', 'japan': 'Japan', 'jersey (united kingdom)': 'Jersey', 'jordan': 'Jordan', 'kazakhstan': 'Kazakhstan', 'kenya': 'Kenya', 'kiribati': 'Kiribati', 'democratic peoples republic of korea': 'North Korea', 'republic of korea': 'South Korea', 'kuwait': 'Kuwait', 'kyrgyzstan': 'Kyrgyzstan', 'latvia': 'Latvia', 'lebanon': 'Lebanon', 'lesotho': 'Lesotho', 'liberia': 'Liberia', 'libya': 'Libya', 'liechtenstein': 'Liechtenstein', 'lithuania': 'Lithuania', 'luxembourg': 'Luxembourg', 'macao': 'Macao', 'macedonia': 'North Macedonia', 'macedonia (fyrom)': 'North Macedonia', 'madagascar': 'Madagascar', 'malawi': 'Malawi', 'malaysia': 'Malaysia', 'maldives': 'Maldives', 'mali': 'Mali', 'malta': 'Malta', 'marshall islands': 'Marshall Islands', 'martinique': 'Martinique', 'martinique (france)': 'Martinique', 'mauritania': 'Mauritania', 'mauritius': 'Mauritius', 'mayotte (france)': 'Mayotte', 'mexico': 'Mexico', 'republic of moldova': 'Moldova', 'monaco': 'Monaco', 'mongolia': 'Mongolia', 'montenegro': 'Montenegro', 'montserrat': 'Montserrat', 'morocco': 'Morocco', 'mozambique': 'Mozambique', 'myanmar/burma': 'Myanmar (Burma)', 'namibia': 'Namibia', 'nauru': 'Nauru', 'nepal': 'Nepal', 'netherlands': 'Netherlands', 'new caledonia (france)': 'New Caledonia', 'new zealand': 'New Zealand', 'nicaragua': 'Nicaragua', 'niger': 'Niger', 'nigeria': 'Nigeria', 'niue': 'Niue', 'norfolk island': 'Norfolk Island', 'northern mariana islands': 'Northern Mariana Islands', 'norway': 'Norway', 'oman': 'Oman', 'pakistan': 'Pakistan', 'palau': 'Palau', 'palestinian territories': 'Palestine', 'panama': 'Panama', 'papua new guinea': 'Papua New Guinea', 'paraguay': 'Paraguay', 'peru': 'Peru', 'philippines': 'Philippines', 'pitcairn': 'Pitcairn Islands', 'poland': 'Poland', 'portugal': 'Portugal', 'puerto rico': 'Puerto Rico', 'puerto rico (united states of america)': 'Puerto Rico', 'reunion (france)': 'Réunion', 'romania': 'Romania', 'russian federation': 'Russia', 'rwanda': 'Rwanda', 'saint helena, ascension and tristan da cunha': 'St Helena, Ascension and Tristan da Cunha', 'saint kitts and nevis': 'St Kitts & Nevis', 'saint lucia': 'St Lucia', 'saint martin (france)': 'St Martin', 'saint pierre and miquelon': 'St Pierre and Miquelon', 'saint vincent and the grenadines': 'St Vincent and the Grenadines', 'samoa': 'Samoa', 'san marino': 'San Marino', 'sao tome and principe': 'São Tomé and Príncipe', 'saudi arabia': 'Saudi Arabia', 'senegal': 'Senegal', 'serbia': 'Serbia', 'seychelles': 'Seychelles', 'sierra leone': 'Sierra Leone', 'singapore': 'Singapore', 'sint maarten': 'Sint Maarten', 'slovakia': 'Slovakia', 'slovenia': 'Slovenia', 'solomon islands': 'Solomon Islands', 'somalia': 'Somalia', 'south africa': 'South Africa', 'south georgia and the south sandwich islands': 'South Georgia and the South Sandwich Islands', 'south sudan': 'South Sudan', 'spain': 'Spain', 'sri lanka': 'Sri Lanka', 'sudan': 'Sudan', 'suriname': 'Suriname', 'svalbard and jan mayen': 'Svalbard and Jan Mayen', 'swaziland': 'Eswatini', 'sweden': 'Sweden', 'switzerland': 'Switzerland', 'syrian arab republic': 'Syria', 'taiwan': 'Taiwan', 'tajikistan': 'Tajikistan', 'united republic of tanzania': 'Tanzania', 'thailand': 'Thailand', 'timor-leste': 'Timor-Leste', 'togo': 'Togo', 'tokelau': 'Tokelau', 'tonga': 'Tonga', 'trinidad and tobago': 'Trinidad and Tobago', 'tunisia': 'Tunisia', 'turkey': 'Turkey', 'turkmenistan': 'Turkmenistan', 'turks and caicos islands': 'Turks and Caicos Islands', 'tuvalu': 'Tuvalu', 'uganda': 'Uganda', 'ukraine': 'Ukraine', 'united arab emirates': 'United Arab Emirates', 'united kingdom': 'United Kingdom', 'united states minor outlying islands': 'United States Minor Outlying Islands', 'united states of america': 'United States', 'uruguay': 'Uruguay', 'uzbekistan': 'Uzbekistan', 'vanuatu': 'Vanuatu', 'venezuela': 'Venezuela', 'vietnam': 'Vietnam', 'virgin islands (british)': 'British Virgin Islands', 'virgin islands (united states of america)': 'U.S. Virgin Islands', 'wallis and futuna': 'Wallis and Futuna', 'western sahara*': 'Western Sahara', 'yemen': 'Yemen', 'zambia': 'Zambia', 'zimbabwe': 'Zimbabwe', 'lao peoples democratic republic': 'Laos', 'cote divoire': "Côte d'Ivoire", 'kosovo': 'Kosovo'}
# 国家缩写后缀
country_suffix_dict = {'af': 'Afghanistan', 'fk': 'land Islands', 'al': 'Albania', 'dz': 'Algeria', 'as': 'American Samoa', 'ad': 'Andorra', 'ao': 'Angola', 'ai': 'Anguilla', 'aq': 'Antarctica', 'ag': 'Antigua and Barbuda', 'ar': 'Argentina', 'am': 'Armenia', 'aw': 'Aruba', 'au': 'Australia', 'at': 'Austria', 'az': 'Azerbaijan', 'bs': 'Bahamas', 'bh': 'Bahrain', 'bd': 'Bangladesh', 'bb': 'Barbados', 'by': 'Belarus', 'be': 'Belgium', 'bz': 'Belize', 'bj': 'Benin', 'bm': 'Bermuda', 'bt': 'Bhutan', 'ga': 'Bolivia', 'ba': 'Bosnia and Herzegovina', 'bw': 'Botswana', 'bv': 'Bouvet Island', 'br': 'Brazil', 'io': 'British Indian Ocean Territory', 'bn': 'Brunei Darussalam', 'bg': 'Bulgaria', 'bf': 'Burkina Faso', 'bi': 'Burundi', 'cv': 'Cape Verde', 'kh': 'Cambodia', 'cm': 'Cameroon', 'ca': 'Canada', 'ky': 'Cayman Islands', 'cf': 'Central African Republic', 'td': 'Chad', 'cl': 'Chile', 'cn': 'China', 'cx': 'Christmas Island', 'cc': 'Cocos', 'co': 'Colombia', 'km': 'Comoros', 'cg': 'Congo', 'ck': 'Cook Islands', 'cr': 'Costa Rica', 'hr': 'Croatia', 'cu': 'Cuba', 'cy': 'Cyprus', 'cz': 'Czech Republic', 'dk': 'Denmark', 'dj': 'Djibouti', 'do': 'Dominica', 'ec': 'Ecuador', 'eg': 'Egypt', 'sv': 'El Salvador', 'cq': 'Equatorial Guinea', 'ee': 'Estonia', 'et': 'Ethiopia', 'fo': 'Faroe Islands (Denmark)', 'fj': 'Fiji', 'fi': 'Finland', 'fr': 'France', 'gf': 'French Guiana (France)', 'pf': 'French Polynesia (France)', 'tf': 'French Southern Territories', 'gm': 'Gambia', 'ge': 'Georgia', 'de': 'Germany', 'gh': 'Ghana', 'gi': 'Gibraltar', 'vi': 'Greece', 'gl': 'Grenada', 'gp': 'Guadeloupe (France)', 'gt': 'Guatemala', 'gn': 'Guinea', 'gw': 'Guinea Bissau', 'gy': 'Guyana', 'ht': 'Haiti', 'hm': 'Heard Island and McDonald Islands', 'va': 'Holy See', 'hn': 'Honduras', 'hk': 'Hong Kong', 'hu': 'Hungary', 'is': 'Iceland', 'in': 'India', 'id': 'Indonesia', 'ir': 'Islamic Republic of Iran', 'iq': 'Iraq', 'ie': 'Ireland', 'il': 'Israel', 'it': 'Italy', 'jm': 'Jamaica', 'jp': 'Japan', 'jo': 'Jordan', 'kz': 'Kazakhstan', 'ke': 'Kenya', 'ki': 'Kiribati', 'kp': 'Democratic Peoples Republic of Korea', 'kr': 'Republic of Korea', 'kw': 'Kuwait', 'kg': 'Kyrgyzstan', 'lv': 'Latvia', 'lb': 'Lebanon', 'ls': 'Lesotho', 'lr': 'Liberia', 'ly': 'Libya', 'li': 'Liechtenstein', 'lt': 'Lithuania', 'lu': 'Luxembourg', 'mo': 'Macao', 'mg': 'Madagascar', 'mw': 'Malawi', 'my': 'Malaysia', 'mv': 'Maldives', 'ml': 'Mali', 'mt': 'Malta', 'mh': 'Marshall Islands', 'mq': 'Martinique (France)', 'mr': 'Mauritania', 'mx': 'Mexico', 'fm': 'Micronesia', 'md': 'Republic of Moldova', 'mc': 'Monaco', 'mn': 'Mongolia', 'ms': 'Montserrat', 'ma': 'Morocco', 'mz': 'Mozambique', 'mm': 'Myanmar/Burma', 'na': 'Namibia', 'nr': 'Nauru', 'np': 'Nepal', 'nl': 'Netherlands', 'nc': 'New Caledonia (France)', 'nz': 'New Zealand', 'ni': 'Nicaragua', 'ne': 'Niger', 'ng': 'Nigeria', 'nu': 'Niue', 'nf': 'Norfolk Island', 'mp': 'Northern Mariana Islands', 'no': 'Norway', 'om': 'Oman', 'pk': 'Pakistan', 'pw': 'Palau', 'pa': 'Panama', 'pg': 'Papua New Guinea', 'py': 'Paraguay', 'pe': 'Peru', 'ph': 'Philippines', 'pn': 'Pitcairn', 'pl': 'Poland', 'pt': 'Portugal', 'qa': 'Qatar', 're': 'Reunion (France)', 'ro': 'Romania', 'ru': 'Russian Federation', 'rw': 'Rwanda', 'kn': 'Saint Kitts and Nevis', 'sh': 'Saint Lucia', 'pm': 'Saint Pierre and Miquelon', 'vc': 'Saint Vincent and the Grenadines', 'sm': 'San Marino', 'st': 'Sao Tome and Principe', 'sa': 'Saudi Arabia', 'sn': 'Senegal', 'sc': 'Seychelles', 'sl': 'Sierra Leone', 'sg': 'Singapore', 'sk': 'Slovakia', 'si': 'Slovenia', 'sb': 'Solomon Islands', 'so': 'Somalia', 'za': 'South Africa', 'ss': 'South Sudan', 'es': 'Spain', 'lk': 'Sri Lanka', 'sd': 'Sudan', 'sr': 'Suriname', 'sz': 'Swaziland', 'se': 'Sweden', 'ch': 'Switzerland', 'sy': 'Syrian Arab Republic', 'tw': 'Taiwan', 'tj': 'Tajikistan', 'tz': 'United Republic of Tanzania', 'th': 'Thailand', 'tl': 'Timor-Leste', 'tg': 'Togo', 'tk': 'Tokelau', 'to': 'Tonga', 'tt': 'Trinidad and Tobago', 'tn': 'Tunisia', 'tr': 'Turkey', 'tm': 'Turkmenistan', 'tc': 'Turks and Caicos Islands', 'tv': 'Tuvalu', 'ug': 'Uganda', 'ua': 'Ukraine', 'ae': 'United Arab Emirates', 'uk': 'United Kingdom', 'pr': 'United States Minor Outlying Islands', 'us': 'United States of America', 'uy': 'Uruguay', 'vu': 'Vanuatu', 've': 'Venezuela', 'vn': 'VietNam', 'vg': 'Virgin Islands (British)', 'wf': 'Wallis and Futuna', 'ws': 'Western Sahara*', 'ye': 'Yemen', 'zm': 'Zambia', 'zw': 'Zimbabwe', 'la': 'Lao Peoples Democratic Republic', 'ci': 'COTE DIVOIRE'}

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
                if len(company) > 14:
                    temp_dict['companyName'] = company[14][11] if company and company[14] and company[14][11] else None
                    temp_dict['url'] = company[14][7][0] if company and company[14] and company[14][7] and company[14][7][
                        0] else None
                    if temp_dict['url']:
                        temp_dict['url'] = unquote(temp_dict['url'])
                    temp_dict['address'] = company[14][39] if company and company[14] and company[14][39] else None
                    if not temp_dict['address']:
                        temp_dict['address'] = company[14][18] if company and company[14] and company[14][18] else None
                    temp_dict['phone'] = company[14][178][0][0] if company and company[14] and company[14][178] and company[14][178][0] and company[14][178][0][0] else None
                    if not temp_dict['phone']:
                        temp_dict['phone'] = company[14][3][0] if company and company[14] and company[14][3] and company[14][3][0] else None
                    temp_dict['category'] = '>'.join(company[14][13]) if company and company[14] and company[14][13] else None
                    temp_dict['countryEn'] = None
                    if temp_dict['address']:
                        for google_country in google_country_dict.values():
                            if google_country.lower() in temp_dict['address'].lower():
                                temp_dict['countryEn'] = google_country
                                break
                    if not temp_dict['countryEn']:
                        if company[14] and company[14][183] and company[14][183][1] and company[14][183][-1].lower() in country_suffix_dict.keys():
                            temp_dict['countryEn'] = country_suffix_dict(company[14][183][-1].lower())
                    temp_dict['city'] = company[14][14] if company and company[14] and company[14][14] else None
                    if temp_dict['companyName']:
                        yield temp_dict
                        result.append(temp_dict)
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
        print('目前倍数%d：'%d1_multiple)
        url = 'https://www.google.com/search?tbm=map&authuser=0&hl=en&pb=!4m12!1m3!1d{}!2d{}!3d{}'
        page = 0
        all_result = []
        while True:
            pb = '!2m3!1f0!2f0!3f0!3m2!1i784!2i644!4f13.1!7i20{}!10b1!12m8!1m1!18b1!2m3!5m1!6e2!20e3!10b1!16b1!19m4!2m3!1i360!2i120!4i8!20m57!2m2!1i203!2i100!3m2!2i4!5b1!6m6!1m2!1i86!2i86!1m2!1i408!2i240!7m42!1m3!1e1!2b0!3e3!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e9!2b1!3e2!1m3!1e10!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e4!2b1!4b1!9b0!22m5!1s5mKzXrHAJNSXr7wP5u-akAQ!4m1!2i5600!7e81!12e30!24m46!1m12!13m6!2b1!3b1!4b1!6i1!8b1!9b1!18m4!3b1!4b1!5b1!6b1!2b1!5m5!2b1!3b1!5b1!6b1!7b1!10m1!8e3!14m1!3b1!17b1!20m2!1e3!1e6!24b1!25b1!26b1!30m1!2b1!36b1!43b1!52b1!55b1!56m2!1b1!3b1!65m5!3m4!1m3!1m2!1i224!2i298!26m4!2m3!1i80!2i92!4i8!30m28!1m6!1m2!1i0!2i0!2m2!1i458!2i644!1m6!1m2!1i734!2i0!2m2!1i784!2i644!1m6!1m2!1i0!2i0!2m2!1i784!2i20!1m6!1m2!1i0!2i624!2m2!1i784!2i644!31b1!34m13!2b1!3b1!4b1!6b1!8m3!1b1!3b1!4b1!9b1!12b1!14b1!20b1!23b1!37m1!1e81!42b1!46m1!1e2!47m0!49m1!3b1!50m13!1m8!3m6!1u17!2m4!1m2!17m1!1e2!2z6Led56a7!4BIAE!2e2!3m2!1b1!3b0!59BQ2dBd0Fn!65m0&q=company&tch=1&ech=4&psi=5mKzXrHAJNSXr7wP5u-akAQ.1588814569168.1'
            page_id = '!8i%d'%page
            headers = {'User-Agent':random.choice(agents)}
            response = requests.get(url.format(d1,d2,d3)+pb.format(page_id), headers = headers)
            result = get_allcom(response)
            all_result += result
            if len(result) != 0:
                page+=20
            else:
                break
        print(d1, len(all_result))
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
            # print(i)

if __name__ == '__main__':
    # d = 20037508.3427892
    # 起始点
    d3 = 22.3527234
    d2 = 114.1277
    get_com(d2, d3)
