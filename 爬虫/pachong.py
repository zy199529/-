#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
res = requests.get('http://bj.xiaozhu.com/')#网站为小猪短租网北京地区网址


# In[5]:


print(res)#print(res.text)


# In[37]:


import requests
from bs4 import BeautifulSoup
def judgment_sex(class_name):#定义判断用户性别的函数
    if class_name == ['member_ico1']:
        return '女'
    else:
        return '男'
def get_url(url):
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    res = requests.get(url,headers=headers)#定义请求头的requests
    soup = BeautifulSoup(res.text,'html.parser')
    links = soup.select('#page_list > ul > li > a')
    for link in links:
        href = link.get("href")#爬取详情页的url
        get_info(href)
def get_info(url):
    headers ={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    wb_data = requests.get(url,headers=headers)
    soup = BeautifulSoup(wb_data.text,'html.parser')
    titles = soup.select('div.pho_info > h4')
    addresses = soup.select('span.pr5')
    prices = soup.select('#pricePart > div.day_l > span')
    imgs = soup.select('#floatRightBox > div.js_box.clearfix > div.member_pic')
    names =soup.select('#floatRightBox > div.js_box.clearfix > div.w_240 > h6 > a')
    sexs = soup.select('#floatRightBox > div.js_box.clearfix > div.member_pic > div')
    for title,address,price,img,name,sex in zip(titles,addresses,prices,imgs,names,sexs):
        data = {
            'title':title.get_text().strip(),
            'address':address.get_text().strip(),
            'price':price.get_text(),
            'img':img.get("src"),
            'name':name.get_text(),
            'sex':judgment_sex(sex.get("class"))
        }
        print(data)
        
urls = ['http://bj.xiaozhu.com/search-duanzufang-p{}-0/'.format(number) for number in range(1,14)]#构造多页的URL
for single_url in urls:
    print(single_url)
    get_url(single_url)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




