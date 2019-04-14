import urllib.parse
url = 'http://www.baidu.com/index.html'
# 加入参数有 name age sex heigh   thttp://www.baidu.com/index.html?name='goudan'&age=18.....
name = '狗蛋'
age = 18
sex = 'nv'
height = '180'
data = {
    'name':name,
    'age':age,
    'sex':sex,
    'height':height,
    'weight':180,
}
query_string = urllib.parse.urlencode(data)
print(query_string)

#遍历字典
# lt=[]
# for k,v in data.items():
#     lt.append(k+'='+str(v))
# query_string = '&'.join(lt)
# url = url+'?'+query_string
# print(url)