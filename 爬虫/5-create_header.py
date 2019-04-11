import urllib.request
url = 'http://www.baidu.com/'
# response =urllib.request.urlopen(url)
# print(response.read().decode())
# 伪装自己的UA，让服务端认为浏览器在上网

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
}
# 构建请求对象
request = urllib.request.Request(url=url, headers=headers)
# 发送请求
response = urllib.request.urlopen(url=request)
print(response.read().decode())
