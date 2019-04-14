import urllib.parse
import urllib.request

post_url = 'https://fanyi.baidu.com/sug'
# 构建post表单数据
#word = input("请输入英文单词：")

word = 'baby'
form_data = {
    'kw': word,
}
# 发送请求过程
# 构建表单头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
}

# 构建请求头
request = urllib.request.Request(url=post_url, headers=headers)
# 发送请求
# 处理表单数据
form_data = urllib.parse.urlencode(form_data).encode()
response = urllib.request.urlopen(request, data=form_data)
print(response.read().decode())
