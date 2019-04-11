import urllib.parse
import urllib.request

word = input("请输入你想要搜索的内容：")
url = 'http://www.baidu.com/s?'
# 参数写成字典
data = {
    'ie': 'utf-8',
    'wd': word,
}
query_string = urllib.parse.urlencode(data)
url = url + query_string
filename = word + '.html'
response = urllib.request.urlopen(url=url)
with open(filename, 'wb') as fp:
    fp.write(response.read())
