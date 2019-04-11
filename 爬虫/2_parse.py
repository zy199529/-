import urllib.parse

image_url = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1554997792992&di=8961a15a26b4e2cc3da7ba40c376b483&imgtype=0&src=http%3A%2F%2Fs3.sinaimg.cn%2Fmw690%2F006hikKrzy7sly9Fzoeb2%26690"
url = 'http://www.baidu.com/index.html?name=狗蛋&pwd=123456'
ret = urllib.parse.quote(url)
re = urllib.parse.unquote(ret)
print(re)
