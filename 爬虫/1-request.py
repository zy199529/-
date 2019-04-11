import urllib.request

# url = 'http://www.baidu.com/'
# # 完整的url  ----http://www.baidu.com:80/index.html?name=goudan&password123#lala
# response = urllib.request.urlopen(url)
# print(response.readlines())
# with open('baidu.html', 'w', encoding='utf8') as fp:
#     fp.write(response.read().decode())
# 下载图片
image_url = "https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1554997792992&di=8961a15a26b4e2cc3da7ba40c376b483&imgtype=0&src=http%3A%2F%2Fs3.sinaimg.cn%2Fmw690%2F006hikKrzy7sly9Fzoeb2%26690"
# response = urllib.request.urlopen(url=image_url)
# # 图片只能写入本地二进制的格式
# with open('a.jpg', 'wb') as fp:
#     fp.write(response.read())
urllib.request.urlretrieve(url=image_url, filename='b.jpg')
