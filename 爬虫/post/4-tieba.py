import os
import urllib.parse
import urllib.request

#  输入吧名，输入起始页码，输入结束页码，然后在当前文件中创建一个吧名为名字的文件夹，里面是每一页的html内容，文件名吧名_page.html
ba_name = input('请输入要爬取得吧名：')
start_page = int(input("请输入起始页码："))
end_page = int(input("请输入结束的页码："))
# 创建文件夹
if not os.path.exists(ba_name):
    os.mkdir(ba_name)
#  搞个循环，一次爬取每一页数据
for page in range(start_page, end_page + 1):
    # page就是当前页，
    # 拼接urlpyt
    url = 'http://tieba.baidu.com/f?ie=utf-8&'
    data = {
        'kw': ba_name,
        'pn': (page - 1) * 50
    }
    data = urllib.parse.urlencode(data)
    url += data
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'

    }
    request = urllib.request.Request(url=url, headers=headers)
    print('第%s页......' % page)
    response = urllib.request.urlopen(request)
    filename = ba_name + "_" + str(page) + '.html'
    filepath = ba_name + '/' + filename
    # 写入
    with open(filepath, 'wb') as fp:
        fp.write(response.read())
