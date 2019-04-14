import urllib.parse
import urllib.request

url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=&'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'
}
start = 2
form_data = {
    'start': start
}

query_string = urllib.parse.urlencode(form_data)
request = url + query_string
request = urllib.request.Request(url=request,headers=headers)
response = urllib.request.urlopen(url=request)
print(response.read().decode())
