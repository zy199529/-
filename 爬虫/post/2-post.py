import urllib.parse
import urllib.request

post_url = 'https://fanyi.baidu.com/v2transapi'
word = 'wolf'
form_data = {
    'from': 'en',
    'to': 'zh',
    'query': word,
    'transtype': 'realtime',
    'simple_means_flag': '3',
    'sign': '814534.560887',
    'token': 'a22730c503177881125d11dfa816eacf',
}
headers = {
    'Host': 'fanyi.baidu.com',
    #'Connection': 'keep-alive',
    #'Content-Length': 121,
    'Accept': '*/*',
    'Origin': 'https://fanyi.baidu.com',
    'X-Requested-With': 'XMLHttpRequest',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    #'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Referer': 'https://fanyi.baidu.com/',
    #'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7,ja-JP;q=0.6,ja;q=0.5,ko-KR;q=0.4,ko;q=0.3',
    'Cookie': 'BAIDUID=38BD55C58C63FBE274F8F9E5B9FC1225:FG=1; BIDUPSID=38BD55C58C63FBE274F8F9E5B9FC1225; PSTM=1546081258; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BDUSS=NkaTdBWC1xQnpCazFxSEkxMXJ2NTRmeERkWW5XaWRMNVlSaWdSdEdiZjdvWE5jQVFBQUFBJCQAAAAAAAAAAAEAAAAklg1Uu9jt-Mj9yfrl-tfT0KYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPsUTFz7FExccz; BDRCVFR[AjWLOTlvo0C]=mk3SLVN4HKm; delPer=0; PSINO=7; H_PS_PSSID=; locale=zh; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1555230298; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1555230298; to_lang_often=%5B%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%2C%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%5D; from_lang_often=%5B%7B%22value%22%3A%22zh%22%2C%22text%22%3A%22%u4E2D%u6587%22%7D%2C%7B%22value%22%3A%22en%22%2C%22text%22%3A%22%u82F1%u8BED%22%7D%5D',
}
request = urllib.request.Request(url=post_url, headers=headers)
form_data = urllib.parse.urlencode(form_data).encode()
response = urllib.request.urlopen(url=request, data=form_data)
print(response.read().decode())
