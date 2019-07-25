from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
mydriver = webdriver.Firefox()
address = input('Enter the address where you want to scrape the links form : ')
mydriver.get(address)

scroll=1

f= open ('links.txt','w')

while scroll==1:
    links =[]
    link_data = mydriver.find_elements_by_xpath('//*[@id="video-title"]')
    for tag in link_data:
        links.append(tag.get_attribute('href'))
    for i in links:
        f.write(i+'\n')
        print('Saved {}'.format(i))
    n_scroll=input('\nContinue?')
    #mydriver.execute_script("window.scrollTo(0, 1080)")    
    #mydriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    for l in link_data:
        l.send_keys(Keys.PAGE_DOWN)

    if n_scroll=='':
        scroll=1
    else:
        scroll=n_scroll

f.close()
mydriver.quit()
print('Completed!')
    