#!/usr/bin/env pipenv-shebang


from selenium import webdriver
import os
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
import random
import pandas as pd
import datetime

driver = None
cService = webdriver.ChromeService(executable_path='./chromedriver-mac-x64/chromedriver')

print("start")


def get_outfit_html_items(url):
    global driver

    # if driver is not None:
    #     driver.close()
    #     pass

    driver = webdriver.Chrome(service=cService)

    def get_body_height():
        return driver.find_element(By.CSS_SELECTOR, 'body').size["height"]

    driver.get(url)
    sleep(7)
    driver.find_element(By.CSS_SELECTOR, '#uc-btn-accept-banner').click()
    # driver.minimize_window()

    last_height = 0

    while last_height != get_body_height():
        last_height = get_body_height()
        driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight - {random.uniform(1400, 1700)});")
        sleep(random.uniform(2, 10))
        driver.execute_script(
            f"document.querySelectorAll('._2Pvyxl')[document.querySelectorAll('._2Pvyxl').length - 1].scrollIntoView();")
        sleep(random.uniform(2, 10))

    return driver.find_elements(By.CSS_SELECTOR, '._LM.JT3_zV.CKDt_l.CKDt_l.LyRfpJ')


print("start crawling")
outfits_sporty = get_outfit_html_items("https://www.zalando.ch/get-the-look-herren/?style=style_sporty")
outfits_extravagant = get_outfit_html_items("https://www.zalando.ch/get-the-look-herren/?style=style_extravagant")
# outfits_party = get_outfit_html_items("")
outfits_urban = get_outfit_html_items("https://www.zalando.ch/get-the-look-herren/?style=style_urban")
outfits_casual = get_outfit_html_items("https://www.zalando.ch/get-the-look-herren/?style=style_casual")
outfits_classic = get_outfit_html_items("https://www.zalando.ch/get-the-look-herren/?style=style_classic")

print("finished crawling")

print("mapping values")
records_sporty = list(map(lambda x: (x.get_attribute('href'), 'sporty'), outfits_sporty))
records_extravagant = list(map(lambda x: (x.get_attribute('href'), 'extravagant'), outfits_extravagant))
# records_party = list(map(lambda x: (x.get_attribute('href'), 'party'), outfits_party))
records_urban = list(map(lambda x: (x.get_attribute('href'), 'urban'), outfits_urban))
records_casual = list(map(lambda x: (x.get_attribute('href'), 'casual'), outfits_casual))
records_classic = list(map(lambda x: (x.get_attribute('href'), 'classic'), outfits_classic))
print("done")

all_records = []
all_records.extend(records_sporty)
all_records.extend(records_extravagant)
# all_records.extend(records_party)
all_records.extend(records_urban)
all_records.extend(records_casual)
all_records.extend(records_classic)

print("writing csv")
df = pd.DataFrame(all_records, columns=['outfit_detail_url', 'outfit_type'])

filename1 = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'

zalando_dir_path = '../datasets/zalando_outfit_detail_pages/men'
if not os.path.exists(zalando_dir_path):
    os.makedirs(zalando_dir_path)
df.to_csv(f'{zalando_dir_path}/{filename1}')

print("done")
