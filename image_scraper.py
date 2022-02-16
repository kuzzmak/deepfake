from pathlib import Path
import random
import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def check_if_result_b64(source):
    possible_header = source.split(',')[0]
    if possible_header.startswith('data') and ';base64' in possible_header:
        image_type = possible_header.replace(
            'data:image/',
            '').replace(
            ';base64',
            '')
        return image_type
    return False


supported_mimes = set(['image/jpeg', 'image/png'])
mime_to_ext = {
    'image/png': 'png',
    'image/jpeg': 'jpg',
}
images_dir = Path('images')

if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/80.0.3987.132 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--allow-cross-origin-auth-prompt")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(
        service=service,
        options=options,
    )
    driver.get('https://www.google.com/search?q=nicolas+cage&tbm=isch')

    results = driver.find_elements(
        By.CSS_SELECTOR,
        '.isv-r.PNCib.MSM1fd.BUooTd',
    )
    last_results_num = len(results)

    while True:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        # time to load images when scrolling
        time.sleep(random.randint(2, 5))
        results = driver.find_elements(
            By.CSS_SELECTOR,
            '.isv-r.PNCib.MSM1fd.BUooTd',
        )
        if len(results) == last_results_num:
            break
        last_results_num = len(results)

    found = 0
    for res in results:
        res.click()
        time.sleep(random.randint(2, 5))

        elem = driver.find_element(
            By.CSS_SELECTOR,
            '.tvh9oe.BIB1wf .eHAdSb>img',
        )
        img_src = elem.get_attribute('src')

        is_b64 = check_if_result_b64(img_src)
        if is_b64:
            print('je kod')
        else:
            try:
                resp = requests.get(img_src)
            except requests.exceptions.RequestException:
                continue
            contents = resp.content
            mime_type = resp.headers['content-type']
            if mime_type not in supported_mimes:
                continue
            else:
                ext = mime_to_ext[mime_type]
                filename = f'{str(found)}.{ext}'
                with open(images_dir / filename, 'wb') as f:
                    f.write(contents)
                found += 1
                print('downloaded: ', img_src)

    driver.quit()

    # results = driver.find_elements(
    #     By.CSS_SELECTOR,
    #     '.isv-r.PNCib.MSM1fd.BUooTd',
    # )
    # found = 0
    # # images that appear when searching on google images
    # results = driver.find_elements(
    #     By.CSS_SELECTOR,
    #     '.isv-r.PNCib.MSM1fd.BUooTd',
    # )
    # for res in results:
    #     res.click()
    #     time.sleep(random.randint(5, 15) / 10)
    #     # after click on some image, panel on right appears where this exact
    #     # image is but in bigger resolution and button for suggested images
    #     # which is then clicked
    #     res = driver.find_element(
    #         By.CSS_SELECTOR,
    #         'div.O8VmIc.a3Wc3 > a'
    #     )
    #     res.click()
    #     time.sleep(random.randint(5, 15) / 10)
    #     # new page with suggested images
    #     suggested_results = driver.find_elements(
    #         By.CSS_SELECTOR,
    #         '.isv-r.PNCib.MSM1fd.BUooTd',
    #     )
    #     for sugg_res in suggested_results:
    #         sugg_res.click()
    #         time.sleep(random.randint(5, 15) / 10)
    #         # image element for a single image on this page
    #         elem = driver.find_element(
    #             By.CSS_SELECTOR,
    #             '.tvh9oe.BIB1wf .eHAdSb>img',
    #         )
    #         img_src = elem.get_attribute('src')
    #         resp = requests.get(img_src)
    #         contents = resp.content
    #         mime_type = resp.headers['content-type']
    #         if mime_type not in supported_mimes:
    #             continue
    #         else:
    #             ext = mime_to_ext[mime_type]
    #             filename = f'{str(found)}.{ext}'
    #             with open(images_dir / filename, 'wb') as f:
    #                 f.write(contents)
    #             found += 1
    #             print('downloaded: ', img_src)
    #     break
    # driver.quit()
