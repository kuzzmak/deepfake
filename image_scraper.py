import base64
import logging
import os
from pathlib import Path
import random
import time
from typing import List, Optional, Union

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webelement import WebElement
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


class Scraper:

    def __init__(self) -> None:
        options = webdriver.ChromeOptions()
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) ' \
            'AppleWebKit/537.36 (KHTML, like Gecko) ' \
            'Chrome/80.0.3987.132 Safari/537.36'
        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--allow-cross-origin-auth-prompt')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--incognito')
        options.add_argument('--headless')
        options.add_argument('--log-level=3')
        service = Service(
            ChromeDriverManager(
                log_level=logging.ERROR,
                print_first_line=False,
            ).install()
        )
        self._driver = webdriver.Chrome(
            service=service,
            options=options,
        )
        self._supported_mimes = set(['image/jpeg', 'image/png'])
        self._mime_to_ext = {
            'image/png': 'png',
            'image/jpeg': 'jpg',
        }

    @staticmethod
    def _check_if_result_b64(source: str) -> Union[bool, str]:
        possible_header = source.split(',')[0]
        if possible_header.startswith('data') and ';base64' in possible_header:
            image_type = possible_header \
                .replace('data:image/', '') \
                .replace(';base64', '')
            return image_type
        return False

    def close_driver(self) -> None:
        self._driver.quit()

    def _search_for(self, keywords: List[str]) -> None:
        keywords = '+'.join(keywords)
        link = f'https://www.google.com/search?q={keywords}&tbm=isch'
        self._driver.get(link)

    def _find_first_images(self) -> List[WebElement]:
        return self._driver.find_elements(
            By.CSS_SELECTOR,
            '.isv-r.PNCib.MSM1fd.BUooTd',
        )

    def _find_second_image(self) -> WebElement:
        return self._driver.find_element(
            By.CSS_SELECTOR,
            '.tvh9oe.BIB1wf .eHAdSb>img',
        )

    def _exists_more(self) -> bool:
        button = self._driver.find_element(
            By.CSS_SELECTOR,
            'div.YstHxe > input',
        )
        button_style = button.find_element(
            By.XPATH,
            '..'
        ).get_attribute('style')
        if button_style == '':
            button.click()
            return True
        return False

    def _scroll_to_page_end(self) -> None:
        self._driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

    def _sleep(self) -> None:
        time.sleep(random.randint(2, 5))

    def _get_suggested_button(self) -> WebElement:
        return self._driver.find_element(
            By.CSS_SELECTOR,
            'div.O8VmIc.a3Wc3 > a'
        )

    def _find_all_images(self, first_page_only=False) -> List[WebElement]:
        results = self._find_first_images()
        if first_page_only:
            return results
        last_results_num = len(results)
        while True:
            print(f'Found {last_results_num} images.')
            self._scroll_to_page_end()
            self._sleep()
            results = self._find_first_images()
            if len(results) == last_results_num:
                if self._exists_more():
                    self._sleep()
                else:
                    break
            last_results_num = len(results)
        return results

    def _download_images(
        self,
        images: List[WebElement],
        save_directory: Path,
    ) -> None:
        found = len(os.listdir(save_directory))
        for image in images:
            image.click()
            self._sleep()
            try:
                elem = self._find_second_image()
            except NoSuchElementException:
                print('Error happened while getting second image, skipping.')
                continue
            img_src = elem.get_attribute('src')
            print('Getting: ', img_src)
            is_b64 = Scraper._check_if_result_b64(img_src)
            if is_b64:
                image_format = is_b64
                contents = base64.b64decode(img_src.split(';base64')[1])
            else:
                try:
                    resp = requests.get(img_src, timeout=10)
                except requests.exceptions.RequestException:
                    continue
                contents = resp.content
                mime_type = resp.headers.get('content-type', '')
                if mime_type not in self._supported_mimes:
                    continue
                else:
                    image_format = self._mime_to_ext[mime_type]

            found += 1
            filename = f'{str(found)}.{image_format}'
            with open(save_directory / filename, 'wb') as f:
                f.write(contents)
            print('downloaded: ', img_src)

    def run(
        self,
        keywords: List[str],
        save_directory: Union[str, Path],
        n_suggested: int,
        from_n: Optional[int] = None,
        to_n: Optional[int] = None,
        skip_main_images: bool = False,
    ) -> None:
        if isinstance(save_directory, str):
            save_directory = Path(save_directory)
        if not save_directory.exists():
            os.makedirs(save_directory)

        self._search_for(keywords)
        print('Finding main page images.')
        images = self._find_all_images(skip_main_images)
        print(f'Found {len(images)} images.')

        print('Downloading main page images.')
        self._download_images(images, save_directory)

        # remember links of first n suggested images pages
        first_n_links = []
        for i in range(n_suggested):
            images[i].click()
            self._sleep()

            sugg_button = self._get_suggested_button()
            link = sugg_button.get_attribute('href')
            first_n_links.append(link)

        for i, link in enumerate(first_n_links):
            self._driver.get(link)
            self._sleep()

            print(f'Finding suggested images for image {i+1}.')
            sugg_images = self._find_all_images()
            print(
                f'Found {len(sugg_images)} suggested images ' +
                f'for main image {i+1}.'
            )
            print('Downloading images.')
            self._download_images(sugg_images, save_directory)
            print('Images downloaded.')


if __name__ == '__main__':
    sc = Scraper()
    sc.run(['donald', 'trump'], 'data/scraped/donald_trump', 10)
    sc.close_driver()
