import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import TextIOBase
from logging import Logger
from typing import Dict, List, Tuple, Union
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.models import Response

from modules.checker import url_canon
from modules.helper import get_requests_header

from modules.classifer_model import classify_text

from stem import Signal
from stem.control import Controller


class Crawler:
    """Crawl input link upto depth (depth) with a pause of pause seconds using multiple threads.

    Attributes:
        website: Website to crawl.
        proxies: Dictionary mapping protocol or protocol and host to the URL of the proxy.
        depth: Depth of the crawl.
        pause: Pause after every depth iteration.
        out_path: Output path to store extracted links.
        external: True if external links are to be crawled else False.
        exclusion: Paths that you don't want to include.
        thread: Number pages to visit (Threads) at the same time.
        logger: A logger object to log the output.
    """

    network_file = "network_structure.json"
    __headers = get_requests_header()

    def __init__(
        self,
        website: str,
        proxies: Dict[str, str],
        depth: int,
        pause: float,
        out_path: str,
        external: bool,
        exclusion: str,
        thread: int,
        logger: Logger,
    ):
        self.website = website
        self.proxies = proxies
        self.depth = depth
        self.pause = pause
        self.out_path = out_path
        self.external = external
        self.exclusion = rf"{exclusion}" if exclusion else None
        self.thread = thread
        self.logger = logger
        self.classifier = classify_text

        self.__executor = ThreadPoolExecutor(max_workers=min(32, self.thread))
        self.__files = {
            "extlinks": open(os.path.join(self.out_path, "extlinks.txt"), "w+", encoding="UTF-8"),
            #"telephones": open(os.path.join(self.out_path, "telephones.txt"), "w+", encoding="UTF-8"),
            #"mails": open(os.path.join(self.out_path, "mails.txt"), "w+", encoding="UTF-8"),
            "network_structure": os.path.join(self.out_path, self.network_file),
            "links": os.path.join(self.out_path, "links.txt"),
            "network_structure_classification": os.path.join(self.out_path, "network_structure_classification.json"),
        }

    def __get_tor_session(self) -> requests.Session:
        """Get a new session with Tor proxies.

        Returns:
            Session object to make requests.
        """
        session = requests.Session()
        session.proxies = self.proxies
        session.headers.update(self.__headers)
        session.verify = False
        return session
    
    def renew_tor_ip(self,stop) -> None: 
        """Renew tor ip every 1 minutes.
        """
        self.logger.info("Starting Tor IP Rotation Thread")
        while True:
            time.sleep(60)
            with Controller.from_port(port=9151) as controller:
                self.logger.info("Renewing Tor IP")
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
            if stop():
                self.logger.info("Stoping Tor IP Rotation Thread")
                break


    def excludes(self, link: str) -> bool:
        """Excludes links that are not required.

        Args:
            link: Link to check for exclusion.

        Returns:
            True if link is to be excluded else False.
        """
        if link is None:
            return True
        # Excludes links that matches the regex path.
        if self.exclusion and re.search(self.exclusion, link, re.IGNORECASE):
            return True
        # Links
        if "#" in link:
            return True
        # External links
        if link.startswith("http") and not link.startswith(self.website):
            if self.external:
                return False
            self.__files["extlinks"].write(str(link) + "\n")
            return True
        # Telephone Number
        if link.startswith("tel:"):
            self.__files["telephones"].write(str(link) + "\n")
            return True
        # Mails
        if link.startswith("mailto:"):
            self.__files["mails"].write(str(link) + "\n")
            return True
        # Type of files
        if re.search("^.*\\.(pdf|jpg|jpeg|png|gif|doc|js|css)$", link, re.IGNORECASE):
            return True

    def canonical(self, base: str, href: str) -> str:
        """Canonization of the link.

        Args:
            base: Base URL.
            href: Hyperlink present in the base URL page.

        Returns:
            parsed canonical url.
        """
        # Already formatted
        if href.startswith("http"):
            return href

        # For relative paths
        return urljoin(base, href)

    def __crawl_link(
        self, url: str, session: requests.Session
    ) -> Tuple[str, set[str], Union[int, Tuple[str, Exception]]]:
        """
        Extracts all the hyperlinks from the given url and returns a tuple of
        the url, set of hyperlinks and either status code or raised Exception.

        Args:
            url: URL to crawl.
            session: Session object to make requests.

        Returns:
            A tuple of the url, set of hyperlinks and either status code or raised Exception.

            (`https://example.com`, {`https://example.com/1`, `https://example.com/2`}, `200`)
            (`https://error.com`, {}, `Exception()`)
        """
        url_data = set()
        html_page = Response
        response_code = 0
        classify_verified = False
        output = None

        try:
            if url is not None:
                html_page = session.get(url, allow_redirects=True, timeout=10)
                response_code = html_page.status_code
        except Exception as err:
            return url, url_data, ("Request", err), classify_verified, output

        try:
            soup = BeautifulSoup(html_page.text, features="html.parser")
        except Exception as err:
            return url, url_data, ("Soup Parse", err), classify_verified, output
        
        #classify the text
        try:
            output,score,classify_verified = self.classifier(soup.text)
            if classify_verified:
                self.logger.info(f"Verified as {output} with score {round(score,4)}")
        except Exception as err:
            self.logger.error(f"Classifier Error: {err}")
            pass


        # For each <a href=""> tag.
        for link in soup.findAll("a"):
            link = link.get("href")

            if self.excludes(link):
                continue

            ver_link = self.canonical(url, link)
            if ver_link is not None:
                url_data.add(url_canon(ver_link)[1])

        # For each <area href=""> tag.
        for link in soup.findAll("area"):
            link = link.get("href")

            if self.excludes(link):
                continue

            ver_link = self.canonical(url, link)
            if ver_link is not None:
                url_data.add(url_canon(ver_link)[1])

        return url, url_data, response_code, classify_verified, output

    def crawl(self) -> Dict[str, List[str]]:
        """Core of the crawler.

        Returns:
            Dictionary of crawled links.

            {
                "link1": [ "link2", "link3", "link4" ],
                "link2": [ "link5", "link6", "link4" ],
                "link3": [ "link7", "link2", "link9" ],
                "link4": [ "link1" ]
            }
        """
        ord_lst = set([self.website])
        old_level = [self.website]
        cur_level = set()

        self.logger.info(
            f"Crawler started from {self.website} with {self.depth} depth, "
            f"{self.pause} second{'s'[:int(self.pause)^1]} delay and using {self.thread} "
            f"Thread{'s'[:self.thread^1]}. Excluding '{self.exclusion}' links."
        )

        ip_reset_thread_stop = False
        ip_reset_thread = threading.Thread(target=self.renew_tor_ip,args =(lambda : ip_reset_thread_stop, ))
        ip_reset_thread.start()

        # Json dictionary
        json_data = {}
        classification_data = {}
        # Depth
        for index in range(0, int(self.depth)):
            session = self.__get_tor_session()

            # Sumbit all the links to the thread pool
            futures = [
                self.__executor.submit(self.__crawl_link, url=url, session=session)
                for url in old_level
                if url not in json_data
            ]

            # Get the results from list of futures and update the json_data
            for future in as_completed(futures):
                url, url_data, response_code,classify_verified, output = future.result()
                if isinstance(response_code, int):
                    self.logger.debug("%s :: %d", url, response_code)
                    if classify_verified:
                        self.logger.info("%s :: %s", url, output)

                else:
                    error, exception = response_code
                    self.logger.debug("%s Error :: %s", error, url, exc_info=exception)

                # Add url_data to crawled links.
                cur_level = cur_level.union(url_data)

                print(f"-- Results: {len(cur_level)}\r", end="", flush=True)

                # Adding to json data
                json_data[url] = list(url_data)
                classification_data[url] = output

            # Get the next level withouth duplicates.
            clean_cur_level = cur_level.difference(ord_lst)
            # Merge both ord_lst and cur_level into ord_lst
            ord_lst = ord_lst.union(cur_level)
            # Replace old_level with clean_cur_level
            old_level = list(clean_cur_level)
            # Reset cur_level
            cur_level = set()
            self.logger.info("Step %d completed :: %d result(s)", index + 1, len(ord_lst))

            # Creating json
            with open(self.__files["network_structure"], "w", encoding="UTF-8") as lst_file:
                json.dump(json_data, lst_file, indent=2, sort_keys=False)

            with open(self.__files["network_structure_classification"], "w", encoding="UTF-8") as lst_file:
                json.dump(classification_data, lst_file, indent=2, sort_keys=False)

            with open(self.__files["links"], "w+", encoding="UTF-8") as file:
                for url in sorted(ord_lst):
                    file.write(f"{url}\n")

            # Pause time
            time.sleep(self.pause)

        # Close the executor, don't wait for all threads to finish
        self.__executor.shutdown(wait=False)
        # Stop the ip reset thread
        
        ip_reset_thread_stop = True
        ip_reset_thread.join()

        # Close the output files and return the json_data
        for file in self.__files.values():
            if isinstance(file, TextIOBase):
                file.close()

        return json_data
