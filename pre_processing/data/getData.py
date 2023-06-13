import os
import subprocess
import json
import requests
from retrying import retry

def getData():

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "env_var.txt")
    pathname = os.path.join(dirname, "WinEvent4688.csv")
    print(filename)
    with open(filename) as env:
        lines = env.read().splitlines()
        username = lines[1]
        password = lines[2]
        url = lines[0]
        ip = lines[3]
        output = subprocess.check_output('curl -XPOST -H "kbn-xsrf: true" -u {}:{} "{}"'.format(username,password,url), shell=True, universal_newlines=True)
        json_obj = json.loads(output)
        #print(json_obj)
        path = json_obj['path']
        download_url = f"http://{ip}{path}"
        print(download_url)
        headers = {
                        "Content-Type": "text/csv",
                        "username":username,
                        "password":password
                    }
        max_retries = 10
        retry_delay = 10
        @retry(stop_max_attempt_number=max_retries, wait_fixed=(retry_delay * 1000))
        def make_request():
            response = requests.get(download_url, headers=headers, auth=(username,password))
            response.raise_for_status()
            return response
        try:
            response = make_request()
            print("Making request...")
            if response.status_code == 200:
                with open(pathname, "wb") as f:
                    f.write(response.content)
                print("Request Sucessful")
            else:
                print(response)
        except requests.exceptions.RequestException as e:
            print("Request failed:", str(e))
                
        
if __name__ == '__main__':
    getData()