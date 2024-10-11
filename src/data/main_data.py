import os
import sys

from .download import download_img_dataset

def main_data() -> None:
    '''
    Call al necessary verifications and methods for downloading image data.          

    Throw:
        FileNotFoundError: if necessary csv or json files are missing.

    Return:
        No data. Save all images on `data/raw/` directory.
    '''

    try:
        if not os.path.isfile('data/raw/all_image_urls.csv'):
            raise FileNotFoundError('all_image_urls.csv file not found on data/raw/')
        
        elif not os.path.isfile('data/raw/annotations.json'):
            raise FileNotFoundError('annotations.json file not found on data/raw/')
        
        if len(os.listdir('data/raw/')) - 2 < 15:
            download_img_dataset()
        
    except FileNotFoundError as e:
        print(e)

        sys.exit(1)

    
    
    