import gdown
from zipfile import ZipFile

# download zip from google drive:
downUrl = 'https://drive.google.com/uc?id=1TGXunp4cZcODfyODZabaggU3LuPrQdHO'
folder_path = 'data/DIV2K.zip' # choose where to download the files

# gdown.download(downUrl, folder_path, quiet=False)

# Unzip the file
with ZipFile(folder_path, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
    zipObj.extractall('data/DIV2K/')