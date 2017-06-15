import os


def mkdir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)

class cd:
    '''dir context manager'''
    def __init__(self,newpath):
        self.newpath = os.path.expanduser(newpath)

    def __enter__(self):
        self.savepath = os.getcwd()
        os.chdir(self.savepath)

    def __exit__(self,etype,value,traceback):
        os.chdir(self.savepath)


