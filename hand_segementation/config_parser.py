import configparser
import io

CONFIGFILE_NAME = '/Users/gejianbang/PycharmProjects/boneage/config.ini'

class config_parser:
    def __init__(self):
        self.configfile_name = CONFIGFILE_NAME
        self.config = configparser.RawConfigParser(allow_no_value=True)
        self.fp = open(self.configfile_name,'r+')
        self.read = self.fp.read()
        self.config.read(self.configfile_name)
        self.fp.close()

    def write(self,section_name,keyword,value):
        self.fp = open(self.configfile_name,'w')
        if not section_name in self.config.sections():
            self.config.add_section(section_name)
        self.config.set(section_name,keyword,value)
        self.config.write(self.fp)
        self.fp.close()

    def get(self,section_name,keyword):
        return self.config.get(section_name,keyword)

    def getint(self,section_name,keyword):
        return self.config.getint(section_name,keyword)

    def getboolean(self,section_name,keyword):
        return self.getboolean(section_name,keyword)

    def getfloat(self,section_name,keyword):
        return self.getfloat(section_name,keyword)

    def printa(self):
        print('hello')

    def printb(self):
        self.printa()

def init_config():
    Config = config_parser()
    Config.write("PATH", "root","../RSNA_boneage_dataset/boneage-training-dataset/")
    Config.write("PATH", "train", 'boneage-training-dataset/')
    Config.write("PATH", "mask",'mask/')
    Config.write('PARAMS','img_size',512)


if __name__=='__main__':
    # init_config()
    cfg = config_parser()
    a = cfg.get('PATH','root')
    print(a)