from .UAVDark70 import UAVDark70Dataset
from .nat import NATDataset
from .uav import UAVDataset
from .nat_l import NAT_LDataset
from .nut_l import NUT_LDataset
from .UAVDark135 import UAVDark135Dataset
from .darktrack2021 import DarkTrack2021Dataset
from .watb import WATBDataset
class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):


        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        
        if 'UAVDark70' == name:
            dataset = UAVDark70Dataset(**kwargs)
        elif 'UAVDark135' == name:
            dataset = UAVDark135Dataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'DarkTrack2021' == name:
            dataset = DarkTrack2021Dataset(**kwargs)
        elif 'NAT' == name:
            dataset = NATDataset(**kwargs)
        elif 'NAT_L' == name:
            dataset = NAT_LDataset(**kwargs)
        elif 'NUT_L'==name:
            dataset=NUT_LDataset(**kwargs)
        elif 'WATB'==name:
            dataset=WATBDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

