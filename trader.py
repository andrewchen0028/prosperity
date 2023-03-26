from typing import Dict, List, Any
from numpy.linalg import inv
from datamodel import *
from io import BytesIO
import numpy as np
import math


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


class BerryModel:
    def __init__(self):
        self.to_hidden_weight = bytes_to_array(
            b'\x93NUMPY\x01\x00v\x00{\'descr\': \'<f2\', \'fortran_order\': False, \'shape\': (40, 24), }                                                        \n\x15\xb3<4\xdf\xa9\x02-\x1e\xa9\x89\x95y\xa5{3\xc92*1m2\x80\xacB/\x0f\xabf)H0\xb2/\x07\xb0b\xad\xd7\xa5\x101\xf1!\xe0\xb0\xcf\xb5\xcd/\xed\xad\x99\xad\t\xb2\xab.\xb4\xb140<\xb4\xcf\xb4\xfc\x9e\xd4\xa8\xef\xb3\xac\xb0\x97\xa0\x822\x87-\x90\xac\xa8\xab=\x9e\xd4\xa5\xa3\xad\x9c\xb2\xa95\x195\xaf\xac[\xaeG\xb0]-\xb3$u0w\xaf\xe2\xb0?*\x07-\xf9\x17\x9c2\xe1&\xc20\x03\xa2I3\x97\xa0V\x1d\xab/\xd9\xb2\n\xb0J\xae\xdc\'\x97\xad\x96\xac\x1d\xb5B,\xef\xb0\xe6,\xce.\xc0\xb2\xe70\xeb\xb3\xb90w\xad;0-\xae#\xa8J.\xec\xb2\x8a\xaf\r\xafz,\x14#V,\xc3-\xc31\xae3\x94\xaf\x884\xf6\xaeQ1\x08/?\xa1\xf02\x10$\xc9/G2\xad\xa4\xad0\xc1\xb1\x191u\xb0\xd2+\x8a!c3\x88162\xb4\xa9m0B\xaf\xb8\xb7I\xadi,73\xe2\xb0\xda\xaaw\xae!)a$T/\x14\xad\x07\xad\xef.!\xa5z,\xd6-\x941\xfb\xb2\x19\xb0\xa31\x1d\xaf\x1b3D\xa1\xb40C6\xb1\xaa\x95.\xe8\xac-\xaa\xaa\xa9\xb3\xb4M3\x99\xb4N.\xea-\xee.\xf9\xae\xbd,q\xab\xaf0\xdd\xacv.\xd91\x15\xb1}\xa5\xa7.\xdc\xb1\r2&5\x10\xb4\xb64\xfc\xa4\xc4\xb1d&_*\x8c4\xf2\xad\xbc0>\xa6t\xac\xcb\xac\xa2.`\xaco\xaf\x8c*\x04\xb0\x9f-\xee\xb1i0\xdc&K+\xf0\x94\xae\xb6\'\xa5|\xae\x0c\xac\xc8\xb2G\xaa\xba\xac\x05\xb5\x91\xae*\x1f\xd5\xa2\x02\xb1\xfc\x9el.$$\x040\xff\xb0\xa1\xb0I-\x8f\xaay-\xeb+\xcb-\xeb2\xa48\xee\xb1K,\xd6(\xff\xb2\x9b2\xa7\xb3\xbf\xb1z\xaaR\x9e\xc51\xc4\xa5C\xae\x06\xa8x\xac\x9d\xa7r\xb3\x01\xa1W\xa79)\x0c\xb2\xf63&\xac\xd7\xa5C2\x0e\xb4\xd9-\xd6\xb0\x00\xb2\xd3#(21\xad\xe4\xb0\x11\xb0f%\xf2)g\xac\xe0\x1f\xf4)\xb6\xaeQ1\t/\xbd2\xbf\x99\x8b\xa8\xed\xb0<2&2\x996\x0b\xb4\xcf.\xfd%O\xa0\\\xa8\xc1\xaf\r0%\xad4\xa9j\xb2\xa5\xac?\xa5\x00\xb2\x99.#\xad\x83\xb17\xac\xa2\xa0\x11\xac\x9b)X\xb1e\'\xc6\'\xec6\x97\xb2C,\x8d\xa7\xf2,\x97\xa8,3e\'\x9a1\x94\xb0J\xadA\xa0G\xb0\xfc+\xe4\xacn\xa7|\xabd\xaf\x0f0\xb3\xa7\xce\xa9\xcb\xb2\x1a\xb0M2B5-/@.\xf02E\xaff\x1c,0\xe30\xcf.\x87\xafI\xa5*1\xb3\xad\x8a+p\xac\xe71\x121\x17\xb0\xb5-\x14*\xc6\xa1\xa8\xb2\xf0\xb0\x15\xb1]\xb6\xdc\xa1\x16/\x11\xb3r\xab\xa2\xabq\xb1\xae\'\xf50\x1d\xb1\xb9\xa4=\xb0\xb2\'\xb03\xaf\xae\x97\xab\xd5+\x91\xa9F\xb0W\xb2?\xafN\'\x0b)\xec2T81\xb5\xa71\xe7\xa3\x19\xaci0\xed1\xdb\xb1\xa8\xac\x84+x\xb4\xa7-k\xad\x92\xb1\\,4\xadF4\x80\xb4*\xb2\xcd\xac\x08\xab\xef4\xf12\x88\xacY\xb2\xed0z\xaej(\xc9\xab \x14\x0c\xae&0\x8d\xb1\x1d\xb2\xa8&\x19*\xfb\xafX\xae\x810\x98*9\xaf\xd1-\x97\xae14\x15.\x911\x8d\xa8\x982\xa2\xa7\xc41g\xa3"\xadr0\x9d/$\xac\xbd\xac\x9c.Q\xa8\x8c\xaf/\xb2!(\xc7+\xf4\x1e\x98\'\xaf0k\xb1\x00\xb4\xfb\xad\xc2\xb4\n\xad\x9b2z-\xa0/t\xb0\xa2%\xfa.c1\xb8\xaf@\xac\xd3\xb0 2\xd2\xb3\xcb\xafN(\xa2\xb1Y\xae\xa81F\xb0\xa1*^\xb1N+>\xad\xd92w\xabj0\xeb3\xe66/3\xd9\xb0\x96\xad\x05/\xa9\xb1\x1b/\xcb+\x85\xb1..\xf6\xa8\xe7\xaf\x194n\x9a\x0f3+3u-\xe20\xf2\xa8\xf8!\x16+\xe83\x902\xae\xaa\xf4\xb4\xfc0\xca\xa4\xbb2*+\xea\xa8\xdd1\xf00\xbf0\xf41T/Z\xab\xa8"\x95.r\'\xbc)J0\x91\xa7\xe6\xaa\xe31\xb70\xcb0\x8a\xb2\xe0\xb3\xb1\xb7\'\xa7\xb14\xd3\xac\xa00\r/\xac\xacE\xb1\x84%,0\x9a\xaek\xad\x98/5\xac\x1b0\x99\xb1\x1a-\x95\xab#\xb0\xd2\xaf\x15,\x17\xacY\xa3/\xad\xb6\xb4\x194\x8e\xb2\x96\xb0{0\x0e\xaf\xbe\xaeA()\xb3\xc9*\xb6.0\xb2t#\x86\'}\xad\xc83\'\xb0\x164\x81\xb1\x1e\xb0\xd5\xae\x00.\x8d\xb2\x1f\xb1\xda\xa9\xb1!\x1b\xb4\x08\xafL1\xe9\xaf\x05\xb0\x103n\'\\\xa9>+\xe6\xb1\xb2&\xc9\xae\xb9\xb105\xce\xb0\xaf3\xfe\xab\xb7-\xbe"\x161$\xb3\xd5\xb1\x08\xb4n.\n.\xc8\xa53,x\xb2H\xb1Z,\xd0\xb2\xcb\xa8\xe2)\x8d\xa6\xed,\x90\xaaa1\xfc\xad\xa8\xb0\xcc\xb0\xe8+u0\xee\xb0X.]\x99\xa4*\x81\xb8\xcb/\x13-\xe9)\n\xac\xb1/i/v\xaa\x0c\xb3\x98!\x8e\x95,/\x90\xb0\n\xb1 03\xb3\xc8\xabt0/\xb1\xa6\xb0_1\xbb\xb0\x83\xafC/\x0c7\x9b0n%\x1e\xb2\x96\xady\xb040\xc1,F\xb4\xaf$K0\xda\xb0O0\x8d1A\xadi\x1e\xfe\xb0g\xa9\x84\xb0\xfc\xab\x8e\xb1\x8b1\xfd\xb06*t8\x8a04\x94\xc8-q\xa5`$\x10 \xcd\xb3\xbc\xaf\xab0_\x11\xee,7&\x0f\xad\xd0\xb1\xc6*!\xa0\xad\xae\x01\xa8\x8a.\xcd\xb0\xbc\xaf@.*\xad\xb2\xb46\xaf\x844\x98\xadC\xb1m\xadB.\xa3/\xe7&u/\xa11\x831\x95\xae\xb1\xb2\xba0t\xb0[\x99\x9f\xa8\x93\xb0f\xaf\x16/\xf5&\xee$\xb0\xb3\xbc\xb4C\xaf\x88\xb0H\xae\x9f\xb2\xc8\xa2a,?\xb1\xc41\x9f\xad\x85\xb2\x8e\xaf\x94\xb0\x083\xce*\x80-y\xacw\xb2\x072\xfd\xb0H1\xd40\xd72\x9f\xa8\x968\x833J\xa0\xf8\xb2&-a\xb3o0\x87\xa9\xa5\xa7\x17\xabC-_\xb4S*\x8d2\x0c\xb2\xa8,\xe7\xb3\xfe\xa5\x17- 1\x030\xcf&c!\x89\x1a\xd2,o-I\xb03\xb3\xf90Q$\xca\xb0\xe1\xa0\xf9,P\xb4\xd81e\xaa\x8c\xb1\xad.\xbf\xb2\xf3\xaft\xaf/\xaf"\xad\xd7*\x85\xae[\xac\x0e+z1=8\xd7\xac\xdb,^2\xc7\xa0\x80\xb1\xea2\x01,\x1a\x1d\x80\x1c\xd5\x9a\xf62\x03(3\x1c\x81\xad\x8e\xae[\xa5+\xa8o\xb2\xa3\xb2\xf4\xab\xcf2\x87,\xa8\xaf#\xb8a2\xb3\xb4\x1b1#1\xc5\xb0k\xa4J\xa8:\xb0s\xadB\xb05\xab\xc00\x02\xae\xe22%.\x1d\xb5I\xac\xd9\xb0\xf60x2"\xb4\xe1\x9f%4\x861\n, \xad\xfe"|\x1e\xf1\xb4\x90-5\xae\x9e\xb3\x9c\xb15 \x06\xb044\xd01\x03\xaf\xf6/\x15\xb3S/|\x96u4R\'\xc2*\xbf\xa9@\xadv%\x0f+/\xb5\x00\xa6x/\xdc1\x82\xb4\xba0!\xb4\xef\xab\xb42~\xacQ\xb1\x82\xad\xb0-X1\xf7\xb0\xb02\xc4\xac\xa9\xa2g\xab=\xa4\xe6\xb0\xe24\xb61v\xb14-\x1a\xb3\x90, \xb07-b\x97\xad\xb3\x88\xae\xc2\xab\xb5\xab}\xb3S\xab\x03+\x1f$\x831\xf0&j\xa9\xc42\xd6,d.\x0b\xb2\xc2$\x87644;\xb1\x08\xaa\xca/b0\xad \x0c+\xe9/\x18-\x19\xa1\xd6\xa2\\\xa3\x9c\xadK\xa0\xf8.a\xb1\x9d-\xc7.\xe5/\xc0\xb1.\xa6\x0c\xb3m\xad%\xb7\x13\xa6\x071X+[\xafn\xac\xc6$\x84,\x8f$x\xaa;\xaf\x94\xac\xe12\x85\xa9\x054\xea3:\xa8M\xac}\xaf\x9e\xa6\xb70\x9d\xaf\x0f\xad\xaf\xb1\xbc\xb7\x14\xadX4\x93\xb2\x84\xb093,\xb30\xa92\x9aC3;\xaa\x95\xa5\x9d/\xd9\xabQ\xac\x0b\xae\x16,\xd9\xab\x1b0\xc6\xb4\x88$8\xb12\xb4\x8d\xadS\xaf'
        ).T
        self.to_hidden_bias = bytes_to_array(
            b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (40,), }                                                           \n3\xa7\xfa\xaep\x9c\xad\xb2\r1\xb4-\xa2\xaf/\xb1./3,K1\xf3\xb160\x86$\x982\x14(\x0e3\x10\xa5\x1d1\xab2\x97-:\xb2\x87\xb1h\xb0\x08\xac\x18\xad1,\xbf)\x1f2\xc22\xea\xa9\xef\xaa:.\xd9/@*\xb3)6/?\xb1\xe4\xb0\x1b\xb2"
        )
        self.hidden_weight = bytes_to_array(
            b'\x93NUMPY\x01\x00v\x00{\'descr\': \'<f2\', \'fortran_order\': False, \'shape\': (40, 40), }                                                        \n\xef h\xaf\xf9\xa8\xaf\xb0\xc9,\xe8\xac41\xbc1\x00\xa6F\x9e\xda1n3\xf8\xaci\xb1\x9f1\xd8\x8f\x8a\x9cT\xadS.\x8c!\xc7\xb2s\x1b\xa1\xb0\xe3\xb1F\xb1q\x1a\x051\xcf\xb1d/?5\x9d%\xd0+<"\xb5\xb1d)\xf6\xb3\x8b,M\xb4\xc0\xadG-\x00\xa8g0\xba-\x8a*\x85\xafk4\xdc\xad\xa6,%.\xdc.#5\x9b\xad\xf1\xa0\xca\xab\x08\xaa\x864-\xabE\xb0\x071\xdc\'u\xacK\xb2q\xb2G\xb5\xfc\xa9t\xa8;\xaa<\xa42$\x0c*s\xaf8\xa8>/\xd9\xb0\xf9)\xe9\xa6\xe12\xb1\xb4\xb2\xaf\x9a\xb0h\xa8v\xb1\xa2&5\xacM\xaaw3(\xb3U\xa6\xff(*%v\xab\xa5\'*,\xea*\xf1\xb0\x1b5\x05-\t/\x06/\x96\xa3\xc62\x1e\x17H\xb4\xfd\xb3\xfe\xaaq*6)\x030\x80\xae\x06\xb1\x1f\xb1\xca\xac\xe83\xd3\xb0%(W\xb4\xb9\xb0\xa5+\x88,G\xb0\x03\xb0#\xa4H\xa7\x9e(l-\xe8)\x18\xb0:\xb2!.\xbb\xb2\x83(.\xb1Y*\xc2,[0\x88\xb22/\x90\xb1\xec\xafc\xaa<-\x93\xb3u-=4\xdd\x9cK/\t/\xae\xafH\xb1t\xaa\xc70\xd10\xa9\xaf\x164\x89/q\x9f\xff%\xc3\xa0\xd0+\xc2\xaf\xf4.\xb2\'B\xb0G\xb2\x0e.\xf1/X/\x0f0\xb2\xa5D0\xcd.w,T0\xad\xacV,Y\xac\x9b(\xc5\xa4\x9c0\xbc\xb2N\xaf\xa3+\xbc\xb1\x93\xa98-\xcb.\xcc+"\xb2\x8a\xa2t\x9cQ)\r2\x1f\xb2\x92\xb3N\xb3\x12.\xf6\xa4\xe9-\x91\x10\x1b\xad\xbb\xa8.,\xc5\xae\x89.\x05\xa4\xc91\xde\xa6\x1b\xb0\xe14\x16\'"*\xf5/\x13\xacD\xb3\xe3(\xaf*\xe5/B-p,$/\n\xad6\xb0}+\xab\xb0!\xb2%\xa9\xb3/\xe7(\xc2\xa9\xe84\xd2\xae\xf2\xa45\xacb4\xba\xadt-|\xac\x05\xae\x12\xb4\xd0\xafK+:\xac=\xa8],"2S\xad\xd0\xa3]\xae\\\xafZ\xb0\x16/D\xad\xfd\xac\x01\xa1\xbd\x1d\x9f.\x01+(\xae\xdc\xaf\xab24\xad\xdc\xae$\xb3<\xb2\xe4.\x88,\x95\xae\xba\xacD0\xa4\xaa\xe5&\xac\xb0\x9723\xa4\x13\xb2\xf9*p1Z\xb0\xb3\x9a\xbc\xac4\x19^-\xa5(\xa1\xacg\xb0\x1b1\xa6\xads\xa5\xb93\x07215\xf8+\xbc.\xeb\xac\x16"Q\xac%1\x95\xab90\x86,m\xb3n\xac2\xad!\xb1\xdc\xb0%#:/\x00,R/+4\x9f\'M3\xd2&\xd4\xaa~\xb2\xce1{/\x12\xb3/\xb3\x03\xb1\xb9-\xce\xb1\x1a\xa8\x11\xb1\x84\xab\x98/\xb3+\xe0\xb1\x15\xb0!\xa9\x88\xae>\xae\xae\xb3\xa6\xb0\x15\xaav\xa141\x17,]\xb0333\xabi\xafK4\xb2/\xc4\xa4\xf5\xb0\x04.\xe1\x9c\xb3+4\xb3U0L.\x8e\xa7\x0c2A4|4\x1a\xa430\x06.\xfb\xb1\x9c\x9c\xfe \xa9*),\xa4\xb0\x1f&-2\xb8\xb1\xd8/\xf00\xe7-\xb6\xab\x9e2\x19)\x9d1\xb3+\x02//*v4*\xacZ\xb3X\xa9U\xb0\x8b\xb1!\xae\x1c,\xc2\x1c\xb2\x1f\xfc\xa9F3v\xa7\x8c3&\xb2\xbe.\x02,\xfc0\x9d!x\xb1.\xb2/.\x92\xab\x92\xae\x93\xa502\x8d\xad\\\x94\xce\xa4\x8b\xa7\x13\xb1\x16\xad/\xb2\xba\xaa\x0c.\xa6 D*\xe3\xb4\xa61G\xa8\xc4)?3`2q\xac\x7f,<1\xd1\xaf-\xad\xc4-\xdd\xab\r\xa3U!\xa13G0d\xb0\xf1.\n3.,C-Y$\xf9\xa1F\xb4C\xaf\x92\xabG\xb0\x17/\xab\xa7\x83\xa1\xf9(I1\xe0\xa6\xad.\xe4\xa0\xde2\xe0,\x0f1\x1d0Z020\x0e\xaa\xf40\xe9\xb3#\xb0\xab\xa6a\xb0J\xa8"\xaf\x8e\xa7\xd4\'C\xaf\xe32L/\xe01\xcd-.,o\xac\xc6-\x02\xae(2\xc9\x9en.\xe2\xac\x841"\xb2\x18\xa5\x96\xb1\x98,\xf5\xaf\xd9\x99\x08,\'$\xa8\xad\x9b\xb0\xfa\xac&\xab\xe41W-\x040S%Y\xa6\xf9(a1r25"\xc0\xb3D\xb0\xc4\xa4\x19\xb4\xf6\xb0\x85\xafj0i,\xde\xaeI\xb1{3\xab\xb0\xe1\xad+\xb4\x86\xb0\x80\xa6\'\xaf\xd8\xab]3A\xb5@,\x93\x96\x13(\xac\xab\xdb\xb2\xd7#Y\xb2F\xabG\xb3\xd6\xacv\xad\xef-\xce,"%\xe6\xb2\xb0\xab\x1c0\xe9\'H4\xfb*&(C*\xf5\xaev\xb1\x1a\xa6\x03.\t-M0I*}\xb45,9\xb0\x8f\xb0\x99\xb4|\xb3\x86\xabq\x1d`%W\xaf9\xa5\x81\xac\x95*C\x1c\x841\xb5)\x82)\xa30\x030@2\xc52w+\xfc\xb2 5\x18\xae\x13\xb1|\xafc*\xb1(6\xb4Y\xac\x9d/\x82\xb1&\xa2U/\x8f2O\xb3\x0f)\xb1+{0\x911R+|\xad\xc40\xc2\xaei\xab\xc8\xb3\xf4(\x85\xa6j$\xbf1\x00\xae\xd3%g(L\xa9^1\xb3\xb0\xbb4J\xa7\x89-6\'\x1f/0\xad\xb54n\xa3\x1b/\xe0\xabu/o\xb1\n\xad\x91\xa0f\xb1Q\xb0\x8e\xacU0\x1a\xad\xe6\xab\x10\xb3A2\xae*\xb14\xdf\xb0\x96\xa8\xd7\xaf\xed2\x1c\xaa\x90\xac\xe1\xb46\xabH"a\xb2\xc9%\n\xafq4~-j.\x801\xab\xb6\x8c\xadh\xa2~\xac\x86\xb1\x9b/J\xb4\x0c\xafy1\xb8\x901\xb2\t\xa8\xf04%\xaf^\xa4\xcb1\xb8,\x13\xab4\xb2T\xa1\n0\xc7\xb3\x1e\xae\xd6\xac\xde1\xc5\xb1\x96\xa9\x81\xaa3\xb0\xf5\xa38#z\xb0N0\xb6\xae\xdd\x9f\xf4\xb1m4\xe5*a-\xdc0\x16\xb4\x92*\x08\xb5\xf3\xa5Z\xb2&3\xf4\xb3\x88\xac\x16"\xc3\xacz\x9c\xb42d/-*\xd2\xa5d\xa7E2\xce\xa8\xef\xb1g\xac\xdb2\xe3\xb2\xfd\xa1\xa2\xb5S2\x86\xac\xc4,\x890>\xad\x0c"\xef4\xbd%\x1a%<\xb5\x1f1p\xae\xa7-10[+-.^\xb0\xcd\xabj*p\xb1\x84.D)n\xb1\xe40\x93\xb3|\xaa\xef\xb2B\x1f\xe60\x0c\x1a\x91(\x07\xb0d\xaa\xee\x98\xe6(\xcb1`\xad>2\xfe%\xb4\xad\x03\xa3\xd6\xb1\xd1\xa5X\xb5-\x1eO(\xc6\xb1\x13.\xc5$\x9e(t\xad\xcf,\xef\xaf)\xa6\x97\xaaE\xb4*\x13\xed\xae\xa62;\xa1\xc1\xac\x9b \xdd\xa5"\xaf\x841\xa8)\xc4/\x9b\x9c}\xb1\x87\x17%$\x14\xb2~\xb0\xfd/r-\t\xb0\xe6\xb0\x0c!F\xb1A3>\xab/+\x88\xad\xdc0\xf3\xb0n\xae\xf2\xb4|\xa3\xba\xadS\xadi0%\xa9\xf6\xb3\xca.h.!\xb1\xad3\xd7\xa1\xec/\xa1\x1e\x07"\xd1\xb2\xba.\x83\xb2I\xad\x95-#1\x8226\xaa\x1d\xb0\xa30408*\xaf\xa9\xf64v\xb0\xab\xb0\xaf.\xc42]/\x9e*\xc20\xf5.s)\x100\xf7\xa9\x91\xae\xb7\xb0\xb8$;\xb3b\xaf\t00+>$\x83\xb2\xe0\xaf8\xb29\xab\xbb*\x1f\xac\x01\xb0\xd4.\xf0\xa7\xd7)b.\xe2\xa26\xb1\x810\xcc(@0\xa8)\'+\x9f&\x98\xab\xe8\xae\x9d1\xe2-\x0b\x1dw"\x14\x94M,2\xb110\xe8\xadz\xb06.P1*,\x7f\xab\xec-I*w\xb2\xbf-\x96\xad\x184\xa53\xd3\xb4\x9d-}\x15\x8d\x10*&\xed\xac\xaf\xb2o+b1\x1e\xb0(\xb39,\x1d1\x97(\xbb.\x95/\xf90Z2\xff\xad\xd0,\x880\xc2\xb2\x90\xad\x12\xab\xd5\x1a\xfc)\t\xaas3\xee\xaf\x93\xad#4\xb2\'\xc8(6.T\x18F\xb2,/U+U)\xf01\x91\xab\x104?\xb4\x07+\x9d\xb1^\xa9,\xa4[\xa8\xf6\xa2d\xaf\xc6\xb4o\xac\x1b\xab\xc9\xad\x934g/\xd2,Z\x1a\xa6/*0j)\x00\xaf\xee\xach\xad\xda1\x92\xac6\xac6,\x1a0\xf10\xa2\xaa\x834\xd2-M\xb5\xa1\xa7\n&\xb3\xa4v-\xa0\xb4\x8d&L\xae\xd5\xab|\xb4\xcd\xb2\x8f0i0}\xb2\xdd4\xd6,\x1a+\x84.\xf1+\xe41r1\x8d\xa7e\xade\xaa\xf6\xacn\xb2h0\xe5!\xb3\xaaU\x9d\x89\x1e\x8a3\xc1/f*\xea\xb1\x0b\xb0\xcb0\x111P\xaf\x11-V\xac\xb6\xb1\x8b-\xc4\xae\xeb*\'2\x82/;\xb2](\xad\xb2(\x9a\xaa\xb0>\xa9\xbf\xab\xcc/\x88\xad\xee\xad\x1c\xb5\xb3\xac;\r\x1c,\x071\xd8-\xbd2C%\x0b-=/\xc3\xa5I\x14\x98)B-a1\x86)\xe8\xa9\x81-\xde\xa4\x91*\x0c\xadx1\x1b\xac\xbb%V0\x991&\xad--o\xa9\xaa\xaad+}(\xbd/q\xaf4\xac\xdb\xb2\x1d\xaa`\xadD\x96C0\xc4%Q.O\'(\xa8j/a\xae6.\xb9,\xa9-\xe7\xb0\xb5\xae\xe3+]\'\xd0\xacf\xb0\x0b\xac\x04,\xb30[0\x8b\xa4#\xae(\xa9\x9f"m,\xee0d\x91\x1e.\x06\xac7\xa6[\xb1=3\xeb)\xa6\xb2\t+1\xaf\x10\xae\xf9\xa3\x0c\xae\x92(\xf8\x99\xd4*\x951\xaf.}\xb1\xee1\xa35w2\xb0\xb0\xa43\xe80H\xb0\x13\x1c\xe60m-\xe2+*,\xca1\xa1-",\x95(.\xa9\xe6\xb0\xb4\xb11\x1bC\xab\xee\'\x1d\xad\xe1\xa0\xd90\x92\xb4C0j(Y\xa6`\xb2p\xad\xd6\xac\xe9$%\xb2\xc7\xacA\xb1\x883K\xaaw+7\xb0\x98,\n(\xd8\xa4\xdc W\x1c\xe2\xaa\x15\xac\xf2.T\xabt\xa8.\xb3\x82+\xb6\xadI,\xf8\xb0\x10\x1bG\xb1\x9f)\xcd\xac\x93\xa9\xc7\xaf\x8a1\x88\xb0\xde\x17(/\xd1+3\xaf\x952w\xac?\xac\x87\xa6;(\x011\xf4-\xbd\xb0H-\xa4\xadz/\x06/\xd0\xb0\xd8-\xbb1\x88\xb2\x9d!\xe0.\x7f\x9do\xb33\xa9Y&\x8d#P\xb2;%\xbd"\xc2\xae.\xa8\xa90_.\x94,K\x9ep1\xda\xab\xe2\xae.\xb3\xb5)\x04\x90&1#-\xbf1\xa1\xb3\xad\xa8K\xb2\xa8\xa5\xd0\xa9\xdf.\x04,\xc7\xaa"\xb2b0b*M.\\3\x00\xa9\x03\xa2=-\xe91\x88/c\xb07+\xa00\xe7(C1I\xafg\x1e\xac\xab<\xaf\xa2\xaa\xd9\xb0\xbe&3\xb5\xf93\x9e\xa7\xde1\xf9\xab\xf7/\x1a\xa0_\xab\x7f\xaa8\xa5\xa0-K3\xb5\xaa\n\xa4\xa1\xac\xa4,\xa2)j\xa2\xe2-\xce\xae]\xb0\xa7\xac9\xaf\x9b\xac\x1009.\x0f2\xaf\xac\xbc-\xcc\xaa\xb0\xb0=\xb1Q!B2w/\xa80\xeb\xab\xbb)\xe9.\xb6/#2\xf4/\xce/\x1e\xaa\x8d-\xc4,\x04\xabI$\x0c0\x051K2\r\xaf\xbc\'5*\xd7/\xcd\'l00\x9b\xa13h&E)s&X\xaa\x14\xac\xaa\xb1Q2\xbb-\xd6\x1d\xa9,\xb6\x9c\xb4\xb1>,\xb5(\x02\xac\x7f\xafW\xae\x8e5m\xb1\xbe\xb0x"\xa42\x1f\xad\xc1\xb08\xb2\xb8,(\x9f\x99/\x12,\xda\xb4\xe8-\x03\xae)\xb0\xb6\xad\xce0 \xb0\xb8\xad30\x07\xaa\xc3\xaa\xbd0\x8c3\x9b%z\xad>)\xc20\xa6\xa9`\xb3H\xa4/4c\xb4\x8e\xafy\xaf\x97-|(\xb4\xaa\xec\x9cy\xae|1\x034{+M)\x93-\x02\xb2\xd2.\xfd\xb4\xd92.1\x05\xb3\xbb\xa0\x811\xdb\xae? \x91\xac\xb3/o\xaa\x16)o-\xe91\xf9\xb1q\xad\x06\xa4^\x98\x90.\x840L\xb1\x03\xb1f\x1b\xfd\xa9<"\x983z\xa8\xca \xa6.\xc6\xad\x13\xadM*\t,\x911a)>-+3|\xab>\xa9\xb8\xb2\'4\xa9\xa6\'\x1di2_\xb5\x94\'o\xad1\xb20\xb1\x103*\xb1\x91\xa8F09\xb1A\xaf21\x1f.\x05-\xc2\xa5n2\xfa2V\xb1\xbc\xae\x10\xaf\xf7&e\xb0\x94\xaf\xbf\xb5a0\xbe-\xc9\xab\xa8\xaeg\xa0\x150l.*,\xa7(70\x96\xac\x12,\xa3)\x11 \x07/c\xab;0\x170\x17\xa6$\xa0\x16,+\xae\xad\xac\'\xaa\xc8\xaf\x12\xb1\xa40r\xb0\\\xb2\x00\xadT\xb1G\'\xc9\xb02\xafh\xb1\xbb\xa4v\xb2*0\xba0*,\xd3\x1e\xc6-\x1e\x9e\xfc)\xd0\xad\x101\x0b\xb2\r2\xa2\xb3\xa5\x13F#7\xa1\x96*J\x18\xce-\xd7\xb3\xd3/\xe4-&\xadw\xb0\xfe$R\xb2\xe61\t\xb4~1\xc4%\'\xace1\xbc\xafq\xb3\xbb\xad\xf2(q\xb1U+~1\xa60\xe8\xb2\xcf2N)5.\x00\xb4h4C0\x1d3I%\xcb\xa7\xdb\xac\x81\xad\x93.\xfb\xad\x94\xb1\xb6\xac@0A0u14*m3\x031\x884\x97\xac\xfa\x1d\xa1\xb0,3\xa2,).\xba#>4\xb6\xa8\xa2\xaav\xaf;\xaf\xd6\xb4U\xae<\xa5\xb7\x1e\xac\xaa+*\xfc4\xa7\xadm\xa6$\xb0\x89\xa5\xb8(\xc0\xa3\x88.#\xb5\x8d\xb2\x00\x9d\x01\xad\xb51\xc5+\x970\xd9\xae~(\xac\xa8\x1a\xa9\xbe-\xd03k\xb0\xf3\xab$\xa6J\xb1:\xa0\x13+\x1b,\'*t(\xbc+\xde/\xaa\xb1\xbe,$5q1\r\xad\xde1\xd6\xac=\xaa\xab\xad\x9d)\x8b/M\xaf\x06\xb2\x080\x83\xad7\xb0,2\xe4-\x9a*'
        ).T
        self.hidden_bias = bytes_to_array(
            b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (40,), }                                                           \n\x1d\x9d\r\xa9\xc6\xad\xdd\xae\x9f\xb0\x8d(~\xb0!3\x0b\xb0\xbf1x\xad:\xa7w\xae\xf3\xaa\xd9(\xdf(\t,\xe51\xef&#\xac\x99*6\xa9\x13\xa8\xeb\xac\x07.\x960\x86\xb0\x1e\xa4\xf3\xa9\xf2\xb0O\xb1|\xaa\x10\xaa\xa4)\r'\x87,\xb9-u\xa0.3\x17,"
        )
        self.to_out_weight = bytes_to_array(
            b'\x93NUMPY\x01\x00v\x00{\'descr\': \'<f2\', \'fortran_order\': False, \'shape\': (2, 40), }                                                         \n\xbb\xb2\xc9\xb1\x92/M\xb0\xfc\xac\x12\xb3m,\x8e\xb4s#\x1c\xb5\xf4$\x0b\xb0\x170^4\xce\xb4]\xb5\x005(6~1\xf7\xb4K\xb5c1V3z4:4\x15(D\xb1d\xb1\xec\xb0j\xab(\xae8\xaa\xd4\xb0\x8b4\xab\xb2\xcf5\xcb\xb2#\xb5F\xb42\xad"\xb5p\xb42\xb5\xa94Y\xb2\xa2\xa8e\xadx\xb0\x824\x01\xb0:4\xaf\x91_\xaf\n\xb2W\xb0\xe6\xae\x0b\x9ee)\xbf\xb1\xd7\x9ee,\xf3\xac}."3k\xb3\xd70\xf3\xa124\xe1& 0\xa11V\xb4\xf8\xb24(\xed2Y(}-\xc6-\xf4\xb283'
        ).T
        self.to_out_bias = bytes_to_array(
            b"\x93NUMPY\x01\x00v\x00{'descr': '<f2', 'fortran_order': False, 'shape': (2,), }                                                            \nZ(\x9e/"
        )

    def __call__(self, x):
        x = x @ self.to_hidden_weight + self.to_hidden_bias
        x = x @ self.hidden_weight + self.hidden_bias
        x = np.maximum(x, 0)
        x = x @ self.to_out_weight + self.to_out_bias
        return np.tanh(x)


class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""


class BaseStrategy:
    def __init__(self):
        self.orders = {}
        self.products = []
        self.state = None
        self.current_steps = 0

    def create_data_dict(self):
        self.data = {
            'ask': [0 for _ in range(len(self.products))],
            'bid': [0 for _ in range(len(self.products))],
            'mid': [[] for _ in range(len(self.products))],
        }

    def place_order(self, i, target_pos):
        product = self.products[i]
        pos = self.state.position.get(product, 0)

        if pos == target_pos:
            return

        else:
            order_size = target_pos - pos

        if order_size > 0:
            price = self.data['ask'][i]

        elif order_size < 0:
            price = self.data['bid'][i]

        self.orders[product].append(Order(product, price, order_size))

    def accumulate(self):
        return

    def strategy(self):
        raise NotImplementedError

    def __call__(self, state: TradingState) -> Dict[str, List[Order]]:
        self.state = state
        self.orders = {product: [] for product in self.products}
        self.accumulate()
        self.current_steps += 1
        self.strategy()
        return self.orders


class AvellanedaMM(BaseStrategy):
    def __init__(self, products: str, y: float, k: float, limit: int = 20, vol_window: int = 30):
        super().__init__()
        self.products = [products]
        self.y = y
        self.k = k
        self.limit = limit
        self.vol_window = vol_window

        self.create_data_dict()
        self.data = {
            'log_return': [],
        }

    def calc_prices(self):
        depth = self.state.order_depths[self.products[0]]
        bids = list(depth.buy_orders.keys())
        asks = list(depth.sell_orders.keys())
        self.data['bid'][0] = min(bids)
        self.data['ask'][0] = max(asks)
        self.data['mid'][0].append(
            (self.data['bid'][0] + self.data['ask'][0]) / 2)

        if self.current_steps > 1:
            self.data['log_return'].append(
                math.log(self.data['mid'][0][-1] / self.data['mid'][0][-2]))

    def accumulate(self):
        self.calc_prices()

    def strategy(self):
        if self.current_steps < self.vol_window + 1:
            return

        vol = np.std(self.data['log_return'][-self.vol_window:]) ** 2
        s = self.data['mid'][-1]
        q = self.state.position.get(self.products[0], 0)
        r = s - q * self.y * vol
        spread = self.y * vol + (2 / self.y) * math.log(1 + self.y / self.k)
        bid = r - spread / 2
        ask = r + spread / 2
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if bid_amount > 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], bid, bid_amount))

        if ask_amount < 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], ask, ask_amount))


class GreatWall(BaseStrategy):
    def __init__(self, product, upper, lower, limit=20):
        super().__init__()
        self.products = [product]
        self.limit = limit
        self.upper = upper + 10000
        self.lower = lower + 10000

    def strategy(self):
        q = self.state.position.get(self.products[0], 0)
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if bid_amount > 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], self.lower, bid_amount))

        if ask_amount < 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], self.upper, ask_amount))


class StatArb(BaseStrategy):
    def __init__(self, gamma, mu, thresh, limit=(600, 300)):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.limit = limit

        self.U = thresh
        self.L = -thresh

        self.products = ('COCONUTS', 'PINA_COLADAS')
        self.target_pos = (gamma * limit[1], limit[1])
        self.data = {
            'mid': [0.0, 0.0],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0],
            'signal': 0
        }

    def place_order(self, i, target_pos):
        product = self.products[i]
        pos = self.state.position.get(product, 0)

        if pos == target_pos:
            return

        else:
            order_size = target_pos - pos

        if order_size > 0:
            price = self.data['ask'][i]

        elif order_size < 0:
            price = self.data['bid'][i]

        self.orders[product].append(Order(product, price, order_size))

    def calc_prices(self):
        for i, product in enumerate(self.products):
            depth1 = self.state.order_depths[product]
            self.data['bid'][i] = min(depth1.buy_orders.keys())
            self.data['ask'][i] = max(depth1.sell_orders.keys())
            tb = max(depth1.buy_orders.keys())
            ta = min(depth1.sell_orders.keys())
            self.data['mid'][i] = (tb + ta) / 2

    def accumulate(self):
        self.calc_prices()
        self.data['signal'] = self.data['mid'][1] - \
            self.gamma * self.data['mid'][0] - self.mu

    def strategy(self):
        signal = self.data['signal']

        if signal is None:
            return

        if signal > self.U:
            for i in range(2):
                target = (self.target_pos[0], -self.target_pos[1])[i]
                self.place_order(i, target)

        elif signal < self.L:
            for i in range(2):
                target = (self.target_pos[0], -self.target_pos[1])[i]
                self.place_order(i, -target)

        elif 0.1 * self.L < signal < 0.1 * self.U:
            for i in range(2):
                self.place_order(i, 0)


class RollLS(StatArb):
    def __init__(self, window, thresh, limit=(600, 300)):
        super().__init__(0, 0, thresh, limit)
        self.window = window
        self.out = None
        self.data = {
            'mid': [[], []],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0],
            'signal': 0
        }

    def accumulate(self):
        self.calc_prices()

        if self.current_steps < self.window:
            self.data['signal'] = None
            return

        X = np.vstack((np.asarray(self.data['mid'][0][-self.window:]),
                       np.ones(self.window))).reshape(-1, 2)
        Y = np.asarray(self.data['mid'][1][-self.window:]).reshape(-1, 1)
        self.out = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()
        self.data['signal'] = self.data['mid'][1][-1] - \
            self.out[0] * self.data['mid'][0][-1] - self.out[1]
        print('{}'.format(self.data['signal']))
        self.target_pos = (self.out[0] * self.limit[1], self.limit[1])


class BerryGPT(BaseStrategy):
    def __init__(self, y, k, vol_window, limit=250):
        super().__init__()
        self.y = y
        self.k = k
        self.vol_window = vol_window
        self.limit = limit

        self.model = BerryModel()
        self.window = 24
        self.products = ['BERRIES']
        self.times = np.linspace(0, 1, 10000) / 10000

        self.create_data_dict()
        self.data['log_return'] = []

        self.target_pos = None

    def calc_prices(self):
        depth = self.state.order_depths['BERRIES']
        bids = list(depth.buy_orders.keys())
        asks = list(depth.sell_orders.keys())
        self.data['bid'][0] = min(bids)
        self.data['ask'][0] = max(asks)
        self.data['mid'][0].append(
            (self.data['bid'][0] + self.data['ask'][0]) / 2)

        if self.current_steps > 1:
            self.data['log_return'].append(
                math.log(self.data['mid'][0][-1] / self.data['mid'][0][-2]))

    def accumulate(self):
        self.calc_prices()
        window = self.window + 1

        if self.current_steps < window:
            return

        x = np.asarray(self.data['mid'][0][-window:])
        x = np.diff(x) / x[1:]
        x = (x - np.mean(x)) / np.std(x)
        x += self.times[-self.window:]
        out = self.model(x.astype(np.float16)).flatten()

        if np.isnan(out[0]):
            self.target_pos = None
            return

        self.target_pos = int(out[0] * self.limit)

    def strategy(self):
        if self.current_steps < self.window + 1 or self.target_pos is None:
            return

        vol = np.std(self.data['log_return'][-self.vol_window:]) ** 2
        s = self.data['mid'][0][-1]
        q = self.state.position.get(self.products[0], 0)
        r = s - self.target_pos * self.y * vol
        spread = self.y * vol + (2 / self.y) * math.log(1 + self.y / self.k)
        bid = r - spread / 2
        ask = r + spread / 2
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if bid_amount > 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], bid, bid_amount))

        if ask_amount < 0:
            self.orders[self.products[0]].append(
                Order(self.products[0], ask, ask_amount))


class Shipwreck(BaseStrategy):
    def __init__(self, products, feature, limit=50, stop_loss=0.2, threshold=3, window=50):
        super().__init__()
        self.products = products
        self.feature = feature
        self.feature_history = []
        self.limit = limit
        self.stop_loss = stop_loss
        self.threshold = threshold
        self.window = window
        self.entry = 0
        self.highOrLow = 0

    def calc_prices(self):
        depth = self.state.order_depths['DIVING_GEAR']
        self.data['bid'][0] = min(depth.buy_orders.keys())
        self.data['ask'][0] = max(depth.sell_orders.keys())
        tb = max(depth.buy_orders.keys())
        ta = min(depth.sell_orders.keys())
        self.data['mid'][0].append((tb + ta) / 2)

    def accumulate(self):
        self.calc_prices()

        # append new feature data to history
        self.feature_history.append(self.state.observations[self.feature])
        if len(self.feature_history) > self.window:
            self.feature_history.pop(0)

        if self.state.position.get(self.products[0], 0) > 0:
            # long position open, update high if necessary
            if self.data['mid'][0][-1] > self.highOrLow or self.highOrLow == 0:
                self.highOrLow = self.data['mid'][0][-1]
        elif self.state.position.get(self.products[0], 0) < 0:
            # short position open, update low if necessary
            if self.data['mid'][0][-1] < self.highOrLow or self.highOrLow == 0:
                self.highOrLow = self.data['mid'][0][-1]

    def strategy(self):
        # OPENING LOGIC: If no positions open, check for spikes
        # TODO: check this logic, probably wrong
        if self.state.position.get(self.products[0], 0) == 0 and len(self.feature_history) > 4:
            # compute latest and running mean diffs. exclude last because it may be a spike
            diffs = [self.feature_history[i] - self.feature_history[i - 1]
                     for i in range(1, len(self.feature_history) - 1)]
            running_mean_diff = abs(sum(diffs)) / (len(diffs) - 2)
            latest_diff = self.feature_history[-1] - self.feature_history[-2]

            # positive/negative spike => open long/short
            if latest_diff > self.threshold * running_mean_diff:
                self.place_order(i=0, target_pos=self.limit)
                self.entry = self.data['mid'][0][-1]
            elif latest_diff < -self.threshold * running_mean_diff:
                self.place_order(i=0, target_pos=-self.limit)
                self.entry = self.data['mid'][0][-1]

        # CLOSING LOGIC: If position open, check for stop loss hit
        else:
            if self.state.position.get(self.products[0], 0) > 0:
                # we are long
                lossMag = self.highOrLow - self.data['mid'][0][-1]
                lossPct = lossMag / self.highOrLow
            elif self.state.position.get(self.products[0], 0) < 0:
                # we are short
                lossMag = self.data['mid'][0][-1] - self.highOrLow
                lossPct = lossMag / self.highOrLow

            # close position if stop loss is reached
            if lossPct > self.stop_loss:
                self.place_order(i=0, target_pos=0)
                self.entry = self.highOrLow = 0


class Trader:
    def __init__(self, local=False):
        Q = np.asarray([[4.58648333e-08, 0],
                        [0, 7.08011170e-16]])
        self.strategies = [
            GreatWall('PEARLS', 1.99, -1.99),
            AvellanedaMM('BANANAS', 5, 0.01),
            Shipwreck('DIVING_GEAR', 'DOLPHIN_SIGHTINGS'),
            BerryGPT(10, 0.05, 0)
        ]
        self.logger = Logger(local)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for strategy in self.strategies:
            strategy_out = strategy(state)

            for product, orders in strategy_out.items():
                if len(orders):
                    result[product] = result.get(product, []) + orders

        self.logger.flush(state, result)
        return result
