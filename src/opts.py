import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='15-663 assign3')
    
    parser.add_argument('--i_path', type=str, default='../data/lamp/lamp_ambient.tif',
                        help='ambient image')
    parser.add_argument('--f_path', type=str, default='',
                        help='flash image')
    
    opts = parser.parse_args()

    return opts