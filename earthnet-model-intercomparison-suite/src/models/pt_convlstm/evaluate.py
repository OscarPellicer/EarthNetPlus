import sys
import os
from pathlib import Path
from argparse import ArgumentParser
import yaml
import warnings

# go to the project root directory and add it to path
proj_root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(proj_root_dir))
os.chdir(proj_root_dir)
#print(f"cwd = {Path.cwd()}")

calc_dir = Path.cwd().parent.parent.parent.parent / 'earthnet-toolkit'
sys.path.append(str(calc_dir))
print(f'File: {__file__}; calc_dir: {calc_dir}')
from earthnet.parallel_score import EarthNetScore

__TRACKS__ = {
        "iid": "iid_test_split/",
        "ood": "ood_test_split/",
        "ex": "extreme_test_split/",
        "sea": "seasonal_test_split/",
    }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--setting', type=str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('--track', type=str, default='all', metavar='iid|ood|ex|sea',
                        help='which track to test: either iid, ood, ex or sea')

    #If version is provided, the rest of the arguments are not needed
    parser.add_argument('--version', type=int, default=None, metavar='0',
                        help='Version of the experiment (int) assigned by the logger')

    #Rest of the arguments
    parser.add_argument('--pred_dir', type=str, default=None, metavar='path/to/predictions/directory/',
                        help='Path where to save predictions')
    parser.add_argument('--dataset_dir', type=str, default=None, metavar='path/to/dataset/directory/',
                        help='Path where the original unprocessed dataset is located')

    args = parser.parse_args()

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    if args.track == 'all':
        tracks= ['iid','ood','ex','sea']
    else:
        tracks= [args.track]

    for track in tracks:
        print('Testing track:', track)
        #If version is provided 
        if args.version is not None or args.pred_dir is None:
            if not f'version_{args.version}' in args.setting:
                warnings.warn(f'You should use the settings.yaml file contained within the version_{args.version} directory')
            setting_dict["Task"]["pred_dir"]= os.path.join(setting_dict["Logger"]["save_dir"], 
                        setting_dict["Logger"]["name"], f'version_{args.version}', 'predictions', __TRACKS__[track])
        else:
            setting_dict["Task"]["pred_dir"] = args.pred_dir
        setting_dict["Data"]["test_track"] = track

        #Output dir
        save_dir = os.path.join(setting_dict["Task"]["pred_dir"].replace('predictions', 'scores'))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        data_output_file = os.path.join(save_dir, 'data.json')
        ens_output_file = os.path.join(save_dir, 'model.json')
        
        #Predicted / target datacubes dir
        if args.dataset_dir is None:
            targ_dir= os.path.join(setting_dict["dataset_dir"], __TRACKS__[track], 'target')
        else:
            targ_dir= args.dataset_dir
        pred_dir = os.path.join(setting_dict["Task"]["pred_dir"], 'target')

        EarthNetScore.get_ENS(pred_dir, targ_dir, n_workers=8, data_output_file=data_output_file,
                            ens_output_file=ens_output_file)
