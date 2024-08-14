import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import trimesh
from os import path as osp
import os
from tqdm import tqdm


from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import colors, imagearray2file


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas=16):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt','')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

    def __len__(self):
       return len(self.ds['trans'])

    def __getitem__(self, idx):
        data =  {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]

        return data
"""
work_dir = 'data'

num_betas = 16 # number of body parameters
testsplit_dir = os.path.join(work_dir, 'stage_III', 'test')

ds = AMASS_DS(dataset_dir=testsplit_dir, num_betas=num_betas)
print('Test split has %d datapoints.'%len(ds))

batch_size = 5
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


support_dir = 'amass/support_data/'
bm_fname = osp.join(support_dir, 'body_models/smplh/male/model.npz')
device = torch.device('cpu')
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(device)
faces = c2c(bm.f)

bdata = next(iter(dataloader))
body_v = bm.forward(**{k:v.to(device) for k,v in bdata.items() if k in ['pose_body', 'betas']}).v

for cId in range(0, batch_size):
    orig_body_mesh = trimesh.Trimesh(vertices=c2c(body_v[cId]), faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
    orig_body_mesh.export(f'test{cId}.obj')"""
    
"""support_dir = 'amass/support_data/'
device = torch.device('cpu')

amass_npz_fname = osp.join(support_dir, 'github_data/dmpl_sample.npz')
bdata = np.load(amass_npz_fname)
subject_gender = bdata['gender'][2:-1]

bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))

num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
faces = c2c(bm.f)"""

DATASET = 'EKUT'
FRAMES_PER_SECOND = 2
DEVICE = 'cpu'
NPY_PATH = os.path.join('amass', 'support_data', 'amass_npz', DATASET)
OUT_PATH = os.path.join('.', DATASET)
SUPPORT_ROOT = os.path.join('amass', 'support_data')
SMPLH_ROOT = os.path.join(SUPPORT_ROOT, 'body_models', 'smplh')
DMPLS_ROOT = os.path.join(SUPPORT_ROOT, 'body_models', 'dmpls')

def generate_meshes(npy_root, output_root, smplh_root, dmpls_root, device='cpu', frames_per_second=2):
    os.makedirs(output_root, exist_ok=True)
    
    num_betas = 16 
    num_dmpls = 8
    bm_fname = osp.join(smplh_root, 'male', 'model.npz')
    dmpl_fname = osp.join(dmpls_root, 'male', 'model.npz')
    male_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    male_faces = c2c(male_model.f)
    bm_fname = osp.join(smplh_root, 'female', 'model.npz')
    dmpl_fname = osp.join(dmpls_root, 'female', 'model.npz')
    female_model = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    female_faces = c2c(female_model.f)

    npz_fnames = glob.glob(os.path.join(npy_root, '*/*_poses.npz'))
    for sequence, npz_fname in tqdm(enumerate(npz_fnames), total=len(npz_fnames)):
        cdata = np.load(npz_fname)
        gender = str(cdata['gender'])
        if gender[0] == 'b':
            gender = gender[2:-1]
            
        if gender == 'female':
            model = female_model
            faces = female_faces
        elif gender == 'male':
            model = male_model
            faces = male_faces
        else:
            print(gender)
            assert gender in ('female', 'male')
       
        body_trans_root = predict(model, cdata, num_dmpls, num_betas, device)
        if body_trans_root is None:
            continue
        frames = body_trans_root.v.size(0)

        step = int(cdata['mocap_framerate'] / frames_per_second)     
        if step == 0:
            step = 1
        start = step // 2
        for i in range(start, frames, step):
            frame = (i - start) // step
            #joints = c2c(body_trans_root.Jtr[frame])
            joints_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.Jtr[i]))
            joints_out_path = os.path.join(output_root, f'seq{sequence + 1}-body{frame}.pose.obj')
            joints_mesh.export(joints_out_path)
            
            body_mesh = trimesh.Trimesh(vertices=c2c(body_trans_root.v[i]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
            body_out_path = os.path.join(output_root, f'seq{sequence + 1}-body{frame}.shape.obj')
            body_mesh.export(body_out_path)


def predict(model, cdata, num_dmpls, num_betas, device):
    N = len(cdata['poses'])
    start = int(N * 0.1)
    end = int(N * 0.9)

    if (end - start) <= 0:
        start = int(N * 0.05)
        end = int(N * 0.95)
    if (end - start) <= 0:
        return None
    poses = cdata['poses'][start : end].astype(np.float32)
    dmpls = cdata['dmpls'][start : end, :num_dmpls].astype(np.float32)
    trans = cdata['trans'][start : end].astype(np.float32)
    betas = np.repeat(cdata['betas'][np.newaxis, :num_betas].astype(np.float32), repeats=end - start, axis=0)
    root_orient = poses[:, :3]
    pose_body = poses[:, 3:66]
    pose_hand = poses[:, 66:]
    body_parms = {
            'root_orient': torch.Tensor(root_orient).to(device), # controls the global root orientation
            'pose_body': torch.Tensor(pose_body).to(device), # controls the body
            'pose_hand': torch.Tensor(pose_hand).to(device), # controls the finger articulation
            'trans': torch.Tensor(trans).to(device), # controls the global body position
            'betas': torch.Tensor(betas).to(device), # controls the body shape. Body shape is static
            'dmpls': torch.Tensor(dmpls).to(device) # controls soft tissue dynamics
            }
    return model(**body_parms)
            
if __name__ == '__main__':
    generate_meshes(NPY_PATH, OUT_PATH, SMPLH_ROOT, DMPLS_ROOT, DEVICE, FRAMES_PER_SECOND)
