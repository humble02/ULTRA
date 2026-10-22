"""Exercise extracted production reset methods on sampled real reference states (CPU only)."""
import argparse, ast, copy, json, sys, math, subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ultra'))
import isaacgym
import torch
from utils import torch_utils
from isaacgym.torch_utils import quat_mul
import yaml
import trimesh
OUT=ROOT/"output/reproduction/initialization_verification"
BASELINE_REF="9c84f6a"
SRC=ROOT/'ultra/env/tasks/ultra_g1_distill_obj_v3rl.py'
ns={'torch':torch,'torch_utils':torch_utils,'quat_mul':quat_mul,'math':math}
helper=ast.parse((ROOT/'ultra/env/tasks/ultra_g1_retarget.py').read_text())
nodes=[n for n in helper.body if (isinstance(n,ast.FunctionDef) and n.name in ('_randn_like','_perturb_quat')) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_DEFAULT_NOISE_STD' for t in n.targets))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_noise_helpers','exec'),ns)
def get_method(name, candidate=False):
    source = (SRC.read_text() if candidate else subprocess.check_output(
        ["git", "show", BASELINE_REF + ":ultra/env/tasks/ultra_g1_distill_obj_v3rl.py"],
        cwd=ROOT, text=True))
    tree=ast.parse(source)
    node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
    node.body=[n for n in node.body if not isinstance(n,ast.ImportFrom)]
    local=dict(ns);exec(compile(ast.Module(body=[node],type_ignores=[]),str(SRC),'exec'),local)
    return local[name]

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((ROOT/'output/reproduction/student_10000_benchmark/manifest.json').read_text())
    frames=json.loads((ROOT/'ultra/data/assets/objects/point_pca_frames.json').read_text())['frames']
    registry={x['asset']:torch.tensor(x['corners']) for x in frames}
    rows=[];corners=[];is_start=[]
    # 256 samples/motion, 10% start frame, remainder uniformly sampled.
    # This diagnostic is stratified over the eval cohort; not a measured training reset rate.
    for motion in manifest['motions']:
        ref=torch.load(ROOT/motion['motion'],map_location='cpu')
        ts=torch.randint(len(ref),(256,)); ts[:26]=0
        rows.append(ref[ts,:84]); corners.append(registry[motion['asset_key']].expand(256,-1,-1));is_start.append(ts==0)
    ref=torch.cat(rows);corners=torch.cat(corners);is_start=torch.cat(is_start);n=len(ref)
    cfg=yaml.safe_load((ROOT/'output/reproduction/finetune_fixed_6000_20261022/environment.yaml').read_text())['env']['rl_finetune']
    def task():
        return SimpleNamespace(prior_mask=torch.ones(n,dtype=torch.bool),progress_buf=torch.zeros(n,dtype=torch.long),
            rl_init_noise_std={k:cfg['init_'+k+'_noise'] for k in ('root_pos','root_rot','target_pos','target_rot')},
            dof_limits_lower=torch.full((29,),-10.),dof_limits_upper=torch.full((29,),10.),
            _humanoid_root_states=torch.zeros(n,13),_dof_pos=torch.zeros(n,29),_dof_vel=torch.zeros(n,29),
            hoi_refs=ref[:,None,None,:].clone(), data_id=torch.arange(n),ref_index=torch.zeros(n,dtype=torch.long),
            _target_states=torch.zeros(n,13),object_id=torch.arange(n),object_corners=corners)
    def bottom(pos,rot):
        qs=rot[:,None,:].expand(-1,8,-1).reshape(-1,4)
        rc=torch_utils.quat_rotate(qs,corners.reshape(-1,3)).reshape(n,8,3)
        return pos[:,2]+rc[:,:,2].amin(1)
    report={'sample_count':n,'seed':42,'motions':20,'sample_design':'256 samples per benchmark motion, 26 forced initial frames and 230 uniformly sampled frames; CPU production-reset replay, not live-training prevalence or policy performance',
            'geometry':'world-space minimum of training asset bounding-box corners; conservative floor clearance bound',
            'reference_box_bounds_below_floor_1cm':int((bottom(ref[:,71:74],ref[:,74:78])<-.01).sum())}
    def mesh_bottom(pos, rot):
        clear=[]
        for index, motion in enumerate(manifest['motions']):
            name=motion['asset_key']
            path=ROOT/'ultra/data/assets/objects/diverse'/name/(name+'.obj')
            mesh=trimesh.load(path, process=False, force='mesh')
            # Convex hull preserves the minimum of every linear projection of the raw mesh.
            vertices=torch.tensor(mesh.convex_hull.vertices.copy(),dtype=pos.dtype)
            count=len(vertices); sl=slice(index*256,(index+1)*256)
            qs=rot[sl,None,:].expand(-1,count,-1).reshape(-1,4)
            vv=vertices[None,:,:].expand(256,-1,-1).reshape(-1,3)
            minimum=torch_utils.quat_rotate(qs,vv).reshape(256,count,3)[:,:,2].amin(1)
            clear.append(pos[sl,2]+minimum)
        return torch.cat(clear)
    report['reference_mesh_below_floor_1cm']=int((mesh_bottom(ref[:,71:74],ref[:,74:78])<-.01).sum())
    old=None
    for label,candidate in [('before_fix',False),('after_fix',True)]:
        t=task(); torch.manual_seed(1234)
        get_method('_set_env_state',candidate)(t,torch.arange(n),ref[:,:3],ref[:,3:7],ref[:,13:42],ref[:,7:10],ref[:,10:13],ref[:,42:71])
        get_method('_reset_target',candidate)(t,torch.arange(n))
        zerr=t._humanoid_root_states[:,2]-(ref[:,2]+.01)
        clearance=bottom(t._target_states[:,:3],t._target_states[:,3:7])
        relative_error=(t._target_states[:,:3]-t._humanoid_root_states[:,:3])-(ref[:,71:74]-ref[:,:3])
        report[label]={'root_z_error_std_m':float(zerr.std()),'root_z_error_abs_max_m':float(zerr.abs().max()),
            'box_bounds_below_floor_1cm':int((clearance<-.01).sum()),'box_bounds_below_floor_5cm':int((clearance<-.05).sum()),
            'box_min_clearance_m':float(clearance.min()),'initial_frame_samples':int(is_start.sum()),
            'initial_frame_box_bounds_below_floor_1cm':int(((clearance<-.01)&is_start).sum()),
            'relative_robot_object_displacement_error_mean_m':float(relative_error.norm(dim=-1).mean())}
        mesh_clearance=mesh_bottom(t._target_states[:,:3],t._target_states[:,3:7])
        report[label].update(box_mesh_below_floor_1cm=int((mesh_clearance<-.01).sum()),
                             box_mesh_below_floor_5cm=int((mesh_clearance<-.05).sum()),
                             box_mesh_min_clearance_m=float(mesh_clearance.min()),
                             initial_frame_box_mesh_below_floor_1cm=int(((mesh_clearance<-.01)&is_start).sum()))
        if candidate:
            assert mesh_clearance.min() >= .01-1e-6
            assert zerr.abs().max()<1e-6
            assert clearance.min()>=.01-1e-6
            assert torch.equal(old._humanoid_root_states[:,:2],t._humanoid_root_states[:,:2])
            assert torch.equal(old._target_states[:,:2],t._target_states[:,:2])
            assert torch.equal(old._target_states[:,3:],t._target_states[:,3:])
        else: old=t
    # Chained advanced indexing also fails with non-contiguous reset IDs; check actual function.
    t=task();t.prior_mask[::2]=False
    ids=torch.tensor([7,2,15,6,23,10])
    get_method('_set_env_state',True)(t,ids,ref[ids,:3],ref[ids,3:7],ref[ids,13:42],ref[ids,7:10],ref[ids,10:13],ref[ids,42:71])
    assert torch.allclose(t._humanoid_root_states[ids[::2],2],ref[ids[::2],2]+.01)
    report['candidate_noncontiguous_reset_ids_pass']=True
    # Actual visibility-gated reward: no difference for hidden root targets.
    assert report['before_fix']['root_z_error_std_m'] > .1
    assert report['before_fix']['box_mesh_below_floor_1cm'] > 0
    report['baseline_commit']=BASELINE_REF
    report['passed']=True
    (OUT/'initialization_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--baseline-ref',default=BASELINE_REF)
    args=parser.parse_args();OUT=args.output_dir.resolve();BASELINE_REF=args.baseline_ref
    OUT.mkdir(parents=True,exist_ok=True)
    main()
