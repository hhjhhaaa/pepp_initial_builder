from __future__ import annotations
import csv, json, math, os, random, shutil, subprocess, sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd, yaml
AMU_TO_G=1.66053906660e-24
MASS={'C':12.011,'H':1.008}
ATOM_TYPE_IDS={'PE_C':1,'PP_CH2':2,'PP_CH':3,'PP_CH3_SIDE':4,'H':5}
ATOM_TYPE_MASS={1:('PE_C',MASS['C']),2:('PP_CH2',MASS['C']),3:('PP_CH',MASS['C']),4:('PP_CH3_SIDE',MASS['C']),5:('H',MASS['H'])}
@dataclass
class Atom:
    atom_id:int; element:str; atom_type:str; polymer_type:str; chain_id:int; chain_type:str; backbone_index:int; is_backbone:bool; is_segment_center:bool; is_side_group:bool; is_hydrogen:bool; parent_segment_id:int; x:float; y:float; z:float; molecule_id:int=0; charge:float=0.0
@dataclass
class Bond: bond_id:int; atom1:int; atom2:int; bond_type:str='generic'
@dataclass
class Angle: angle_id:int; atom1:int; atom2:int; atom3:int; angle_type:str='generic'
@dataclass
class Chain:
    chain_id:int; chain_type:str; chain_length_backbone:int; atom_ids:List[int]=field(default_factory=list); backbone_atom_ids:List[int]=field(default_factory=list)
@dataclass
class SystemTopology:
    atoms:List[Atom]; bonds:List[Bond]; angles:List[Angle]; chains:List[Chain]; box:Tuple[float,float,float]; system_id:str; builder_used:str='packmol_fallback'; topology_source:str='python_all_atom_builder'; coordinate_source:str='packmol_coordinates'
    def atom_by_id(self): return {a.atom_id:a for a in self.atoms}
def load_config(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)
def project_root(c): return Path(c['paths']['root'])
def ensure_dirs(c):
    r=project_root(c)
    for k in ['systems_dir','matrix_dir','exports_dir','logs_dir']: (r/c['paths'][k]).mkdir(parents=True,exist_ok=True)
def sid(pe,pp,n,seed): return f"pepp_PE{int(round(pe*100)):02d}_PP{int(round(pp*100)):02d}_N{n}_seed{seed}"
def estimated_atoms(t,n):
    if t=='PE': return 3*n+2
    if n%2: raise ValueError('PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern')
    return int(4.5*n+2)
def matrix_rows(c,mode='matrix'):
    m=c[mode]; total=int(m['total_backbone_carbons']); rows=[]
    for pe,pp in m['compositions']:
      for n in m['chain_lengths_backbone']:
        n=int(n)
        if pp>0 and n%2: raise ValueError('PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern')
        if total%n: raise ValueError('total_backbone_carbons must be divisible by chain_length_backbone')
        tc=total//n; npe=int(round(tc*float(pe))); npp=tc-npe; pebc=npe*n; ppbc=npp*n; actual=pebc+ppbc
        for seed in m['seeds']:
          rows.append({'system_id':sid(float(pe),float(pp),n,int(seed)),'pe_fraction_target':float(pe),'pp_fraction_target':float(pp),'pe_fraction_actual':pebc/actual if actual else 0.0,'pp_fraction_actual':ppbc/actual if actual else 0.0,'chain_length_backbone':n,'seed':int(seed),'total_backbone_carbons_target':total,'total_backbone_carbons_actual':actual,'n_pe_chains':npe,'n_pp_chains':npp,'n_pe_backbone_carbons':pebc,'n_pp_backbone_carbons':ppbc,'initial_packing_density_g_cm3':float(c['density']['initial_packing_density_g_cm3']),'planned_downstream_density_scales':json.dumps(m['planned_downstream_density_scales']),'target_temperature_K_for_later_mlff':float(c['conditions_for_later_mlff']['target_temperature_K']),'target_pressure_atm_for_later_mlff':float(c['conditions_for_later_mlff']['target_pressure_atm']),'estimated_total_atoms':npe*estimated_atoms('PE',n)+npp*estimated_atoms('PP',n),'builder_status':'pending','cleanup_status':'not_run'})
    return rows
def write_matrix(c,mode):
    ensure_dirs(c); rows=matrix_rows(c,mode); out=project_root(c)/c['paths']['matrix_dir']; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'base_initial_matrix.csv',index=False); (out/'base_initial_matrix.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); return out/'base_initial_matrix.csv',out/'base_initial_matrix.json'
def discover_tools(c):
    t=c.get('tools',{}); emc=Path(t.get('known_emc_root','/home/jinhao/software/EMC')); pack=Path(t.get('known_packmol_executable','/home/jinhao/software/packmol/packmol-21.1.1/packmol')); lmp=Path(t.get('known_lammps_executable','/home/jinhao/software/lammps/build-cmake/lmp'))
    r={'python_executable':sys.executable,'python_version':sys.version.split()[0],'conda_environment':os.environ.get('CONDA_DEFAULT_ENV',''),'python_modules':{},'emc':{'root':str(emc) if emc.exists() else None,'executable':str(emc/'bin/emc_linux_x86_64') if (emc/'bin/emc_linux_x86_64').exists() else shutil.which('emc'),'emc_pl':str(emc/'scripts/emc.pl') if (emc/'scripts/emc.pl').exists() else shutil.which('emc.pl'),'emc_setup':str(emc/'scripts/emc_setup.pl') if (emc/'scripts/emc_setup.pl').exists() else shutil.which('emc_setup'),'scripts_dir':str(emc/'scripts') if (emc/'scripts').exists() else None,'examples_dir':str(emc/'examples') if (emc/'examples').exists() else None,'field_dir':str(emc/'field') if (emc/'field').exists() else None,'preferred_fields_found':[]},'packmol':{'executable':str(pack) if pack.exists() else shutil.which('packmol')},'lammps':{'executable':str(lmp) if lmp.exists() else shutil.which('lmp') or shutil.which('lammps')},'obabel':{'executable':shutil.which('obabel')}}
    for m in ['numpy','pandas','yaml','ase','MDAnalysis','rdkit','openbabel']:
      try: __import__(m); r['python_modules'][m]='FOUND'
      except Exception as e: r['python_modules'][m]=f'MISSING: {e}'
    fd=r['emc']['field_dir']
    if fd:
      fr=Path(fd)
      for name, rels in {'OPLS-AA':['opls/2024/opls-aa.prm','opls/2012/opls-aa.prm'],'PCFF':['pcff/pcff.frc'],'TraPPE':['trappe/2014/trappe-ua.prm']}.items():
        if any((fr/x).exists() for x in rels): r['emc']['preferred_fields_found'].append(name)
    return r
def write_discovery_report(c,extra=None):
    ensure_dirs(c); r=discover_tools(c); 
    if extra: r.update(extra)
    p=project_root(c)/c['paths']['logs_dir']/'emc_discovery_report.txt'; p.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8'); return p
def unit(rng):
    v=np.array([rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-1,1)],float); n=np.linalg.norm(v); return v/n if n>1e-12 else np.array([1.,0.,0.])
def walk(n,rng,step=1.54,min_nonbond=2.0):
    cs=[np.zeros(3)]; d=unit(rng)
    for i in range(1,n):
      acc=None
      for _ in range(400):
        nd=unit(rng)
        if np.dot(nd,d)<-0.35: continue
        tr=cs[-1]+step*nd
        if all(i-j<=2 or np.linalg.norm(tr-o)>=min_nonbond for j,o in enumerate(cs[:-1])): acc=tr; d=nd; break
      cs.append(acc if acc is not None else cs[-1]+step*unit(rng))
    arr=np.array(cs); arr-=arr.mean(axis=0); return [np.array(x) for x in arr]
def add_atom(atoms,element,atype,poly,chain,bi,is_back,is_side,parent,pos):
    aid=len(atoms)+1; ish=element=='H'; atoms.append(Atom(aid,element,atype,poly,chain.chain_id,chain.chain_type,bi,is_back,is_back and element=='C',is_side,ish,parent,float(pos[0]),float(pos[1]),float(pos[2]),chain.chain_id,0.0)); chain.atom_ids.append(aid); 
    if is_back: chain.backbone_atom_ids.append(aid)
    return aid
def make_chain(chain_type,n,chain_id,rng):
    if chain_type=='PP' and n%2: raise ValueError('PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern')
    atoms=[]; bonds=[]; chain=Chain(chain_id,chain_type,n); bb=walk(n,rng); ids=[]
    for i,pos in enumerate(bb):
      at='PE_C' if chain_type=='PE' else ('PP_CH2' if i%2==0 else 'PP_CH')
      aid=add_atom(atoms,'C',at,chain_type,chain,i+1,True,False,0,pos); atoms[-1].parent_segment_id=aid; ids.append(aid)
    for a,b in zip(ids[:-1],ids[1:]): bonds.append(Bond(len(bonds)+1,a,b,'C-C'))
    if chain_type=='PP':
      for i in range(1,n,2):
        tangent=(bb[i+1]-bb[i-1]) if 0<i<n-1 else (bb[i]-bb[i-1])
        tangent=tangent/(np.linalg.norm(tangent)+1e-12)
        best=None
        for _try in range(200):
          d=unit(rng); d=d-np.dot(d,tangent)*tangent; d=d/(np.linalg.norm(d)+1e-12); d*= -1 if rng.random()<0.5 else 1
          trial=bb[i]+1.54*d
          existing=[]
          for j,pos in enumerate(bb):
            if abs(j-i)>2: existing.append(pos)
          for old_atom in atoms:
            if old_atom.element=='C' and old_atom.atom_id!=ids[i]: existing.append(np.array([old_atom.x,old_atom.y,old_atom.z]))
          if all(np.linalg.norm(trial-pos)>=2.0 for pos in existing):
            best=d; break
        if best is None: best=d
        sid=add_atom(atoms,'C','PP_CH3_SIDE',chain_type,chain,i+1,False,True,ids[i],bb[i]+1.54*best); bonds.append(Bond(len(bonds)+1,ids[i],sid,'C-C'))
    neigh={a.atom_id:[] for a in atoms}
    for b in bonds: neigh[b.atom1].append(b.atom2); neigh[b.atom2].append(b.atom1)
    for c in list(atoms):
      for _ in range(4-len(neigh[c.atom_id])):
        hid=add_atom(atoms,'H','H',chain_type,chain,c.backbone_index,False,False,c.atom_id,np.array([c.x,c.y,c.z])+1.09*unit(rng)); bonds.append(Bond(len(bonds)+1,c.atom_id,hid,'C-H'))
    return atoms,bonds,chain
def offset_chain(atoms,bonds,chain,ao,bo,newcid):
    mp={a.atom_id:a.atom_id+ao for a in atoms}; ch=Chain(newcid,chain.chain_type,chain.chain_length_backbone); out=[]
    for a in atoms:
      na=Atom(mp[a.atom_id],a.element,a.atom_type,a.polymer_type,newcid,a.chain_type,a.backbone_index,a.is_backbone,a.is_segment_center,a.is_side_group,a.is_hydrogen,mp.get(a.parent_segment_id,a.parent_segment_id),a.x,a.y,a.z,newcid,0.0); out.append(na); ch.atom_ids.append(na.atom_id); 
      if na.is_backbone: ch.backbone_atom_ids.append(na.atom_id)
    return out,[Bond(i+bo+1,mp[b.atom1],mp[b.atom2],b.bond_type) for i,b in enumerate(bonds)],ch
def build_angles(bonds):
    nb={}
    for b in bonds: nb.setdefault(b.atom1,[]).append(b.atom2); nb.setdefault(b.atom2,[]).append(b.atom1)
    out=[]
    for c,ns in sorted(nb.items()):
      for a,d in combinations(sorted(ns),2): out.append(Angle(len(out)+1,a,c,d,'generic'))
    return out
def box_length_from_atoms(atoms,density): return (sum(MASS[a.element] for a in atoms)*AMU_TO_G/density*1e24)**(1/3)
def build_python_topology(row):
    rng=random.Random(int(row['seed'])); atoms=[]; bonds=[]; chains=[]; cid=1; n=int(row['chain_length_backbone'])
    for typ,count in [('PE',int(row['n_pe_chains'])),('PP',int(row['n_pp_chains']))]:
      for _ in range(count):
        ca,cb,ch=make_chain(typ,n,cid,rng); oa,ob,och=offset_chain(ca,cb,ch,len(atoms),len(bonds),cid); atoms+=oa; bonds+=ob; chains.append(och); cid+=1
    L=box_length_from_atoms(atoms,float(row['initial_packing_density_g_cm3'])); return SystemTopology(atoms,bonds,build_angles(bonds),chains,(L,L,L),row['system_id'])
def write_pdb(path,atoms,box):
  with open(path,'w',encoding='utf-8') as f:
    f.write(f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n")
    for a in atoms:
      f.write(f"HETATM{a.atom_id:5d} {a.atom_type[:4].ljust(4)} {a.chain_type[:3].rjust(3)} A{a.chain_id%10000:4d}    {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {a.element:>2s}\n")
    f.write('END\n')
def write_xyz(path,atoms,box,ext=False):
  with open(path,'w',encoding='utf-8') as f:
    f.write(f"{len(atoms)}\n")
    f.write((f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" Properties=species:S:1:pos:R:3 pbc="T T T"\n') if ext else f"box_lx_A={box[0]:.6f} box_ly_A={box[1]:.6f} box_lz_A={box[2]:.6f} pbc=true\n")
    for a in atoms: f.write(f"{a.element} {a.x:.8f} {a.y:.8f} {a.z:.8f}\n")
def write_lammps_data(path,topo):
  with open(path,'w',encoding='utf-8') as f:
    f.write(f"LAMMPS data for {topo.system_id}; atom_style full; cleanup-only zero charges\n\n{len(topo.atoms)} atoms\n{len(topo.bonds)} bonds\n{len(topo.angles)} angles\n\n{len(ATOM_TYPE_IDS)} atom types\n2 bond types\n1 angle types\n\n")
    lx,ly,lz=topo.box; f.write(f"0.0 {lx:.8f} xlo xhi\n0.0 {ly:.8f} ylo yhi\n0.0 {lz:.8f} zlo zhi\n\nMasses\n\n")
    for tid,(name,m) in ATOM_TYPE_MASS.items(): f.write(f"{tid} {m:.6f} # {name}\n")
    f.write('\nAtoms # full\n\n')
    for a in topo.atoms: f.write(f"{a.atom_id} {a.molecule_id} {ATOM_TYPE_IDS[a.atom_type]} 0.000000 {a.x:.8f} {a.y:.8f} {a.z:.8f}\n")
    f.write('\nBonds\n\n')
    for b in topo.bonds: f.write(f"{b.bond_id} {1 if b.bond_type=='C-C' else 2} {b.atom1} {b.atom2}\n")
    f.write('\nAngles\n\n')
    for a in topo.angles: f.write(f"{a.angle_id} 1 {a.atom1} {a.atom2} {a.atom3}\n")
def write_tables(sd,topo):
  pd.DataFrame([a.__dict__ for a in topo.atoms]).to_csv(sd/'atom_table.csv',index=False)
  pd.DataFrame([b.__dict__ for b in topo.bonds]).to_csv(sd/'bond_table.csv',index=False)
  pd.DataFrame([a.__dict__ for a in topo.angles]).to_csv(sd/'angle_table.csv',index=False)
  pd.DataFrame([{'chain_id':c.chain_id,'chain_type':c.chain_type,'chain_length_backbone':c.chain_length_backbone,'n_atoms':len(c.atom_ids),'n_backbone_carbons':len(c.backbone_atom_ids)} for c in topo.chains]).to_csv(sd/'chain_table.csv',index=False)
  pd.DataFrame([{'segment_id':a.atom_id,'atom_id':a.atom_id,'chain_id':a.chain_id,'polymer_type':a.polymer_type,'backbone_index':a.backbone_index,'is_segment_center':True} for a in topo.atoms if a.is_segment_center]).to_csv(sd/'segment_table.csv',index=False)
def write_chain_template(path,atoms):
  loc=[Atom(i,a.element,a.atom_type,a.polymer_type,1,a.chain_type,a.backbone_index,a.is_backbone,a.is_segment_center,a.is_side_group,a.is_hydrogen,0,a.x,a.y,a.z,1,0.0) for i,a in enumerate(atoms,1)]
  write_pdb(path,loc,(80,80,80))
def write_packmol_inputs(sd,topo,c,row):
  pd=sd/'packmol'; td=pd/'chain_templates'; td.mkdir(parents=True,exist_ok=True); amap=topo.atom_by_id(); lines=[f"tolerance {c['packmol']['tolerance_A']}",'filetype pdb',f"output {pd/'raw_packmol.pdb'}",f"seed {int(row['seed'])}",f"maxit {int(c['packmol']['maxit'])}",'']
  for ch in topo.chains:
    p=td/f"chain_{ch.chain_id:04d}_{ch.chain_type}.pdb"; write_chain_template(p,[amap[i] for i in ch.atom_ids]); lx,ly,lz=topo.box
    lines += [f"structure {p}",'  number 1',f"  inside box 0.0 0.0 0.0 {lx:.6f} {ly:.6f} {lz:.6f}",'end structure','']
  inp=pd/'packmol.inp'; inp.write_text('\n'.join(lines),encoding='utf-8'); return inp
def parse_pdb_coords(path):
  out=[]
  for line in path.read_text(errors='ignore').splitlines():
    if line.startswith(('ATOM','HETATM')): out.append((float(line[30:38]),float(line[38:46]),float(line[46:54])))
  return out
def apply_coords(topo,coords,source):
  if len(coords)!=len(topo.atoms): raise ValueError(f'Coordinate count mismatch: got {len(coords)} expected {len(topo.atoms)}')
  for a,(x,y,z) in zip(topo.atoms,coords): a.x=float(x); a.y=float(y); a.z=float(z)
  topo.coordinate_source=source
def internal_pack(topo,seed):
  rng=random.Random(seed+77); amap=topo.atom_by_id(); lx,ly,lz=topo.box
  for ch in topo.chains:
    arr=np.array([[amap[i].x,amap[i].y,amap[i].z] for i in ch.atom_ids]); shift=np.array([rng.uniform(.1*lx,.9*lx),rng.uniform(.1*ly,.9*ly),rng.uniform(.1*lz,.9*lz)])-arr.mean(axis=0)
    for i in ch.atom_ids:
      a=amap[i]; a.x=(a.x+shift[0])%lx; a.y=(a.y+shift[1])%ly; a.z=(a.z+shift[2])%lz
  topo.coordinate_source='python_internal_coordinates'
def run_packmol(sd,topo,c,row):
  inp=write_packmol_inputs(sd,topo,c,row); exe=discover_tools(c)['packmol']['executable']; log=sd/'packmol/packmol.log'
  if not exe or not Path(exe).exists(): internal_pack(topo,int(row['seed'])); log.write_text('Packmol missing; used internal coordinates\n',encoding='utf-8'); return False,'packmol_missing_internal_coordinates_used'
  try:
    with open(inp,'rb') as fin, open(log,'wb') as fout: subprocess.run([exe],stdin=fin,stdout=fout,stderr=subprocess.STDOUT,cwd=inp.parent,timeout=int(c['packmol'].get('timeout_seconds',300)),check=True)
    apply_coords(topo,parse_pdb_coords(inp.parent/'raw_packmol.pdb'),'packmol_coordinates'); return True,'packmol_success'
  except Exception as e:
    internal_pack(topo,int(row['seed'])); log.write_text(log.read_text(errors='ignore')+f'\nPackmol failed; used internal coordinates: {e}\n' if log.exists() else str(e),encoding='utf-8'); return False,f'packmol_failed_internal_coordinates_used: {e}'
def emc_attempt(c,row,sd):
  ed=sd/'emc'; ed.mkdir(parents=True,exist_ok=True); r={'attempted':True,'success':False,'timeout_seconds':int(c.get('emc',{}).get('attempt_timeout_seconds',300)),'failure_reason':'emc_adapter_v0_no_verified_all_atom_pe_pp_recipe; switched_to_packmol_fallback'}; (ed/'emc_build.log').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8'); return r
def comp_masses(topo):
  pe=sum(MASS[a.element] for a in topo.atoms if a.polymer_type=='PE'); pp=sum(MASS[a.element] for a in topo.atoms if a.polymer_type=='PP'); tot=pe+pp; return (pe/tot if tot else 0, pp/tot if tot else 0)
def write_metadata(sd,topo,row,c,emc,cleanup=None):
  pe_m,pp_m=comp_masses(topo); cleanup=cleanup or {'cleanup_performed':False,'cleanup_method':'none','cleanup_is_training_data':False,'cleanup_is_production_md':False}; kind='cleaned_initial' if cleanup.get('cleanup_performed') else 'raw_initial'
  scales=json.loads(row['planned_downstream_density_scales']) if isinstance(row['planned_downstream_density_scales'],str) else row['planned_downstream_density_scales']
  meta={'system_id':topo.system_id,'builder':{'builder_used':topo.builder_used,'representation':'all_atom_C_H','emc_attempted':bool(emc.get('attempted')),'emc_success':bool(emc.get('success')),'emc_failure_reason':emc.get('failure_reason'),'packmol_attempted':True,'packmol_success':topo.coordinate_source=='packmol_coordinates','packmol_fallback_used':True,'topology_source':topo.topology_source,'coordinate_source':topo.coordinate_source},'condition_for_later_mlff':{'target_temperature_K':c['conditions_for_later_mlff']['target_temperature_K'],'target_pressure_atm':c['conditions_for_later_mlff']['target_pressure_atm'],'planned_downstream_density_scales':scales},'density':{'density_definition':'initial_packing_density_only','initial_packing_density_g_cm3':row['initial_packing_density_g_cm3'],'not_equilibrium_density':True,'rho_eq_generated_here':False},'composition':{'pe_fraction_actual':row['pe_fraction_actual'],'pp_fraction_actual':row['pp_fraction_actual'],'pe_mass_fraction_actual':pe_m,'pp_mass_fraction_actual':pp_m},'polymer_metadata':{'chain_length_definition':'backbone_carbon_number','segment_center_definition':'backbone_carbons','pp_tacticity':'atactic_like_v0','atom_table_path':str(sd/'atom_table.csv'),'bond_table_path':str(sd/'bond_table.csv'),'segment_table_path':str(sd/'segment_table.csv'),'chain_table_path':str(sd/'chain_table.csv')},'box':{'box_lx_A':topo.box[0],'box_ly_A':topo.box[1],'box_lz_A':topo.box[2],'pbc':True},'paths':{'raw_initial_extxyz':str(sd/'raw_initial.extxyz'),'raw_initial_xyz':str(sd/'raw_initial.xyz'),'raw_initial_pdb':str(sd/'raw_initial.pdb'),'raw_initial_lammps_data':str(sd/'raw_initial.data'),'cleaned_initial_extxyz':str(sd/'cleaned_initial.extxyz'),'cleaned_initial_xyz':str(sd/'cleaned_initial.xyz'),'cleaned_initial_pdb':str(sd/'cleaned_initial.pdb'),'cleaned_initial_lammps_data':str(sd/'cleaned_initial.data'),'mlff_start_structure':str(sd/f'{kind}.data')},'cleanup':cleanup}
  (sd/'metadata.yaml').write_text(yaml.safe_dump(meta,sort_keys=False),encoding='utf-8')
def write_structure_outputs(sd,topo,prefix='raw_initial'):
  write_xyz(sd/f'{prefix}.extxyz',topo.atoms,topo.box,True); write_pdb(sd/f'{prefix}.pdb',topo.atoms,topo.box); write_lammps_data(sd/f'{prefix}.data',topo); write_xyz(sd/f'{prefix}.xyz',topo.atoms,topo.box,False)
def append_emc_report(c,row,emc,msg):
  p=write_discovery_report(c); open(p,'a',encoding='utf-8').write('\nEMC_ATTEMPT\n'+json.dumps({'system_id':row['system_id'],'emc':emc,'fallback':msg},indent=2)+'\n')
def build_system(c,row):
  sd=project_root(c)/c['paths']['systems_dir']/row['system_id']; sd.mkdir(parents=True,exist_ok=True); emc=emc_attempt(c,row,sd); topo=build_python_topology(row); ok,msg=run_packmol(sd,topo,c,row); topo.builder_used='packmol_fallback'; topo.topology_source='python_all_atom_builder'; write_structure_outputs(sd,topo); write_tables(sd,topo); write_metadata(sd,topo,row,c,emc); (sd/'system_summary.json').write_text(json.dumps({'system_id':topo.system_id,'total_atoms_actual':len(topo.atoms),'total_bonds_actual':len(topo.bonds),'total_angles_actual':len(topo.angles),'n_chains':len(topo.chains),'box_lx_A':topo.box[0],'box_ly_A':topo.box[1],'box_lz_A':topo.box[2],'pbc':True,'builder_used':topo.builder_used,'topology_source':topo.topology_source,'coordinate_source':topo.coordinate_source},indent=2),encoding='utf-8'); append_emc_report(c,row,emc,msg); return sd
def select_rows(c,tiny=False,pilot=False,max_systems=None):
  rows=matrix_rows(c,'tiny' if tiny else 'pilot' if pilot else 'matrix'); return rows[:max_systems] if max_systems is not None else rows
def build_systems(c,tiny=False,pilot=False,max_systems=None): ensure_dirs(c); return [build_system(c,r) for r in select_rows(c,tiny,pilot,max_systems)]
def load_topology_from_tables(sd):
  adf=pd.read_csv(sd/'atom_table.csv'); bdf=pd.read_csv(sd/'bond_table.csv'); agdf=pd.read_csv(sd/'angle_table.csv') if (sd/'angle_table.csv').exists() else pd.DataFrame(); meta=yaml.safe_load((sd/'metadata.yaml').read_text(encoding='utf-8'))
  atoms=[]
  for _,r in adf.iterrows():
    atoms.append(Atom(int(r.atom_id),str(r.element),str(r.atom_type),str(r.polymer_type),int(r.chain_id),str(r.chain_type),int(r.backbone_index),bool(r.is_backbone),bool(r.is_segment_center),bool(r.is_side_group),bool(r.is_hydrogen),int(r.parent_segment_id),float(r.x),float(r.y),float(r.z),int(r.molecule_id),float(r.charge)))
  bonds=[Bond(int(r.bond_id),int(r.atom1),int(r.atom2),str(r.bond_type)) for _,r in bdf.iterrows()]
  angles=[Angle(int(r.angle_id),int(r.atom1),int(r.atom2),int(r.atom3),str(r.angle_type)) for _,r in agdf.iterrows()] if not agdf.empty else build_angles(bonds)
  chains=[]
  for cid,g in adf.groupby('chain_id'):
    ch=Chain(int(cid),str(g.iloc[0]['chain_type']),int(g['backbone_index'].max())); ch.atom_ids=[int(x) for x in g['atom_id']]; ch.backbone_atom_ids=[int(x) for x in g[g['is_backbone']==True]['atom_id']]; chains.append(ch)
  box=meta['box']; return SystemTopology(atoms,bonds,angles,chains,(float(box['box_lx_A']),float(box['box_ly_A']),float(box['box_lz_A'])),meta['system_id'],meta['builder']['builder_used'],meta['builder']['topology_source'],meta['builder']['coordinate_source'])
def relation_exclusions(topo,exclude_14=True):
  ex=set(); nb={}
  for b in topo.bonds:
    ex.add(tuple(sorted((b.atom1,b.atom2)))); nb.setdefault(b.atom1,set()).add(b.atom2); nb.setdefault(b.atom2,set()).add(b.atom1)
  for c,ns in nb.items():
    for a,b in combinations(ns,2): ex.add(tuple(sorted((a,b))))
  if exclude_14:
    for a in list(nb):
      for b in nb.get(a,[]):
        for c in nb.get(b,[]):
          if c==a: continue
          for d in nb.get(c,[]):
            if d not in (a,b): ex.add(tuple(sorted((a,d))))
  return ex
def validate_system(sd,heavy_threshold=1.8):
  req=['raw_initial.xyz','raw_initial.extxyz','raw_initial.pdb','raw_initial.data','metadata.yaml','atom_table.csv','bond_table.csv','segment_table.csv']; res={'system_id':sd.name,'usable_for_mlff_start':True}; miss=[p for p in req if not (sd/p).exists()]; res['missing_files']=';'.join(miss)
  if miss: res['usable_for_mlff_start']=False; return res
  topo=load_topology_from_tables(sd); res.update({'atom_count':len(topo.atoms),'bond_count':len(topo.bonds),'segment_center_count':sum(a.is_segment_center for a in topo.atoms),'backbone_carbon_count':sum(a.is_backbone and a.element=='C' for a in topo.atoms),'box_positive':all(x>0 for x in topo.box)})
  nb={a.atom_id:[] for a in topo.atoms}; amap=topo.atom_by_id()
  for b in topo.bonds: nb[b.atom1].append(b.atom2); nb[b.atom2].append(b.atom1)
  badc=[a.atom_id for a in topo.atoms if a.element=='C' and len(nb[a.atom_id])!=4]; badh=[a.atom_id for a in topo.atoms if a.element=='H' and (len(nb[a.atom_id])!=1 or amap[nb[a.atom_id][0]].element!='C')]
  res['bad_carbon_valence_count']=len(badc); res['bad_hydrogen_count']=len(badh); pp_side=sum(a.atom_type=='PP_CH3_SIDE' for a in topo.atoms); pp_bb=sum(a.polymer_type=='PP' and a.is_backbone for a in topo.atoms); res['pp_side_methyl_count']=pp_side; res['pp_side_methyl_expected']=pp_bb//2
  ex=relation_exclusions(topo,True); heavy=[a for a in topo.atoms if a.element!='H']; overlaps=0; mind=999.0
  for i,a in enumerate(heavy):
    va=np.array([a.x,a.y,a.z])
    for b in heavy[i+1:]:
      if tuple(sorted((a.atom_id,b.atom_id))) in ex: continue
      d=float(np.linalg.norm(va-np.array([b.x,b.y,b.z]))); mind=min(mind,d)
      if d<heavy_threshold: overlaps+=1
      if overlaps>20: break
    if overlaps>20: break
  forbidden=[]
  for n in ['prod.lammpstrj','production.lammpstrj']:
    if list(sd.rglob(n)): forbidden.append(n)
  res['nonbonded_heavy_overlap_count']=overlaps; res['min_nonbonded_heavy_distance_A']=None if mind==999.0 else mind; res['forbidden_production_trajectory_files']=';'.join(forbidden); res['usable_for_mlff_start']=not badc and not badh and res['box_positive'] and not forbidden and res['segment_center_count']==res['backbone_carbon_count']; return res
def validate_systems(c,tiny=False,pilot=False):
  ensure_dirs(c); rec=[]
  for r in select_rows(c,tiny,pilot,None):
    sd=project_root(c)/c['paths']['systems_dir']/r['system_id']; rec.append(validate_system(sd,float(c['builder'].get('min_heavy_atom_distance_A',1.8))) if sd.exists() else {'system_id':r['system_id'],'usable_for_mlff_start':False,'missing_files':'system_dir'})
  out=project_root(c)/c['paths']['logs_dir']/'initial_structure_validation.csv'; pd.DataFrame(rec).to_csv(out,index=False); return out
def cleanup_inputs_text(c):
  cc=c['lammps_cleanup']; cut=cc['pair_cutoff_A']
  soft=f"""units real\natom_style full\nread_data raw_initial.data\npair_style soft {cut}\npair_coeff * * 5.0\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\nthermo {cc['thermo_stride']}\nvariable prefactor equal ramp(5.0,50.0)\nfix push all adapt 1 pair soft a * * v_prefactor\nfix int all nve/limit 0.05\nrun {cc['soft_push_steps']}\nunfix int\nunfix push\nwrite_data cleaned_initial.data\nwrite_dump all xyz cleaned_initial.xyz modify sort id\n"""
  final=f"""units real\natom_style full\nread_data cleaned_initial.data\npair_style lj/cut {cut}\npair_coeff * * 0.02 3.5\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\nthermo {cc['thermo_stride']}\nminimize 1.0e-4 1.0e-6 {cc['minimize_maxiter']} {cc['minimize_maxiter']}\nwrite_data cleaned_initial.data\nwrite_dump all xyz cleanup_check.lammpstrj modify sort id\n"""
  nvt=f"""units real\natom_style full\nread_data cleaned_initial.data\npair_style lj/cut {cut}\npair_coeff * * 0.02 3.5\nbond_style zero\nbond_coeff *\nangle_style zero\nangle_coeff *\nvelocity all create 523.0 4928459 mom yes rot yes dist gaussian\nfix int all nve/limit 0.05\nthermo {cc['thermo_stride']}\nrun {cc['short_nvt_steps']}\nunfix int\nwrite_data cleaned_initial.data\nwrite_dump all xyz cleanup_check.lammpstrj modify sort id\n"""
  return {'in.00_minimize.lmp':final,'in.01_soft_push.lmp':soft,'in.02_short_nvt_cleanup.lmp':nvt}
def write_cleanup_inputs(c,tiny=False,pilot=False,max_systems=None):
  out=[]
  for r in select_rows(c,tiny,pilot,max_systems):
    sd=project_root(c)/c['paths']['systems_dir']/r['system_id']; sd.mkdir(parents=True,exist_ok=True)
    for n,t in cleanup_inputs_text(c).items(): p=sd/n; p.write_text(t,encoding='utf-8'); out.append(p)
  return out
def run_cleanup(c,tiny=False,pilot=False,max_systems=None):
  exe=discover_tools(c)['lammps']['executable']; outs=[]
  for r in select_rows(c,tiny,pilot,max_systems):
    sd=project_root(c)/c['paths']['systems_dir']/r['system_id']; (sd/'logs').mkdir(parents=True,exist_ok=True); status={'cleanup_performed':False,'cleanup_method':'soft_then_lj','minimize_success':False,'soft_push_success':False,'short_nvt_success':False,'cleanup_is_training_data':False,'cleanup_is_production_md':False}
    if exe and Path(exe).exists() and (sd/'raw_initial.data').exists():
      for n,t in cleanup_inputs_text(c).items(): (sd/n).write_text(t,encoding='utf-8')
      try:
        with open(sd/'logs/cleanup.log','wb') as log: subprocess.run([exe,'-in','in.01_soft_push.lmp'],cwd=sd,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
        status.update({'cleanup_performed':True,'minimize_success':True,'soft_push_success':True})
        try:
          topo=load_topology_from_tables(sd)
          lines=(sd/'cleaned_initial.xyz').read_text(encoding='utf-8',errors='ignore').splitlines()[2:]
          coords=[]
          for line in lines:
            parts=line.split()
            if len(parts)>=4: coords.append((float(parts[1]),float(parts[2]),float(parts[3])))
          if len(coords)==len(topo.atoms):
            apply_coords(topo,coords,'lammps_cleanup_coordinates')
            write_xyz(sd/'cleaned_initial.extxyz',topo.atoms,topo.box,True)
            write_pdb(sd/'cleaned_initial.pdb',topo.atoms,topo.box)
            write_xyz(sd/'cleaned_initial.xyz',topo.atoms,topo.box,False)
        except Exception as e:
          status['cleaned_aux_output_warning']=str(e)
        with open(sd/'logs/cleanup.log','ab') as log:
          try: subprocess.run([exe,'-in','in.02_short_nvt_cleanup.lmp'],cwd=sd,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True); status['short_nvt_success']=True
          except Exception as e: status['short_nvt_failure_reason']=str(e)
      except Exception as e: status['failure_reason']=str(e)
    else: status['failure_reason']='lammps_missing_or_raw_initial_missing'
    status['mlff_start_structure']='cleaned_initial.data' if status.get('cleanup_performed') else 'raw_initial.data'
    try:
      mp=sd/'metadata.yaml'
      if mp.exists():
        meta=yaml.safe_load(mp.read_text(encoding='utf-8'))
        meta['cleanup']=status.copy()
        meta['cleanup']['cleanup_method']=status.get('cleanup_method','soft_then_lj')
        meta['paths']['mlff_start_structure']=str(sd/status['mlff_start_structure'])
        mp.write_text(yaml.safe_dump(meta,sort_keys=False),encoding='utf-8')
    except Exception as e:
      status['metadata_update_warning']=str(e)
    (sd/'cleanup_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8'); outs.append(sd/'cleanup_status.json')
  return outs
def export_manifest(c):
  ensure_dirs(c); rows=[]; sdir=project_root(c)/c['paths']['systems_dir']
  for sd in sorted(sdir.glob('pepp_*')):
    mp=sd/'metadata.yaml'
    if not mp.exists(): continue
    m=yaml.safe_load(mp.read_text(encoding='utf-8')); cs=json.loads((sd/'cleanup_status.json').read_text()) if (sd/'cleanup_status.json').exists() else {}; clean=bool(cs.get('cleanup_performed',m.get('cleanup',{}).get('cleanup_performed',False))); kind='cleaned_initial' if clean else 'raw_initial'; v=validate_system(sd)
    rows.append({'system_id':m['system_id'],'builder_used':m['builder']['builder_used'],'topology_source':m['builder']['topology_source'],'coordinate_source':m['builder']['coordinate_source'],'mlff_start_structure_kind':kind,'mlff_start_xyz_path':str(sd/f'{kind}.xyz'),'mlff_start_extxyz_path':str(sd/f'{kind}.extxyz'),'mlff_start_pdb_path':str(sd/f'{kind}.pdb'),'mlff_start_lammps_data_path':str(sd/f'{kind}.data'),'raw_initial_xyz_path':str(sd/'raw_initial.xyz'),'cleaned_initial_xyz_path':str(sd/'cleaned_initial.xyz'),'metadata_yaml_path':str(mp),'atom_table_path':str(sd/'atom_table.csv'),'bond_table_path':str(sd/'bond_table.csv'),'segment_table_path':str(sd/'segment_table.csv'),'chain_table_path':str(sd/'chain_table.csv'),'pe_fraction_actual':m['composition']['pe_fraction_actual'],'pp_fraction_actual':m['composition']['pp_fraction_actual'],'pe_mass_fraction_actual':m['composition']['pe_mass_fraction_actual'],'pp_mass_fraction_actual':m['composition']['pp_mass_fraction_actual'],'chain_length_backbone':m['polymer_metadata']['chain_length_definition'],'total_atoms_actual':v.get('atom_count'),'density_definition':m['density']['density_definition'],'initial_packing_density_g_cm3':m['density']['initial_packing_density_g_cm3'],'not_equilibrium_density':m['density']['not_equilibrium_density'],'planned_downstream_density_scales':json.dumps(m['condition_for_later_mlff']['planned_downstream_density_scales']),'target_temperature_K_for_later_mlff':m['condition_for_later_mlff']['target_temperature_K'],'target_pressure_atm_for_later_mlff':m['condition_for_later_mlff']['target_pressure_atm'],'pp_tacticity':m['polymer_metadata']['pp_tacticity'],'cleanup_performed':clean,'cleanup_is_training_data':False,'usable_for_mlff_start':v.get('usable_for_mlff_start',False),'box_lx_A':m['box']['box_lx_A'],'box_ly_A':m['box']['box_ly_A'],'box_lz_A':m['box']['box_lz_A'],'pbc':True})
  out=project_root(c)/c['paths']['exports_dir']; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'mlff_start_manifest.csv',index=False); (out/'mlff_start_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); return out/'mlff_start_manifest.csv',out/'mlff_start_manifest.json'
