"""Portable adapter of the measured release-candidate-2 campaign.

Preserves native AR/NAR/VAE execution; explicit paths and safer resume identity.
"""
import os
import json,time
import tempfile
from .cli import digest, safe_name, CAMPAIGN_FILES, OPTIMIZED_ENV, check_source_modules
from .verification import read_json, checked_manifest, manifest_signature, verify_song
from pathlib import Path

def write(path,data):
    path = Path(path)
    if path.is_symlink(): raise ValueError('Symlink metadata target forbidden')
    # Exclusive unpredictable temporary file: never follow a planted *.tmp link.
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(data, indent=2, allow_nan=False))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
def checkpoint(folder,stage,names,extra=None):
    write(folder/'checkpoint.json',{'stage':stage,'files':{n:digest(folder/n) for n in names},'extra':extra or {}})
def checked(folder):
    f=folder/'checkpoint.json'
    if f.is_symlink(): raise ValueError('Symlink checkpoint forbidden')
    if not f.exists():
        if any(folder.iterdir()): raise ValueError('Nonempty song directory has no checkpoint; use a fresh output directory')
        return None
    d=read_json(f)
    if d.get('stage') not in {'planned','semantic','latent','complete'}: raise ValueError('Unknown checkpoint stage')
    required={'plan.json','plan_manifest.json','prefix.npy','abc_tokens.npy'}
    if d['stage'] in {'semantic','latent','complete'}: required.update({'semantic.npy','semantic-stage.json'})
    if d['stage'] in {'latent','complete'}: required.update({'latent.npy','nar-stage.json'})
    if d['stage']=='complete': required.update({'audio.flac','result.json'})
    if not isinstance(d.get('files'),dict) or not required <= set(d['files']): raise ValueError('Incomplete checkpoint')
    if 'checkpoint.json' in d['files']: raise ValueError('Self-referential checkpoint')
    for n,h in d['files'].items():
        safe_name(n)
        p=folder/n
        if p.is_symlink() or not p.is_file() or digest(p)!=h:raise ValueError('Invalid checkpoint artifact: '+str(p))
    return d


def preflight_output(out, ids, resume):
    """Validate the whole flat output tree before any stage can overwrite files."""
    if out.is_symlink(): raise ValueError('Unsafe output directory')
    if not out.exists(): return
    if not out.is_dir(): raise ValueError('Unsafe output directory')
    entries = list(out.iterdir())
    if entries and not resume: raise FileExistsError('Fresh output required unless --resume')
    for entry in entries:
        if entry.is_symlink(): raise ValueError('Symlink output entry forbidden')
        if entry.name in CAMPAIGN_FILES and entry.is_file(): continue
        if entry.name not in ids or not entry.is_dir(): raise ValueError('Unexpected output entry: ' + entry.name)
        for artifact in entry.iterdir():
            if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_nlink != 1:
                raise ValueError('Unsafe song artifact: ' + str(artifact))
            safe_name(artifact.name)
        checked(entry)

def run(a):
    import numpy as np
    import torch
    check_source_modules(a.source)
    from yue2.pipeline import YuE2Pipeline,SymbolicPlan,SemanticResult,SongResult
    from yue2.protocol import SongRequest,GenerationConfig
    from yue2.batch import plan_batch,generate_semantic_batch
    from yue2.vae_optimized import ReusableVAEDecoder
    from yue2.storage import identity, model_identity
    check_source_modules(a.source)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Optimized generation requires a coherent ROCm GPU runtime")
    out=Path(a.output)
    preflight_output(out, {r['id'] for r in a.data}, a.resume)
    out.mkdir(parents=True,exist_ok=True)
    data=a.data;requests=[SongRequest(**r) for r in data]
    if len({r.id for r in requests})!=len(requests):raise ValueError('Duplicate ID')
    generation=GenerationConfig.from_dict({'abc':{'max_tokens':128,'min_tokens':32},'semantic':{'max_tokens':200,'min_tokens':200}}) if a.smoke else GenerationConfig()
    if a.generation_config:
        if a.smoke:raise ValueError('Do not combine smoke and explicit generation config')
        generation=GenerationConfig.from_dict(json.loads(Path(a.generation_config).read_text()))
    import yue2
    source={f.name:digest(f) for f in Path(yue2.__file__).parent.glob('*.py')}
    launcher={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    weights={'model':model_identity(a.model),'vae':model_identity(a.vae)}
    auxiliary={role:{f.name:digest(f) for f in Path(path).iterdir() if f.is_file() and (f.suffix=='.json' or f.name=='qwen.tiktoken')} for role,path in [('model',a.model),('vae',a.vae)]}
    manifest={'schema_version':1,'requests':data,'generation':generation.to_dict(),'source':source,'torch':torch.__version__,'hip':torch.version.hip,'sequential':a.sequential,'weights':weights,'auxiliary':auxiliary,'launcher':launcher,'threads':a.threads,'budget':a.budget,'smoke':a.smoke,'env':{k:os.environ[k] for k in OPTIMIZED_ENV}}
    manifest['signature']=manifest_signature(manifest)
    meta=out/'campaign.json'
    if a.resume and not meta.exists() and any(out.iterdir()): raise ValueError('Missing campaign identity; refuse unsafe resume')
    if meta.is_symlink(): raise ValueError('Symlink campaign manifest forbidden')
    if meta.exists() and checked_manifest(meta)!=manifest:raise ValueError('Resume identity mismatch')
    if not meta.exists():write(meta,manifest)
    attempt=time.perf_counter();stages={};counts={};last=[attempt]
    def status(stage,**extra):
        write(out/'status.json',{'state':'running','stage':stage,'elapsed_seconds':time.perf_counter()-attempt,**extra});print(json.dumps({'stage':stage,**extra}),flush=True)
    def token(rid,phase,value):
        counts[rid+':'+phase]=counts.get(rid+':'+phase,0)+1
        if time.perf_counter()-last[0]>15:status(phase,token_counts=counts);last[0]=time.perf_counter()
    pipe=None
    try:
        torch.set_num_threads(a.threads)
        status('loading')
        pipe=YuE2Pipeline.from_pretrained(a.model,vae=a.vae,device='cuda',memory_budget_gib=a.budget,backend='torch',generation_config=generation,local_files_only=True,progress=False)
        plans=[];pending=[];folders={r.id:out/r.id for r in requests}
        for r in requests:
            folder=folders[r.id]
            if folder.is_symlink(): raise ValueError('Symlink output directory forbidden')
            folder.mkdir(exist_ok=True);c=checked(folder)
            if c:
                plan = SymbolicPlan.load(folder)
                if plan.request.to_dict() != r.to_dict(): raise ValueError('Saved plan request disagrees with campaign')
                plans.append(plan)
            else:plans.append(None);pending.append(r)
        if pending:
            status('planning',batch=len(pending));t=time.perf_counter()
            if a.sequential:new=[pipe.plan(request=r,on_token=lambda phase,v,r=r:token(r.id,phase,v)) for r in pending]
            else:new=plan_batch(pipe,pending,on_token=token)
            stages['planning_seconds']=time.perf_counter()-t
            by_id={v.request.id:v for v in new}
            for i,r in enumerate(requests):
                if plans[i] is None:
                    plan=by_id[r.id];folder=folders[r.id];plan.save(folder)
                    checkpoint(folder,'planned',['plan.json','plan_manifest.json','prefix.npy','abc_tokens.npy']+(['score.abc'] if plan.abc is not None else []));plans[i]=plan
        semantics=[];pending=[]
        for plan in plans:
            folder=folders[plan.request.id];c=checked(folder)
            if c['stage'] in ['semantic','latent','complete']:
                s=json.loads((folder/'semantic-stage.json').read_text());semantics.append(SemanticResult(plan,np.load(folder/'semantic.npy',allow_pickle=False).tolist(),s['timing'],s['truncated']))
            else:semantics.append(None);pending.append(plan)
        if pending:
            status('semantic',batch=len(pending));t=time.perf_counter()
            if a.sequential:new=[pipe.generate_semantic(pl,on_token=lambda phase,v,pl=pl:token(pl.request.id,phase,v)) for pl in pending]
            else:new=generate_semantic_batch(pipe,pending,on_token=token)
            stages['semantic_seconds']=time.perf_counter()-t;by_id={s.plan.request.id:s for s in new}
            for i,plan in enumerate(plans):
                if semantics[i] is None:
                    sem=by_id[plan.request.id];folder=folders[plan.request.id];c=checked(folder)
                    np.save(folder/'semantic.npy',np.asarray(sem.tokens,dtype=np.int32));write(folder/'semantic-stage.json',{'timing':sem.timing,'truncated':sem.truncated})
                    checkpoint(folder,'semantic',list(c['files'])+['semantic.npy','semantic-stage.json']);semantics[i]=sem
        write(out/'ar-stages.json',stages)
        decoder=None
        for sem in semantics:
            rid=sem.plan.request.id;folder=folders[rid];c=checked(folder)
            if c['stage']=='complete':continue
            if not sem.tokens:raise ValueError('Empty semantic output: '+rid)
            if c['stage']=='latent':latent=np.load(folder/'latent.npy',allow_pickle=False);nar_time=json.loads((folder/'nar-stage.json').read_text())
            else:
                status('nar',request=rid);t=time.perf_counter();latent=pipe.synthesize(sem);torch.cuda.synchronize();nar_time={'seconds':time.perf_counter()-t,'detail':pipe.synthesis_timing}
                np.save(folder/'latent.npy',latent.astype(np.float32));write(folder/'nar-stage.json',nar_time);checkpoint(folder,'latent',list(c['files'])+['latent.npy','nar-stage.json'])
            status('vae',request=rid);t=time.perf_counter()
            if decoder is None:decoder=ReusableVAEDecoder.from_pretrained(a.vae,core_frames=1024,halo_frames=16)
            raw=decoder.decode(torch.as_tensor(latent).T.unsqueeze(0));torch.cuda.synchronize();vae_seconds=time.perf_counter()-t
            if not bool(torch.isfinite(raw).all()):raise ValueError('Nonfinite audio')
            audio=raw[0].float().clamp(-1,1).T.contiguous().numpy();rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
            if rms<1e-5:raise ValueError('Silent audio')
            cfg=pipe.effective_config(sem.plan.request);cfg['optimized']={'miopen_find_mode':'FAST','fused_snake':True,'ar_attention':'triton','ar_linear':'triton','ar_norm':'triton','ar_fuse_projections':True,'source_snapshot':'portable-release-candidate-2','batch_ar':not a.sequential}
            timing={'abc':sem.plan.timing,'semantic':sem.timing,'nar_seconds':nar_time['seconds'],'nar':nar_time['detail'],'vae_seconds':vae_seconds,'campaign_stages_shared':stages,'campaign_elapsed_at_completion_seconds':time.perf_counter()-attempt}
            result=SongResult(audio,48000,sem,latent,cfg,pipe.weights,timing,identity({'request':sem.plan.request.to_dict(),'config':cfg,'weights':pipe.weights}))
            saved=result.save_artifacts(folder)
            saved['artifacts'].pop('checkpoint.json',None)  # Mutable recovery state is not an audio artifact.
            write(folder/'result.json',saved)
            c=checked(folder);checkpoint(folder,'complete',list(c['files'])+['audio.flac','result.json'],{'rms':rms})
            print(json.dumps({'complete':rid,'audio_seconds':len(audio)/48000,'nar_seconds':nar_time['seconds'],'vae_seconds':vae_seconds,'truncated':result.truncated}),flush=True)
        pipe.close()
        # Do not publish a completed summary before every song passes the same
        # CPU acceptance gate as the public verify command, including resumes.
        for r in requests:
            verify_song(folders[r.id], a.smoke, r.to_dict())
        completed=[json.loads((folders[r.id]/'result.json').read_text()) for r in requests]
        write(out/'summary.json',{'state':'completed','attempt_seconds':time.perf_counter()-attempt,'stages':stages,'songs':[{'id':r.id,'audio_seconds':v['audio_seconds'],'truncated':v['truncated'],'timing':v['timing']} for r,v in zip(requests,completed)],'complete_song_quality_claim':False})
        write(out/'status.json',{'state':'completed','stage':'artifacts_saved','attempt_seconds':time.perf_counter()-attempt})
    except BaseException as e:
        write(out/'status.json',{'state':'failed','error':type(e).__name__+': '+str(e),'elapsed_seconds':time.perf_counter()-attempt});raise

    finally:
        if pipe is not None: pipe.close()
