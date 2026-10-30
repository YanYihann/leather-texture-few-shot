"""Run after other encoders stop. Fixed CPU cost protocol; decoded input PIL images."""
from run_computations import *
import statistics, winreg
from threadpoolctl import threadpool_limits,threadpool_info
def peak_memory():
 class PMC(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
 pm=PMC();pm.cb=ctypes.sizeof(pm)
 ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
 ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
 handle=ctypes.windll.kernel32.GetCurrentProcess();ok=ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(pm),pm.cb);assert ok and pm.PeakWorkingSetSize>0
 return pm.PeakWorkingSetSize
def elapsed(fn):
 start=time.perf_counter_ns();v=fn();return v,(time.perf_counter_ns()-start)/1e6
def run(method):
 torch.set_num_threads(3);torch.set_num_interop_threads(1);limits=threadpool_limits(limits=3)
 raw=[imread(p,False) for p in paths];rawrefs=[imread(p,False) for p in rp];model=load(method) if not method.endswith('histogram') else None;p=refs(method).astype(np.float32)
 def prep(im):return norm(resize(im)).unsqueeze(0) if model is not None else resize(im)
 def feature(x):
  if model is None:return hist([x],method)
  with torch.inference_mode():return torch.nn.functional.normalize(model(x),dim=1).numpy()
 def match(f):
  score=f@p.T
  return np.argsort(-score,axis=1,kind='stable')[:,:5]
 def query(im):return match(feature(prep(im)))
 cold,coldms=elapsed(lambda:query(raw[0]))
 for i in range(10):query(raw[i])
 records=[]
 for i in range(100):
  x,pre=elapsed(lambda:prep(raw[i]));f,enc=elapsed(lambda:feature(x));s,matching_ms=elapsed(lambda:match(f));s,total=elapsed(lambda:query(raw[i]));records.append(dict(method=method,repeat=i,query=paths[i].parent.name,preprocess_ms=pre,encode_ms=enc,match_ms=matching_ms,total_ms=total))
 write(D/f'cost_query_{method}.csv',records)
 aug=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.4,1.)),transforms.RandomHorizontalFlip(.5),transforms.RandomVerticalFlip(.5),transforms.ColorJitter(.2,.2,.2,0.)])
 registration=[]
 for n in [1,5,30]:
  for repeat in range(3):
   start=time.perf_counter();feats=[];batch=[]
   for c,im in zip(CLASSES,rawrefs):
    for v in range(n):
     if n==1:x=resize(im)
     else:
      sd=int.from_bytes(hashlib.sha256(f'42|{c}|{v}'.encode()).digest()[:4],'little');torch.manual_seed(sd);x=aug(im)
     batch.append(x)
     if len(batch)==16:
      feats.append(hist(batch,method) if model is None else encode(model,batch));batch=[]
   if batch:feats.append(hist(batch,method) if model is None else encode(model,batch))
   pp=unit(np.concatenate(feats).reshape(203,n,-1).mean(1));sec=time.perf_counter()-start
   registration.append(dict(method=method,views=n,repeat=repeat,encodings=203*n,seconds=sec,prototype_bytes=pp.nbytes,peak_working_set_bytes=peak_memory(),reference_mode='deterministic_center' if n==1 else 'seed42_random_nested'))
   write(D/f'cost_registration_{method}.csv',registration);print('C4',method,n,repeat,round(sec,3),flush=True)
 with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
 metadata=dict(method=method,cpu=cpu,logical_cpus=os.cpu_count(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),threadpools=threadpool_info(),batch_query=1,batch_registration=16,python=sys.version,torch=torch.__version__,platform=platform.platform(),warmup=10,repeats=100,registration_repeats=3,post_load_first_query_ms=coldms,peak_process_working_set_bytes=peak_memory(),scope='PIL images decoded before timing; preprocessing + encoding + full203 matrix and stable Top5 ranking; no disk IO or weight load; first post-load call separate; peak is whole-process high-water mark including predecoded benchmark image collection',histogram_averaging='cost-only optional descriptor pools; C1 uses deterministic histograms only')
 if model is not None:
  weight=Path.home()/'.cache/torch/hub/checkpoints'/('resnet50-11ad3fa6.pth' if method=='resnet50' else 'dinov2_vits14_pretrain.pth');metadata['weight_file_bytes']=weight.stat().st_size;metadata['weight_sha256']=sha(weight)
 else:metadata['weight_file_bytes']=0
 (O/'audit'/f'cost_{method}.json').write_text(json.dumps(metadata,indent=2))
 print('COMPLETE C4',method,flush=True)
if __name__=='__main__':run(sys.argv[1])
