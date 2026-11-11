"""Recompute Tables II--VII and Figure 3 from records; exit nonzero on mismatch.

Only the Python standard library is required. No models, images, network, or GPU
are accessed. Expected manuscript values are used only by the comparison step.
"""
from pathlib import Path
import argparse, csv, json, math, statistics, sys


def read(root,name):
    p=root/name
    if p.suffix=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    return json.loads(p.read_text(encoding='utf-8'))


def ious(cm):
    n=len(cm)
    return [100*cm[i][i]/u if (u:=sum(cm[i])+sum(row[i] for row in cm)-cm[i][i]) else None for i in range(n)]


def accuracy(ds,m):
    vals=ious(m['pooled_confusion'])
    return vals[1] if ds=='inria' else statistics.mean(vals[i] for i in m['selected_classes'] if vals[i] is not None)


def quantile(vals,q):
    v=sorted(vals);p=(len(v)-1)*q;i=int(p)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)


def timing(rows):
    vals=[float(x['ms']) for x in rows]
    if not vals:raise ValueError('Missing per-call timing evidence')
    return {'latency_ms':statistics.mean(vals),'n':len(vals),
            **{f'p{int(q*100)}_ms':quantile(vals,q) for q in [.5,.9,.95]},
            'memory_gib':max(int(x['peak_allocated_bytes']) for x in rows if x.get('peak_allocated_bytes'))/2**30}


def upper_hull(points):
    points=sorted(points,key=lambda p:(p[0],p[1]));dedup=[]
    for p in points:
        if dedup and abs(p[0]-dedup[-1][0])<1e-9:dedup[-1]=max(dedup[-1],p,key=lambda x:x[1])
        else:dedup.append(p)
    out=[]
    for p in dedup:
        while len(out)>=2:
            a,b=out[-2:]
            if (b[1]-a[1])/(b[0]-a[0]) <= (p[1]-b[1])/(p[0]-b[0])+1e-15:out.pop()
            else:break
        out.append(p)
    return out[:max(range(len(out)),key=lambda i:out[i][1])+1] if out else []


def calculate(root):
    calculated={};checks=[];points=[]
    def check(label,a,b,tolerance=1e-8):
        ok=math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=tolerance)
        checks.append({'scope':'raw evidence','item':label,'paper_or_record':b,'recomputed':a,'status':'PASS' if ok else 'FAIL'})
    def put(name,value):calculated[name]=value
    rows=read(root,'main_timing/routes.json');calls=read(root,'main_timing/calls.csv')
    route={}
    for row in rows:
        key=(row['dataset'],row['precision'],row['family'],row['path'])
        item=dict(row)
        if row.get('metrics'):
            item['score']=accuracy(row['dataset'],row['metrics'])
            check('/'.join(key)+'/accuracy',item['score'],row['main_pct'])
        raw=[x for x in calls if tuple(x[k] for k in ['dataset','precision','family','path'])==key]
        if raw:
            item.update(timing(raw))
            for k in ['latency_ms','n','p50_ms','p90_ms','p95_ms']:
                if row.get(k) is not None:check('/'.join(key)+'/'+k,item[k],row[k])
        elif row.get('latency_ms') is not None:
            raise ValueError('Missing raw calls for '+ '/'.join(key))
        route[key]=item
    def get(ds,family,path):return route[ds,'fp32',family,path]
    pub=read(root,'published_references.json');protocol=read(root,'protocol.json')
    ranks=read(root,'quality/seed0_validation.json')['datasets']
    def select(ds,budget):
        candidates=[get(ds,r['family'],r['path']) for r in ranks[ds]['ranked_candidates']]
        return next(x for x in candidates if x.get('latency_ms') is not None and x['latency_ms']<=budget and 'score' in x)
    fixed={x['dataset']:x['metrics'] for x in read(root,'main_timing/fixed_input.json')}
    put('eSevenDgLiteratureDeepLabPp',pub['deepglobe']['historical_deeplab'])
    put('eSevenInriaGLNetPp',pub['inria']['historical_deeplab'])
    put('eSevenDgRFiveZeroAtFiveZeroEightPp',accuracy('deepglobe',fixed['deepglobe']))
    put('eSevenDgRFiveZeroSZeroFivePp',get('deepglobe','dlv3','S_0.5')['score'])
    put('eSevenInriaRFiveZeroAtFiveZeroEightBuildingPp',accuracy('inria',fixed['inria']))
    put('eSevenInriaRFiveZeroAtFiveZeroEightMeanPp',statistics.mean(x for x in ious(fixed['inria']['pooled_confusion']) if x is not None))
    m=get('inria','dlv3','native_slide_1024_stride_768')['metrics']
    put('pInriaRBuildingPooled',accuracy('inria',m));put('pInriaRMeanPooled',statistics.mean(ious(m['pooled_confusion'])))
    image_ious=m['per_image_class_iou_pct']
    # The saved per-image IoUs are raw per-image metric observations.
    put('pInriaRBuildingImage',statistics.mean(x[1] for x in image_ious if x[1] is not None))
    put('pInriaRMeanImage',statistics.mean(statistics.mean(v for v in x if v is not None) for x in image_ious))
    partitions=read(root,'main_timing/partitions.json')
    for ds,tag in [('deepglobe','Dg'),('inria','Inria')]:
        official=get(ds,'isdnet','ISDNet_official_down_0.8');gpu=get(ds,'isdnet','ISDNet_unified_S_0.8')
        for prefix,r in [('Official',official),('Fair',gpu)]:
            put(f'isdnet{tag}FpThreeTwo{prefix}Accuracy'+('Pp' if prefix=='Official' else 'RFourPp'),r['score'])
            put(f'isdnet{tag}FpThreeTwo{prefix}LatencyRFourTime',r['latency_ms'])
        put(f'isdnet{tag}FpThreeTwoFairMemoryRFourGib',gpu['memory_gib']);put(f'p{tag}OfficialMemory',official['memory_gib'])
        put(f'isdnet{tag}OfficialReproPp',official['score'])
        put(f'isdnet{tag}Official'+('PaperPp' if ds=='deepglobe' else 'ReadmePp'),protocol['isdnet_reproduction_references'][ds])
        part=next(x for x in partitions if x['dataset']==ds and x['path']=='ISDNet_official_down_0.8' and x['precision']=='fp32' and x['opencv_threads']==1)
        put(f'eSix{tag}FpThreeTwoCpuPreprocessPct',100*sum(x['cpu_preprocessing_ms'] for x in part['rows'])/sum(x['partition_sum_ms'] for x in part['rows']))
        for ref,prefix in [(official,f'officialBudget{tag}FpThreeTwo'),(gpu,f'budget{tag}')]:
            sel=select(ds,ref['latency_ms']);suffix='Pp' if ref is official else 'RFourPp'
            put(prefix+'SelectedAccuracy'+suffix,sel['score']);put(prefix+'Delta'+suffix,sel['score']-ref['score'])
        den=pub[ds]['specialized']-pub[ds]['isdnet_published']
        put(f'p{tag}BudgetRecovery',100*(select(ds,official['latency_ms'])['score']-official['score'])/den)
        put(f'p{tag}Ftwo',pub[ds]['specialized']);put(f'p{tag}IPub',pub[ds]['isdnet_published'])
    special=read(root,'main_timing/specialized.json');rawspecial=read(root,'main_timing/specialized_calls.csv')
    for row in special:
        name=row['path'];tag={'MagNet':'MagNet','MagNet-Fast':'MagNetFast','Baseline':'MagNetBackbone','FCtL':'FCtL','FCtL-noTTA':'FCtLNoTta'}[name]
        t=timing([x for x in rawspecial if x['path']==name]);score=accuracy(row['dataset'],row['metrics'])
        check(name+'/accuracy',score,row['observed_pct'])
        for k in ['latency_ms','n','p50_ms','p90_ms','p95_ms']:check(name+'/'+k,t[k],row[k])
        put(f'eEight{tag}ObservedPp',score);put(f'eEight{tag}LatencyTime',t['latency_ms'])
        put(f'eEight{tag}MemoryGib',row['memory']['max_memory_allocated_bytes']/2**30)
        if row.get('reference_pct') is not None:put(f'eEight{tag}ReferencePp',row['reference_pct'])
        if row.get('paper_pct') is not None:put(f'eEight{tag}PaperPp',row['paper_pct'])
        put(f'eEight{tag}ValDeltaPp',select(row['dataset'],t['latency_ms'])['score']-score)
        part=next((x for x in partitions if x['path']==name),None)
        if part:
            cpu_key='cpu_preprocessing_and_aggregation_ms' if name=='FCtL-noTTA' else 'cpu_preprocessing_ms'
            put(f'eEight{tag}CpuSharePct',100*sum(x[cpu_key] for x in part['rows'])/sum(x['partition_sum_ms'] for x in part['rows']))
    put('eSevenDgMagNetFpnAtFiveZeroEightPp',calculated['eEightMagNetBackboneObservedPp'])
    seeds=read(root,'quality/seed_metrics.json')
    for ds,tag in [('deepglobe','Dg'),('inria','Inria')]:
        scores=[];recoveries=[]
        for seed,word in enumerate(['One','Two','Three']):
            s=next(x for x in seeds if x['dataset']==ds and x['seed']==seed)
            val=accuracy(ds,s['test_metrics']);recovery=100*(val-get(ds,'isdnet','ISDNet_official_down_0.8')['score'])/(pub[ds]['specialized']-pub[ds]['isdnet_published'])
            scores.append(val);recoveries.append(recovery)
            put(f'seed{tag}Md{word}Pp',val);put(f'seed{tag}C{word}Pct',recovery)
        for suffix,arr in [('Pp',scores),('Pct',recoveries)]:
            mid='' if suffix=='Pp' else 'C'
            put(f'seed{tag}{mid}Mean{suffix}',statistics.mean(arr));put(f'seed{tag}{mid}Sd{suffix}',statistics.stdev(arr))
        if ds=='inria':
            put('pInriaRecovery',recoveries[0]);put('pInriaSeedOneC',recoveries[1]);put('pInriaSeedTwoC',recoveries[2])
    for tag,ds,path in [('DgBHalf','deepglobe','S_0.5'),('DgBThird','deepglobe','S_0.333333'),('DgBThreeQuarters','deepglobe','S_0.75'),('InriaBSlide','inria','native_slide_1024_stride_768')]:
        put('p'+tag+'Ms',get(ds,'segformer_b5',path)['latency_ms'])
    put('pFixedInput',protocol['fixed_input']);put('pFairScale',protocol['fair_scale'])
    for k,v in protocol['scales'].items():put(k,v)
    for i,word in enumerate(['Zero','One','Two']):put('pSeed'+word,protocol['seed_labels'][i])
    urcm=read(root,'urur/confusion_matrices.json')['records'];urcalls=read(root,'urur/calls.csv')
    urmetrics={}
    for x in urcm:
        if x.get('confusion_matrix'):
            urmetrics[x['model'],x['split'],x['route']]=statistics.mean(v for v in ious(x['confusion_matrix']) if v is not None)
    for model,macro in [('R18','ururROneEightBestTest'),('R50','ururRFiveZeroBestTest'),('B5','ururMTest')]:put(macro,urmetrics[model,'test','scale_1_4'])
    for model,name,lat,mem in [('R50','scale_1_4','ururRFiveZeroLatencyOneDp','ururRFiveZeroPeakAllocatedGiB'),('B5','scale_1_4','ururMLatencyOneDp','ururBFivePeakAllocatedGiB'),('ISDNet','ISDNet_official','ururISDOfficialLatencyOneDp','ururISDOfficialPeakAllocatedGiB'),('ISDNet','ISDNet_gpu','ururISDGpuLatencyOneDp','ururISDGpuPeakAllocatedGiB')]:
        t=timing([x for x in urcalls if x['kind']==model and x['path']==name and x['primary'].lower()=='true']);put(lat,t['latency_ms']);put(mem,t['memory_gib'])
    for name,macro in [('DeepLabV3+ R18','ururLitDeepLabROneEight'),('ISDNet','ururISDReportedTest'),('WSDNet','ururLitWSDNet'),('Relay','ururLitRelay'),('RUE','ururLitRUE'),('CASWiT','ururLitCASWiT'),('SRMF','ururLitSRMF')]:put(macro,pub['urur'][name])
    # Figure 3: every DG/Inria data point is matched to its independent raw route.
    fig=read(root,'manuscript/figure3_points.json')['rows']
    for row in fig:
        if row.get('latency_ms') is None or row.get('main_pct') is None:continue
        key=tuple(row[k] for k in ['dataset','precision','family','path'])
        r=route[key]
        check('Figure3/'+'/'.join(key)+'/x',r['latency_ms'],row['latency_ms'])
        check('Figure3/'+'/'.join(key)+'/y',r['score'],row['main_pct'])
        if row['precision']=='fp32':points.append({'dataset':row['dataset'],'family':row['family'],'path':row['path'],'latency_ms':r['latency_ms'],'accuracy':r['score']})
    for row in read(root,'urur/urur_routes.csv'):
        if row['accuracy_latency_eligible'].lower()!='true':continue
        t=timing([x for x in urcalls if x['kind']==row['model'] and x['path']==row['route'] and x['primary'].lower()=='true'])
        score=urmetrics[row['model'],'test',row['route']]
        check('Figure3/urur/'+row['model']+'/'+row['route']+'/x',t['latency_ms'],row['latency_ms'])
        check('Figure3/urur/'+row['model']+'/'+row['route']+'/y',score,row['test_miou_8class_percent'])
        points.append({'dataset':'urur','family':row['model'],'path':row['route'],'latency_ms':t['latency_ms'],'accuracy':score})
    for row in read(root,'urur/urur_routes.csv')+read(root,'urur/urur_isdnet.csv'):
        for split in ['val','test']:
            key=(row['model'],split,row['route']);recorded=row.get(split+'_miou_8class_percent')
            if recorded:check('URUR/'+ '/'.join(key),urmetrics[key],recorded)
        raw=[x for x in urcalls if x['kind']==row['model'] and x['path']==row['route'] and x['primary'].lower()=='true']
        if raw:
            t=timing(raw)
            for key in ['latency_ms','n','p50_ms','p90_ms','p95_ms']:
                check('URUR/'+row['model']+'/'+row['route']+'/'+key,t[key],row[key])
            check('URUR/'+row['model']+'/'+row['route']+'/memory_bytes',t['memory_gib']*2**30,row['peak_allocated_bytes'])
    hulls={}
    for ds in ['deepglobe','inria','urur']:
        hulls[ds]=upper_hull([(x['latency_ms'],x['accuracy'],x['family'],x['path']) for x in points
                             if x['dataset']==ds and x['family'] in ['dlv3','segformer_b5','R50','B5']])
    expected_hull=read(root,'urur/urur_hull.csv')
    check('Figure3/URUR/hull_vertex_count',len(hulls['urur']),len(expected_hull),0)
    for actual,expected in zip(hulls['urur'],expected_hull):
        check('Figure3/URUR/hull/'+expected['model']+'/'+expected['route']+'/x',actual[0],expected['latency_ms'])
        check('Figure3/URUR/hull/'+expected['model']+'/'+expected['route']+'/y',actual[1],expected['test_miou_8class_percent'])
    calculated['_hulls']=hulls
    return calculated,checks,points


def display(macro,value,expected):
    if isinstance(value,str):return value
    if macro.endswith('Time') and abs(value)>=1000:return f'{value/1000:.3g}'
    if '.' in expected:
        decimals=len(expected.split('.')[-1]);return format(value,('+' if expected.startswith('+') else '')+f'.{decimals}f')
    return f'{value:.0f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parents[1]/'records');parser.add_argument('--format',choices=['csv','json'],default='csv');args=parser.parse_args()
    values,checks,points=calculate(args.records)
    for cell in read(args.records,'manuscript/expected_tables.json'):
        macro=cell['macro'];computed=display(macro,values[macro],cell['display']) if macro in values else 'MISSING'
        checks.append({'scope':'Table '+cell['table'],'item':macro,'paper_or_record':cell['display'],'recomputed':computed,'status':'PASS' if computed==cell['display'] else 'FAIL'})
    failures=[x for x in checks if x['status']!='PASS']
    if args.format=='json':print(json.dumps({'status':'FAIL' if failures else 'PASS','checks':checks,'figure3_points':points,'figure3_hulls':values['_hulls']},indent=2))
    else:
        w=csv.DictWriter(sys.stdout,['scope','item','paper_or_record','recomputed','status'],lineterminator='\n');w.writeheader();w.writerows(checks)
    print(f'{len(checks)} checks; {len(failures)} mismatches.',file=sys.stderr)
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
