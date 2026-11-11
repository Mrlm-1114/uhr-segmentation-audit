"""Three-dataset FP32 accuracy-latency comparison.

DeepGlobe/Inria use the primary timing benchmark. URUR uses its dedicated
benchmark with reported or retrained ISDNet accuracy. Each benchmark has its
own latency values and upper hull, ending at the highest-accuracy point.
Exports are exactly 7.16 x 4.0 inches, with embedded TrueType and text >=8 pt.
"""
import argparse, json, csv, hashlib, shutil, subprocess, re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np


def linear_ms_curve(front):
    """Sample physical milliseconds before matplotlib's log-axis transform."""
    if not front:
        return [], []
    x, y = [], []
    for left, right in zip(front, front[1:]):
        xs = np.geomspace(left[0], right[0], 100)
        ys = left[1] + (xs-left[0])/(right[0]-left[0])*(right[1]-left[1])
        x.extend(xs); y.extend(ys)
    return x or [front[0][0]], y or [front[0][1]]


COLORS = {'dlv3':'#2878AD','segformer_b5':'#419A61','isdnet':'#C54A43','e8':'#8964AB'}
MARKERS = {'dlv3':'o','segformer_b5':'s','fair':'^','official':'>','MagNet':'D','MagNet-Fast':'d','Baseline':'X'}
SCALE_ORDER = {'scale_1_8':0,'scale_1_6':1,'scale_1_4':2,'scale_1_3':3,'scale_1_2':4,'scale_3_4':5,'scale_1':6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def panel(ax, data, ds, routes, isd, urur_hull, retrained):
    points=[]
    if ds == 'urur':
        subset=[dict(family='dlv3' if r['model']=='R50' else 'segformer_b5',path=r['route'],
                     latency_ms=float(r['latency_ms']),main_pct=float(r['test_miou_8class_percent']))
                for r in routes if r['model'] in {'R50','B5'} and r['accuracy_latency_eligible'].lower()=='true']
        for r in isd:
            subset.append(dict(family='isdnet',path=r['route'],latency_ms=float(r['latency_ms']),
                          main_pct=float(r['test_miou_8class_percent'] if retrained else r['reported_accuracy_percent'])))
        front=[(float(r['latency_ms']),float(r['test_miou_8class_percent']),r['model'],r['route']) for r in urur_hull]
        assert all(r['interpolation_space']=='physical_ms_linear' for r in urur_hull)
        assert next(r['latency_ms'] for r in subset if r['path']=='ISDNet_gpu') > front[-1][0]
    else:
        subset=[r for r in data['rows'] if r['dataset']==ds and r['precision']=='fp32'
                and r.get('latency_ms') not in (None,'') and r['path']!='ISDNet_unified']
        front=upper_frontier(generic_points(data['rows'],ds,'fp32'))
    for family in ('dlv3','segformer_b5'):
        scaled=[r for r in subset if r['family']==family and 'slide' not in r['path']]
        scaled.sort(key=lambda r:SCALE_ORDER[r['path']] if ds=='urur' else float(r['path'][2:]))
        slide=[r for r in subset if r['family']==family and 'slide' in r['path']]
        xs=[float(r['latency_ms']) for r in scaled];ys=[float(r['main_pct']) for r in scaled]
        line,=ax.plot(xs,ys,color=COLORS[family],marker=MARKERS[family],markersize=4.2,lw=1,zorder=4)
        assert np.array_equal(line.get_xdata(),xs) and np.array_equal(line.get_ydata(),ys)
        ax.scatter([float(r['latency_ms']) for r in slide],[float(r['main_pct']) for r in slide],
                   color=COLORS[family],marker='*',s=54,edgecolors='#222222',linewidths=.4,zorder=5)
        points.extend([[family,r['path'],float(r['latency_ms']),float(r['main_pct'])] for r in scaled+slide])
    for r in subset:
        if r['family']!='isdnet':continue
        key='fair' if ('S_0.8' in r['path'] or r['path']=='ISDNet_gpu') else 'official'
        ax.scatter(float(r['latency_ms']),float(r['main_pct']),s=36,marker=MARKERS[key],
                   color=COLORS['isdnet'],edgecolors='#222222',linewidths=.5,zorder=6)
        points.append(['isdnet',r['path'],float(r['latency_ms']),float(r['main_pct'])])
    ax.plot(*linear_ms_curve(front),color='#424242',ls='--',lw=1,zorder=2)
    if ds!='urur':
        for r in data['e8']:
            if r['dataset']!=ds:continue
            method,x,y=r['path'],float(r['latency_ms']),float(r['observed_pct'])
            points.append(['e8',method,x,y])
            if ds=='deepglobe':
                ax.scatter(x,y,s=42,marker=MARKERS[method],facecolors='white',edgecolors=COLORS['e8'],lw=1,zorder=3)
            else:
                label='FCtL' if method=='FCtL' else 'FCtL w/o TTA'
                ax.annotate(f'{label}\n{x/1000:.1f} s',xy=(.995,y),xycoords=('axes fraction','data'),
                            xytext=(.95,.38 if method=='FCtL' else .19),textcoords='axes fraction',
                            ha='right',va='center',fontsize=8,color='#55436C',
                            arrowprops=dict(arrowstyle='-|>',color=COLORS['e8'],lw=.65,
                                            connectionstyle='angle,angleA=90,angleB=0,rad=2'),
                            bbox=dict(facecolor='white',edgecolor='none',pad=1.1),zorder=7)
        expected=[[r['family'],r['path'],float(r['latency_ms']),float(r['main_pct'])] for r in subset]
        expected += [['e8',r['path'],float(r['latency_ms']),float(r['observed_pct'])] for r in data['e8'] if r['dataset']==ds]
        assert sorted(points)==sorted(expected)
    specs={
        'deepglobe':((10,2000),[20,100,500,2000],(60,78),[60,65,70,75],'(a) DeepGlobe · FP32','mIoU (%)','primary'),
        'inria':((20,5000),[20,100,500,5000],(40,85),[40,50,60,70,80],'(b) Inria · FP32','Building IoU (%)','primary'),
        'urur':((50,5000),[50,200,1000,5000],(42,50),[42,44,46,48,50],'(c) URUR · FP32','mIoU (%)','R5-T')}
    xlim,ticks,ylim,yticks,title,metric,round_name=specs[ds]
    ax.set_xscale('log');ax.set_xlim(xlim);ax.set_ylim(ylim)
    ax.set_xticks(ticks,[str(x) for x in ticks]);ax.set_yticks(yticks)
    ax.tick_params(axis='both',which='major',labelsize=8,length=3,pad=3)
    ax.tick_params(axis='x',which='minor',labelbottom=False,length=2)
    ax.set_ylabel(metric,fontsize=8,labelpad=3);ax.set_xlabel('Latency (ms)',fontsize=8,labelpad=4)
    ax.set_title(title,fontsize=8.5,pad=8);ax.grid(which='major',color='#D9DEE2',lw=.45);ax.set_axisbelow(True)
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
    return dict(dataset=ds,points=points,hull=front,series_sha256=hashlib.sha256(json.dumps(sorted(points)).encode()).hexdigest(),
                hull_extrapolated=False)


def legend(fig,retrained):
    entries=[('DeepLabV3+ R50',dict(marker='o',color=COLORS['dlv3'])),
             ('SegFormer-B5',dict(marker='s',color=COLORS['segformer_b5'])),
             ('sliding window',dict(marker='*',color='#777777',markeredgecolor='#222222')),
             ('generic upper hull',dict(color='#424242',ls='--',lw=1)),
             ('ISDNet · GPU path*',dict(marker='^',color=COLORS['isdnet'])),
             ('ISDNet · official path*',dict(marker='>',color=COLORS['isdnet'])),
             ('MagNet',dict(marker='D',color=COLORS['e8'],markerfacecolor='white')),
             ('MagNet-Fast',dict(marker='d',color=COLORS['e8'],markerfacecolor='white')),
             ('MagNet backbone\n(R50-FPN, 508 px)',dict(marker='X',color=COLORS['e8'],markerfacecolor='white'))]
    handles=[Line2D([],[],**kw) if 'ls' in kw else Line2D([],[],linestyle='None',markersize=5,**kw) for _,kw in entries]
    leg=fig.legend(handles,[name for name,_ in entries],loc='lower center',bbox_to_anchor=(.51,.115),
                   ncol=3,fontsize=8,frameon=False,columnspacing=1.7,handlelength=1.6,handletextpad=.4,labelspacing=.6)
    origin='retrained accuracy' if retrained else 'reported accuracy (45.8%)'
    fig.text(.5,.077,f'*URUR: {origin}; DeepGlobe/Inria: reproduced accuracy.',ha='center',va='center',fontsize=8)
    fig.text(.5,.033,'GPU-path scales: ×0.8 (DeepGlobe/Inria), ×1.0 (URUR).',ha='center',va='center',fontsize=8)
    return leg


def check_layout(fig,axes,leg):
    fig.canvas.draw();r=fig.canvas.get_renderer();items=[];outside=[]
    minor=[t for ax in axes for t in ax.get_xminorticklabels()]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text() or t in minor:continue
        b=t.get_window_extent(r);items.append(dict(text=t.get_text(),size_pt=t.get_fontsize(),bbox=list(b.extents)))
        if not(b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1):outside.append(t.get_text())
    minimum=min(t['size_pt'] for t in items);assert minimum>=8 and not outside,(minimum,outside)
    gap=min(ax.xaxis.label.get_window_extent(r).y0 for ax in axes)-leg.get_window_extent(r).y1
    assert gap>3,gap
    for ax in axes:
        labels=[t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_visible()]
        assert not any(a.overlaps(b) for a,b in zip(labels,labels[1:])),labels
    return dict(minimum_text_pt=minimum,outside_canvas=outside,legend_xlabel_gap_px=gap,text=items)


def check_pdf(pdf):
    fonts=subprocess.run([shutil.which('pdffonts'),str(pdf)],capture_output=True,text=True,check=True).stdout
    info=subprocess.run([shutil.which('pdfinfo'),str(pdf)],capture_output=True,text=True,check=True).stdout
    fl=[x for x in fonts.splitlines()[2:] if x.strip()]
    assert fl and all('Type 3' not in x and re.search(r'\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',x) for x in fl),fonts
    m=re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts',info);assert m,info
    w,h=map(float,m.groups());assert abs(w/72-7.16)<1e-5 and h/72<=4+1e-8
    return dict(width_inches=w/72,height_inches=h/72,embedded=True,type3=False,pdffonts=fonts,pdfinfo=info)


def fval(v):
    if v is None or v == "" or str(v).lower() in {"none", "nan", "na"}:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def generic_points(rows, dataset, precision):
    pts = []
    for r in rows:
        if r.get("dataset") != dataset or r.get("precision") != precision:
            continue
        if r.get("family") not in {"dlv3", "segformer_b5"}:
            continue
        if r.get("quality_status", "complete") != "complete":
            continue
        x, y = fval(r.get("latency_ms")), fval(r.get("main_pct"))
        if x is not None and y is not None:
            pts.append((x, y, r.get("family", ""), r.get("path", "")))
    pts.sort(key=lambda z: (z[0], z[1]))
    # Collapse exact/near-exact duplicate latencies using the best accuracy.
    dedup = []
    for pt in pts:
        if dedup and abs(dedup[-1][0] - pt[0]) < 1e-9:
            if pt[1] > dedup[-1][1]:
                dedup[-1] = pt
        else:
            dedup.append(pt)
    return dedup

def upper_frontier(points):
    """Upper convex hull used for the descriptive accuracy/latency curve.

    The analysis uses linear interpolation between the
    supporting points of this hull (for example, DG DLV3 S_0.25 to the B5
    S_0.333333 point).  A simple running maximum would retain interior points
    and would not reproduce the corresponding interpolation endpoints.
    """
    front = []
    for pt in points:
        while len(front) >= 2:
            a, b = front[-2], front[-1]
            s1 = (b[1] - a[1]) / (b[0] - a[0])
            s2 = (pt[1] - b[1]) / (pt[0] - b[0])
            # Upper concave envelope: slopes must not increase.
            if s1 <= s2 + 1e-15:
                front.pop()
            else:
                break
        front.append(pt)
    # The figure truncates the descriptive curve at its highest
    # generic point; lower-accuracy points to its right (for example a
    # sliding route) remain isolated markers and are not part of the hull.
    if front:
        imax = max(range(len(front)), key=lambda i: front[i][1])
        front = front[: imax + 1]
    return front


if __name__ == '__main__':
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from scripts.reproduce_tables import calculate,read,main as verify
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/figures'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    values,checks,points=calculate(root/'records')
    if any(x['status']!='PASS' for x in checks):raise SystemExit('Record mismatch; plot stopped')
    data={'rows':read(root/'records','main_timing/routes.json'),'e8':read(root/'records','main_timing/specialized.json')}
    routes=csv_rows(root/'records/urur/urur_routes.csv');isd=csv_rows(root/'records/urur/urur_isdnet.csv')
    hull=csv_rows(root/'records/urur/urur_hull.csv')
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':8,
        'font.family':'Arial','axes.linewidth':.7,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(7.16,4.0));fig.subplots_adjust(left=.071,right=.960,bottom=.405,top=.89,wspace=.40)
    for ax,ds in zip(axes,['deepglobe','inria','urur']):panel(ax,data,ds,routes,isd,hull,False)
    leg=legend(fig,False);check_layout(fig,axes,leg)
    for ext in ['pdf','svg','png']:fig.savefig(args.output/('figure3.'+ext),**({'dpi':300} if ext=='png' else {}))
    plt.close(fig)
