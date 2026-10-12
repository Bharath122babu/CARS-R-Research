"""Generate the README's original vector diagrams and measured charts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
OUT.mkdir(parents=True, exist_ok=True)
BG, CARD, FG, MUTED = '#081321', '#13253a', '#edf6ff', '#9fb4ce'
TEAL, PURPLE, GOLD, RED = '#38dbc2', '#ae91ff', '#ffca70', '#ff879a'
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'text.color':FG})

def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set(xlim=(0,100), ylim=(0,100)); ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    return fig, ax

def text(ax, x, y, label, size=14, color=FG, **kw):
    ax.text(x,y,label,fontsize=size,color=color,va='center',**kw)

def card(ax, x,y,w,h,title,subtitle='',color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=2',
                               facecolor=CARD,edgecolor=color,linewidth=1.5))
    text(ax,x+w/2,y+h*.65,title,15,color,ha='center',weight='bold')
    if subtitle: text(ax,x+w/2,y+h*.28,subtitle,10,MUTED,ha='center')

def arrow(ax, a, b, color=MUTED, rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,linewidth=1.6,
                                color=color,connectionstyle=f'arc3,rad={rad}'))

def save(fig, name):
    for ext in ('svg','png'):
        path = OUT/f'{name}.{ext}'
        fig.savefig(path,dpi=150,facecolor=BG)
        if ext == 'svg':
            path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)

fig, ax = canvas(14,4.7)
text(ax,5,83,'CARS-R',44,weight='bold')
text(ax,5,64,'Cayley transport + associative reads + ridge correction',17,MUTED)
for x,title,sub,color in [(5,'2.04M','laptop model parameters',PURPLE),
                         (36,'3.78x','vs original CARS execution',TEAL),
                         (67,'Rank 2','block evidence correction',GOLD)]:
    card(ax,x,20,28,27,title,sub,color)
text(ax,5,8,'Completed epoch: Wiki NLL 1.38768. All four models reach 100% single-object / 0% two-object ownership.',10,MUTED)
save(fig,'hero')

fig, ax = canvas(14,7.5)
text(ax,5,93,'The laptop backbone',26,weight='bold')
text(ax,5,85,'Byte tokens -> 192-wide hidden states -> three recurrent blocks -> one causal MLA anchor',13,MUTED)
card(ax,4,59,15,16,'Byte input','259-token vocabulary',GOLD)
card(ax,23,59,16,16,'CARS 1','4 heads / 32 x 32',TEAL)
card(ax,43,59,16,16,'CARS 2','4 heads / 32 x 32',TEAL)
card(ax,63,59,16,16,'CARS 3','4 heads / 32 x 32',TEAL)
card(ax,83,59,13,16,'MLA','causal attention',PURPLE)
for a,b in [(19,23),(39,43),(59,63),(79,83)]: arrow(ax,(a,67),(b,67))
text(ax,50,46,'Every block: pre-RMSNorm mixer + residual, then pre-RMSNorm SwiGLU + residual',13,ha='center')
card(ax,8,16,26,17,'State per CARS head','continuous learned matrix',TEAL)
card(ax,38,16,24,17,'SwiGLU','feed-forward width 576',GOLD)
card(ax,66,16,26,17,'Next-byte logits','final RMSNorm + tied head',PURPLE)
arrow(ax,(90,59),(79,34),PURPLE,rad=-.2)
text(ax,5,6,'Training context: 128. Configured maximum: 512. MLA attention means the complete hybrid is not constant-memory.',10,MUTED)
save(fig,'architecture')

fig, ax = canvas(14,8.2)
text(ax,5,94,'Inside one CARS head',26,weight='bold')
text(ax,5,85,'Transform history, read it, then correct it with current evidence.',14,MUTED)
card(ax,4,56,18,17,'Previous state','S(t-1)',PURPLE)
card(ax,29,56,26,17,'Transport + decay','Sbar = delta S(t-1) T',TEAL)
card(ax,64,56,29,17,'Read BEFORE write','r = Sbar q',GOLD)
arrow(ax,(22,64),(29,64)); arrow(ax,(55,64),(64,64))
card(ax,64,23,29,17,'Normalize + project','read RMSNorm / scale -> output',GOLD)
arrow(ax,(78,56),(78,40),GOLD)
card(ax,4,23,24,17,'Current evidence','K: key directions / V: values',PURPLE)
card(ax,35,23,22,17,'Ridge correction','E = V - Sbar K',TEAL)
arrow(ax,(28,31),(35,31)); arrow(ax,(42,56),(46,40),TEAL)
arrow(ax,(35,24),(23,15),TEAL,rad=.15)
text(ax,5,10,'S(t) = Sbar + E (K^T K + lambda I)^(-1) K^T',17,TEAL,weight='bold')
text(ax,5,4,'T is a Cayley transform; lambda > 0. Ridge is computed with Cholesky solves. Recurrence stays FP32.',10,MUTED)
save(fig,'operator')

fig, ax = canvas(14,6.3)
text(ax,5,92,'Measured training speedup',26,weight='bold')
text(ax,5,82,'RTX 4070 Laptop / 1,000 updates / batch 32 / context 128 / same data and optimizer',12,MUTED)
for y, minutes, label, color in [(55,29.76,'Original OG',PURPLE),(26,7.88,'Factorized fast path',TEAL)]:
    text(ax,5,y+12,label,14,color,weight='bold')
    ax.add_patch(FancyBboxPatch((5,y),minutes/30*73,9,boxstyle='round,pad=.1,rounding_size=1',facecolor=color,edgecolor='none'))
    text(ax,minutes/30*73+7,y+4.5,f'{minutes:.2f} min',15,color,weight='bold')
text(ax,5,12,'3.78x faster step compute  |  Peak allocated CUDA: 975 -> 708 MiB',15,FG,weight='bold')
text(ax,5,5,'New full training wall time: 8.38 min including startup/validation/checkpoints; evaluation/export are additional.',10,MUTED)
save(fig,'speedup')

fig, ax = canvas(14,6.8)
text(ax,5,93,'Ownership: what the completed epoch actually tests',23,weight='bold')
card(ax,5,38,42,38,'One object: 100%', 'A owns x. A -> B -> C. Query x: C',TEAL)
card(ax,53,38,42,38,'Two-object stress: 0%', 'x -> B, then y -> E. Query x: B',RED)
text(ax,26,54,'Original IID / fresh cases / five transfers',11,MUTED,ha='center')
text(ax,74,54,'The last recipient is E, but x belongs to B',11,MUTED,ha='center')
text(ax,5,24,'Candidate accuracy chooses among owners A-H. It is a synthetic task, not overall language accuracy.',12,MUTED)
text(ax,5,13,'CARS, KDA, GDN-2 and Transformer all show this shortcut after one epoch.',15,FG,weight='bold')
save(fig,'ownership')

fig, ax = canvas(14,5.2)
text(ax,5,91,'Train -> inspect -> evaluate -> export',25,weight='bold')
for x,title,sub,color in [(4,'Qualify','outputs / states / gradients',PURPLE),
                         (28,'Train one epoch','2,887 shuffled updates',TEAL),
                         (52,'Evaluate','Wiki + ownership + stress',GOLD),
                         (76,'Export','weights + fixed-context CPU',PURPLE)]:
    card(ax,x,42,20,28,title,sub,color)
for a,b in [(24,28),(48,52),(72,76)]: arrow(ax,(a,56),(b,56))
text(ax,5,26,'Resume restores optimizer/RNG state and reconstructs the remaining shuffled chunk order.',12,MUTED)
text(ax,5,13,'One epoch = every one of the 11,824,794 next-token targets once; padded tail labels are ignored.',12,MUTED)
save(fig,'workflow')
fig, ax = canvas(14, 7.3)
text(ax,5,93,'Memory geometry: learned features, not object slots',23,weight='bold')
text(ax,5,84,'One head / one sequence position. Four heads per layer; three independent CARS layers.',12,MUTED)
card(ax,5,42,26,30,'S: 32 x 32','value coordinates x key coordinates',TEAL)
card(ax,41,58,21,17,'q: 32','read query',GOLD)
card(ax,72,58,23,17,'r: 32','Sbar @ q',GOLD)
arrow(ax,(31,65),(41,65),GOLD); arrow(ax,(62,65),(72,65),GOLD)
card(ax,41,30,21,17,'K: 32 x 2','two evidence directions',PURPLE)
card(ax,72,30,23,17,'V: 32 x 2','projected value targets',PURPLE)
arrow(ax,(41,38),(31,45),PURPLE)
arrow(ax,(83,47),(31,48),PURPLE,rad=.25)
text(ax,5,20,'Transport changes the key basis. Read retrieves history. Correction fits current key/value evidence.',12,MUTED)
text(ax,5,10,'Each head stores 4 KiB FP32 state. All 12 CARS heads: 48 KiB / sequence; training uses more memory.',11,MUTED)
save(fig,'memory_geometry')

epoch = json.loads((ROOT/'reports/epoch1_comparison.json').read_text(encoding='utf-8'))
fig, ax = canvas(14, 6.6)
text(ax,5,93,'Completed one-epoch comparison',26,weight='bold')
text(ax,5,84,'2,887 updates / 11,824,794 real targets each / seed 1337 / approximately matched parameters',12,MUTED)
headers = [(5,'Model'), (30,'Wiki NLL'), (49,'IID / 5 transfers'), (72,'Two objects'), (92,'Minutes')]
for x,label in headers:
    text(ax,x,71,label,12,ha='left' if x == 5 else 'center',weight='bold')
ax.plot([5,96],[65,65],color=MUTED,lw=.7)
labels = {'cars':'CARS-R fast','kda':'KDA hybrid','gdn2':'GDN-2 hybrid','transformer':'Transformer'}
for i,row in enumerate(epoch['models']):
    y = 56-i*11
    color = TEAL if row['model']=='cars' else FG
    text(ax,5,y,labels[row['model']],13,color,weight='bold')
    text(ax,30,y,f"{row['wiki_nll']:.5f}",14,color,ha='center')
    text(ax,49,y,f"{row['iid']:.0%} / {row['five_transfers']:.0%}",13,color,ha='center')
    text(ax,72,y,f"{row['two_objects']:.0%}",13,RED,ha='center')
    text(ax,92,y,f"{row['training_minutes']:.2f}",13,color,ha='center')
text(ax,5,12,'Lower Wiki byte NLL is better. Fresh IID is 100% and unchanged-object accuracy is 0% for every model.',10,MUTED)
text(ax,5,5,'Times cover logged training steps; evaluation/export excluded. GPU load varied; these are local portable baselines.',10,MUTED)
save(fig,'epoch_results')
print(f'Generated eight original diagrams, each as PNG and editable SVG: {OUT}')
