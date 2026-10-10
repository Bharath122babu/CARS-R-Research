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
    for ext in ('svg','png'): fig.savefig(OUT/f'{name}.{ext}',dpi=150,facecolor=BG)
    plt.close(fig)

fig, ax = canvas(14,4.7)
text(ax,5,83,'CARS-R',44,weight='bold')
text(ax,5,64,'Recurrent relational memory + causal attention anchors',17,MUTED)
for x,title,sub,color in [(5,'2.04M','laptop model parameters',PURPLE),
                         (36,'3.78x','measured training speedup',TEAL),
                         (67,'100%','single-object ownership',GOLD)]:
    card(ax,x,20,28,27,title,sub,color)
text(ax,5,8,'Original state law. Qualified faster execution. One-seed research findings; two-object stress remains 0%.',10,MUTED)
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
text(ax,5,93,'Ownership: strength and remaining weakness',24,weight='bold')
card(ax,5,38,42,38,'One object: 100%', 'A owns x. A -> B -> C. Query x: C',TEAL)
card(ax,53,38,42,38,'Two-object stress: 0%', 'x -> B, then y -> E. Query x: B',RED)
text(ax,26,54,'Original IID / fresh cases / five transfers',11,MUTED,ha='center')
text(ax,74,54,'The last recipient is E, but x belongs to B',11,MUTED,ha='center')
text(ax,5,24,'Candidate accuracy chooses among owners A-H. It is a synthetic task, not overall language accuracy.',12,MUTED)
text(ax,5,13,'Fast execution preserved both the successful behavior and the failure pattern.',15,FG,weight='bold')
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
print(f'Generated six original diagrams, each as PNG and editable SVG: {OUT}')
