"""
Grokking & Fourier Features — Complete Pipeline
=================================================
Trains, analyzes, and generates all 7 publication figures.
Single file = no model class mismatch issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, time, traceback

# ─── Config ───────────────────────────────────────────────────────
P = 113
FRAC_TRAIN = 0.3
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
N_LAYERS = 1
LR = 1e-3
WD = 1.0
EPOCHS = 25000
LOG_EVERY = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EQ = "a + b"

def target_fn(a, b, p): return (a + b) % p

print(f"=== f(a,b) = ({EQ}) mod {P} ===")
print(f"{N_LAYERS}L d={D_MODEL} {N_HEADS}H MLP={D_MLP} | WD={WD} LR={LR} | {DEVICE}\n")

# ─── Dataset ──────────────────────────────────────────────────────
rng = np.random.RandomState(42)
pairs = [(a, b) for a in range(P) for b in range(P)]
rng.shuffle(pairs)
n = int(len(pairs) * FRAC_TRAIN)
def tens(pl):
    a = torch.tensor([x[0] for x in pl], dtype=torch.long)
    b = torch.tensor([x[1] for x in pl], dtype=torch.long)
    c = torch.tensor([target_fn(x[0], x[1], P) for x in pl], dtype=torch.long)
    return a, b, c
(tra, trb, trc), (tea, teb, tec) = tens(pairs[:n]), tens(pairs[n:])
print(f"Train: {len(tra)}  Test: {len(tea)}\n")

# ─── Model ────────────────────────────────────────────────────────
class TLayer(nn.Module):
    def __init__(self, d, nh, dh, dm):
        super().__init__()
        self.nh, self.dh = nh, dh
        self.WQ = nn.Parameter(torch.randn(nh,d,dh)*d**-0.5)
        self.WK = nn.Parameter(torch.randn(nh,d,dh)*d**-0.5)
        self.WV = nn.Parameter(torch.randn(nh,d,dh)*d**-0.5)
        self.WO = nn.Parameter(torch.randn(nh,dh,d)*d**-0.5)
        self.mlp = nn.Sequential(nn.Linear(d,dm), nn.GELU(), nn.Linear(dm,d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, x):
        s = x.shape[1]; r = x; x = self.ln1(x)
        ao = torch.zeros_like(r)
        m = torch.triu(torch.ones(s,s,device=x.device)*float('-inf'), diagonal=1)
        for h in range(self.nh):
            Q,K,V = x@self.WQ[h], x@self.WK[h], x@self.WV[h]
            ao += (F.softmax(Q@K.transpose(-2,-1)/self.dh**0.5+m, dim=-1)@V)@self.WO[h]
        x = r + ao; x = x + self.mlp(self.ln2(x)); return x

class GrokModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P, D_MODEL)
        self.pos = nn.Embedding(3, D_MODEL)
        self.layers = nn.ModuleList([TLayer(D_MODEL,N_HEADS,D_HEAD,D_MLP) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.unemb = nn.Linear(D_MODEL, P, bias=False)
    def forward(self, a, b):
        B = a.shape[0]; pe = self.pos(torch.arange(3,device=a.device))
        x = torch.stack([self.emb(a), self.emb(b), torch.zeros(B,D_MODEL,device=a.device)],1)+pe
        for l in self.layers: x = l(x)
        return self.unemb(self.ln(x)[:,2,:])

model = GrokModel().to(DEVICE)
np_ = sum(p.numel() for p in model.parameters())
print(f"Params: {np_:,}\n")

# ─── Training ────────────────────────────────────────────────────
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9,0.98))
td = [t.to(DEVICE) for t in [tra,trb,trc]]
ed = [t.to(DEVICE) for t in [tea,teb,tec]]

hist = {"epoch":[],"train_loss":[],"test_loss":[],"train_acc":[],"test_acc":[]}
t0 = time.time(); grok_ep = None

for ep in range(EPOCHS+1):
    model.train()
    lo = model(td[0],td[1])
    loss = F.cross_entropy(lo, td[2])
    opt.zero_grad(); loss.backward(); opt.step()

    if ep % LOG_EVERY == 0:
        model.eval()
        with torch.no_grad():
            ta = (lo.argmax(-1)==td[2]).float().mean().item()
            tl = model(ed[0],ed[1])
            tel = F.cross_entropy(tl,ed[2]).item()
            tea_ = (tl.argmax(-1)==ed[2]).float().mean().item()
        hist["epoch"].append(ep)
        hist["train_loss"].append(loss.item())
        hist["test_loss"].append(tel)
        hist["train_acc"].append(ta)
        hist["test_acc"].append(tea_)
        m=""
        if tea_>0.95 and not grok_ep: grok_ep=ep; m=" << GROKKING!"
        elif tea_>0.5 and not grok_ep: m=" ^"
        if ep%500==0 or m:
            print(f"{ep:>6} | {loss.item():.4f} | {tel:.4f} | {ta:.1%} | {tea_:.1%}{m}")

    if ep>3000 and len(hist["test_acc"])>30 and all(a>0.999 for a in hist["test_acc"][-30:]):
        print(f"\nPerfect at {ep}"); break

el = time.time()-t0
me = next((hist["epoch"][i] for i,a in enumerate(hist["train_acc"]) if a>0.99), None)
print(f"\nDone in {el:.0f}s | Train:{hist['train_acc'][-1]:.0%} Test:{hist['test_acc'][-1]:.0%}")
if grok_ep: print(f"Memorized:{me} Grokked:{grok_ep} Gap:{grok_ep-me}")

os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/model.pt")
cfg = {"p":P,"equation":EQ,"frac_train":FRAC_TRAIN,"d_model":D_MODEL,
       "n_heads":N_HEADS,"d_head":D_HEAD,"d_mlp":D_MLP,"n_layers":N_LAYERS,
       "lr":LR,"wd":WD,"n_params":np_}
for name,data in [("history",hist),("config",cfg)]:
    with open(f"outputs/{name}.json","w") as f: json.dump(data,f)


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS & FIGURES
# ═══════════════════════════════════════════════════════════════════
print("\n=== Generating figures ===")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Dark theme
BG='#0d1117'; CD='#161b22'; ED='#30363d'
plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':CD,'axes.edgecolor':ED,
    'axes.labelcolor':'#c9d1d9','text.color':'#c9d1d9',
    'xtick.color':'#8b949e','ytick.color':'#8b949e',
    'grid.color':'#21262d','grid.alpha':0.5,'font.size':11,
    'axes.grid':True,'axes.axisbelow':True,
    'mathtext.default':'regular',  # avoid LaTeX issues
})
BLU='#58a6ff';ORA='#f0883e';GRN='#3fb950';RED='#f85149'
PUR='#bc8cff';YEL='#d29922';CYN='#39d2c0';GRY='#8b949e'

model.cpu().eval()
WE = model.emb.weight.detach().numpy()
WU = model.unemb.weight.detach().numpy()
Wmlp = model.layers[0].mlp[0].weight.detach().numpy()

# Fourier math
def fbasis(p):
    F=np.zeros((p,p)); F[0]=1/np.sqrt(p)
    for k in range(1,(p+1)//2):
        x=np.arange(p)
        F[2*k-1]=np.sqrt(2/p)*np.cos(2*np.pi*k*x/p)
        F[2*k]=np.sqrt(2/p)*np.sin(2*np.pi*k*x/p)
    return F

def fpower(W,p):
    Fb=fbasis(p); c=Fb@W; pw=np.zeros((p+1)//2)
    pw[0]=np.sum(c[0]**2)
    for k in range(1,(p+1)//2): pw[k]=np.sum(c[2*k-1]**2)+np.sum(c[2*k]**2)
    return pw

Fb = fbasis(P)
efc = Fb @ WE  # fourier coefficients (P, D_MODEL)
ep = fpower(WE,P); up = fpower(WU,P)
epn = ep/(ep.sum()+1e-10); upn = up/(up.sum()+1e-10)
te_ = np.argsort(epn)[::-1]; tu_ = np.argsort(upn)[::-1]
t5e = np.sort(epn)[::-1][:5].sum(); t5u = np.sort(upn)[::-1][:5].sum()

ew = Wmlp @ WE.T
nf = np.zeros((D_MLP,(P+1)//2))
for n in range(D_MLP): nf[n] = fpower(ew[n:n+1,:].T, P)
nfn = nf/(nf.sum(1,keepdims=True)+1e-10)
sel = np.max(nfn,1); mf = np.argmax(nfn,1)
fc = np.bincount(mf[sel>0.15],minlength=(P+1)//2)
af = np.where(fc>0)[0]

print(f"Top-5 embed:{t5e:.0%} unembed:{t5u:.0%} tuned_neurons:{(sel>0.15).sum()}")

def smooth(y, w=7):
    y=np.array(y,dtype=float)
    if len(y)<w: return y
    # Median filter to kill spikes, then moving average
    try:
        from scipy.ndimage import median_filter
        y = median_filter(y, size=min(w,len(y))//2*2+1)
    except ImportError:
        # Manual spike removal: clip to 3x rolling median
        hw=min(w//2,5)
        for i in range(hw,len(y)-hw):
            med=np.median(y[max(0,i-hw):i+hw+1])
            if y[i]>3*med+0.01: y[i]=med
    return np.convolve(y,np.ones(w)/w,mode='same')

epochs = np.array(hist["epoch"])
freqs = np.arange(len(epn))

# ─── FIG 1: Grokking ─────────────────────────────────────────────
try:
    fig,(a1,a2)=plt.subplots(1,2,figsize=(18,6.5))
    
    trl=smooth(hist["train_loss"],51); tel_=smooth(hist["test_loss"],51)
    a1.semilogy(epochs,trl,color=BLU,lw=2,alpha=0.9,label="Train")
    a1.semilogy(epochs,tel_,color=ORA,lw=2,alpha=0.9,label="Test")
    a1.fill_between(epochs,trl,alpha=0.08,color=BLU)
    a1.fill_between(epochs,tel_,alpha=0.08,color=ORA)
    a1.set_xlabel("Epoch",fontsize=13); a1.set_ylabel("Cross-Entropy Loss",fontsize=13)
    a1.set_title(f"Loss  --  f(a,b) = ({EQ}) mod {P}",fontsize=15,fontweight='bold',pad=14)
    a1.legend(fontsize=12,framealpha=0.2)
    
    tra__=100*smooth(hist["train_acc"],31); tea__=100*smooth(hist["test_acc"],31)
    a2.plot(epochs,tra__,color=BLU,lw=2.5,alpha=0.9,label="Train")
    a2.plot(epochs,tea__,color=ORA,lw=2.5,alpha=0.9,label="Test")
    a2.axhline(100,color=GRY,ls='--',alpha=0.2)
    a2.set_xlabel("Epoch",fontsize=13); a2.set_ylabel("Accuracy (%)",fontsize=13)
    a2.set_title("Grokking -- Delayed Generalization",fontsize=15,fontweight='bold',pad=14)
    a2.set_ylim(-5,110); a2.set_xlim(0,epochs[-1])
    
    if me and grok_ep:
        a2.axvspan(0,me,alpha=0.05,color=BLU)
        a2.axvspan(me,grok_ep,alpha=0.07,color=RED)
        a2.axvspan(grok_ep,epochs[-1],alpha=0.05,color=GRN)
        a2.text(me/2,106,"Learning",ha='center',fontsize=9,color=BLU,alpha=0.7)
        a2.text((me+grok_ep)/2,106,"Memorization",ha='center',fontsize=9,color=RED,alpha=0.7)
        a2.text((grok_ep+epochs[-1])/2,106,"Generalized",ha='center',fontsize=9,color=GRN,alpha=0.7)
        a2.axvline(grok_ep,color=GRN,ls=':',alpha=0.6,lw=2)
        a2.annotate(f'Grok @ {grok_ep}',xy=(grok_ep,95),
                    xytext=(grok_ep+epochs[-1]*0.08,50),
                    fontsize=12,fontweight='bold',color=GRN,
                    arrowprops=dict(arrowstyle='->',color=GRN,lw=2),
                    bbox=dict(boxstyle='round,pad=0.4',fc=CD,ec=GRN,alpha=0.9))
    a2.legend(fontsize=12,loc='center right',framealpha=0.2)
    plt.tight_layout(pad=2.5)
    plt.savefig("outputs/01_grokking_curve.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [1/7] grokking_curve.png")
except Exception as e:
    print(f"  [1/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 2: Fourier Spectrum ─────────────────────────────────────
try:
    fig,(a1,a2)=plt.subplots(1,2,figsize=(18,6))
    for ax,data,top,bc,hc,title in [
        (a1,epn,te_,BLU,CYN,f"Embedding -- top-5: {t5e:.0%}"),
        (a2,upn,tu_,GRN,YEL,f"Unembedding -- top-5: {t5u:.0%}")]:
        cols=[hc if i in top[:5] else bc for i in range(len(data))]
        bars=ax.bar(freqs,data,color=cols,width=0.85,alpha=0.6)
        for k in top[:5]: bars[k].set_alpha(1.0)
        ax.set_xlabel("Fourier Frequency k",fontsize=12)
        ax.set_ylabel("Power Fraction",fontsize=12)
        ax.set_title(title,fontsize=14,fontweight='bold',pad=12)
        for k in top[:3]:
            if data[k]>0.01:
                ax.annotate(f'k={k}',xy=(k,data[k]),xytext=(k+3,data[k]+0.005),
                           fontsize=10,color=hc,fontweight='bold',
                           arrowprops=dict(arrowstyle='->',color=hc,lw=1.3))
    plt.tight_layout(pad=2.5)
    plt.savefig("outputs/02_fourier_spectrum.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [2/7] fourier_spectrum.png")
except Exception as e:
    print(f"  [2/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 3: Heatmap ──────────────────────────────────────────────
try:
    fig,ax=plt.subplots(figsize=(18,9))
    ny,nx=36,min(80,D_MODEL)
    pc=np.percentile(np.abs(efc),96)
    im=ax.imshow(efc[:ny,:nx],aspect='auto',cmap='RdBu_r',interpolation='nearest',vmin=-pc,vmax=pc)
    yl=["DC"]
    for k in range(1,ny//2+1): yl+=[f"cos {k}",f"sin {k}"]
    ax.set_yticks(range(min(ny,len(yl)))); ax.set_yticklabels(yl[:ny],fontsize=7)
    ax.set_xlabel("Model Dimension",fontsize=12); ax.set_ylabel("Fourier Component",fontsize=12)
    ax.set_title("Fourier Decomposition of Embedding Matrix W_E",fontsize=15,fontweight='bold',pad=12)
    plt.colorbar(im,ax=ax,shrink=0.8,pad=0.02)
    plt.tight_layout(pad=2)
    plt.savefig("outputs/03_embedding_fourier_heatmap.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [3/7] embedding_fourier_heatmap.png")
except Exception as e:
    print(f"  [3/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 4: MLP Neurons ──────────────────────────────────────────
try:
    si=np.argsort(sel)[::-1]
    nn_=min(100,D_MLP); nf_=min(45,(P+1)//2)
    fig,ax=plt.subplots(figsize=(16,9))
    im=ax.imshow(nfn[si[:nn_],:nf_],aspect='auto',cmap='hot',interpolation='nearest')
    ax.set_xlabel("Fourier Frequency k",fontsize=12)
    ax.set_ylabel("MLP Neuron (sorted by selectivity)",fontsize=12)
    ax.set_title(f"MLP Neuron Frequency Selectivity ({(sel>0.15).sum()} tuned neurons)",
                 fontsize=14,fontweight='bold',pad=12)
    plt.colorbar(im,ax=ax,shrink=0.8,pad=0.02)
    for k in af:
        if k<nf_ and fc[k]>1:
            ax.text(k,-2.5,str(fc[k]),ha='center',fontsize=7,color=YEL,fontweight='bold')
    plt.tight_layout(pad=2)
    plt.savefig("outputs/04_mlp_neuron_fourier.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [4/7] mlp_neuron_fourier.png")
except Exception as e:
    print(f"  [4/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 5: Fourier Circles ──────────────────────────────────────
try:
    # Pick top non-DC frequency
    tk = te_[0] if te_[0]!=0 else te_[1]
    
    # Project embeddings onto Fourier cos/sin DIRECTIONS (not raw dims!)
    cos_dir = efc[2*tk-1, :]  # (D_MODEL,)
    sin_dir = efc[2*tk, :]    # (D_MODEL,)
    cos_n = cos_dir / (np.linalg.norm(cos_dir)+1e-10)
    sin_n = sin_dir / (np.linalg.norm(sin_dir)+1e-10)
    
    xv = np.arange(P)
    lcos = WE @ cos_n   # learned projections
    lsin = WE @ sin_n
    icos = np.cos(2*np.pi*tk*xv/P)  # ideal
    isin = np.sin(2*np.pi*tk*xv/P)
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    # Learned circle
    ax=axes[0]
    sc=ax.scatter(lcos,lsin,c=xv,cmap='twilight_shifted',s=40,alpha=0.9,
                  edgecolors='white',linewidths=0.3,zorder=3)
    for i in range(P):
        j=(i+1)%P
        ax.plot([lcos[i],lcos[j]],[lsin[i],lsin[j]],color='white',alpha=0.06,lw=0.5)
    ax.set_xlabel("Projection onto cos direction",fontsize=11)
    ax.set_ylabel("Projection onto sin direction",fontsize=11)
    ax.set_title(f"Learned Embeddings\nprojected onto Fourier subspace k={tk}",fontsize=13,fontweight='bold',pad=10)
    ax.set_aspect('equal')
    plt.colorbar(sc,ax=ax,shrink=0.85,pad=0.03)
    
    # Ideal circle
    ax=axes[1]
    sc2=ax.scatter(icos,isin,c=xv,cmap='twilight_shifted',s=40,alpha=0.9,
                   edgecolors='white',linewidths=0.3,zorder=3)
    for i in range(P):
        j=(i+1)%P
        ax.plot([icos[i],icos[j]],[isin[i],isin[j]],color='white',alpha=0.06,lw=0.5)
    ax.set_xlabel(f"cos(2pi*{tk}*x/{P})",fontsize=11)
    ax.set_ylabel(f"sin(2pi*{tk}*x/{P})",fontsize=11)
    ax.set_title(f"Ideal Fourier Circle k={tk}",fontsize=13,fontweight='bold',pad=10)
    ax.set_aspect('equal')
    plt.colorbar(sc2,ax=ax,shrink=0.85,pad=0.03)
    
    # Correlation
    ax=axes[2]
    lcn = lcos/(np.std(lcos)+1e-10)
    lsn = lsin/(np.std(lsin)+1e-10)
    rc = np.corrcoef(lcn,icos)[0,1]
    rs = np.corrcoef(lsn,isin)[0,1]
    ax.scatter(icos,lcn,s=18,alpha=0.7,color=BLU,label=f"cos: r={rc:.3f}")
    ax.scatter(isin,lsn,s=18,alpha=0.7,color=ORA,label=f"sin: r={rs:.3f}")
    lim=max(abs(icos).max(),abs(lcn).max())*1.15
    ax.plot([-lim,lim],[-lim,lim],color=GRY,ls='--',alpha=0.3)
    ax.set_xlabel("Ideal Fourier component",fontsize=11)
    ax.set_ylabel("Learned projection (normalized)",fontsize=11)
    ax.set_title(f"Learned vs Ideal Correlation k={tk}",fontsize=13,fontweight='bold',pad=10)
    ax.legend(fontsize=11,framealpha=0.2)
    ax.set_aspect('equal')
    
    plt.tight_layout(pad=2.5)
    plt.savefig("outputs/05_fourier_circles.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [5/7] fourier_circles.png")
except Exception as e:
    print(f"  [5/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 6: Summary ──────────────────────────────────────────────
try:
    fig=plt.figure(figsize=(20,14))
    gs=gridspec.GridSpec(3,4,hspace=0.55,wspace=0.4,height_ratios=[1,0.6,1.1])
    
    # Spectra
    ax=fig.add_subplot(gs[0,0:2])
    cols=[CYN if i in te_[:5] else BLU for i in range(min(35,len(epn)))]
    ax.bar(range(min(35,len(epn))),epn[:35],color=cols,width=0.85,alpha=0.7)
    ax.set_title("Embedding Spectrum",fontsize=12,fontweight='bold'); ax.set_xlabel("Freq k")
    
    ax=fig.add_subplot(gs[0,2:4])
    cols=[YEL if i in tu_[:5] else GRN for i in range(min(35,len(upn)))]
    ax.bar(range(min(35,len(upn))),upn[:35],color=cols,width=0.85,alpha=0.7)
    ax.set_title("Unembedding Spectrum",fontsize=12,fontweight='bold'); ax.set_xlabel("Freq k")
    
    # Algorithm flow
    ax=fig.add_subplot(gs[1,:]); ax.axis('off')
    steps=[(0.6,"Input\na,b in Z/113Z",ORA),(2.8,"Embed\ncos(2pi*k*x/p)\nsin(2pi*k*x/p)",BLU),
           (5.2,"Attention\nTrig Identity\ncos(2t)=2cos^2(t)-1",GRN),
           (7.6,"MLP\nFreq-selective\nneurons",PUR),(9.6,"Output\n(a^2+b^2)\nmod 113",ORA)]
    for bx,txt,c in steps:
        ax.text(bx,0.5,txt,ha='center',va='center',fontsize=10,
                bbox=dict(boxstyle='round,pad=0.55',fc=CD,ec=c,lw=2.5,alpha=0.95))
    for i in range(len(steps)-1):
        ax.annotate('',xy=(steps[i+1][0]-0.8,0.5),xytext=(steps[i][0]+0.8,0.5),
                    arrowprops=dict(arrowstyle='->,head_width=0.3',color=GRY,lw=2.5))
    ax.set_xlim(-0.3,10.5); ax.set_ylim(-0.5,1.3)
    ax.text(5.2,-0.3,"cos(2pi*k*x^2/p) via double-angle: cos(2t)=2cos^2(t)-1",
            ha='center',fontsize=13,fontweight='bold',color=YEL,
            bbox=dict(boxstyle='round,pad=0.4',fc='#1a1800',ec=YEL,alpha=0.9))
    
    # Grokking
    ax=fig.add_subplot(gs[2,0:3])
    ax.plot(epochs,100*smooth(hist["train_acc"],31),color=BLU,lw=2,label="Train")
    ax.plot(epochs,100*smooth(hist["test_acc"],31),color=ORA,lw=2,label="Test")
    ax.axhline(100,color=GRY,ls='--',alpha=0.2)
    if me and grok_ep:
        ax.axvspan(me,grok_ep,alpha=0.06,color=RED)
        ax.axvline(grok_ep,color=GRN,ls=':',alpha=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Grokking",fontsize=12,fontweight='bold'); ax.legend(fontsize=10,framealpha=0.2)
    
    # Stats
    ax=fig.add_subplot(gs[2,3]); ax.axis('off')
    ax.text(0.05,0.95,
        f"({EQ}) mod {P}\n\nParams: {np_:,}\nLayers: {N_LAYERS}  Heads: {N_HEADS}\n"
        f"d: {D_MODEL}  MLP: {D_MLP}\nTrain: {FRAC_TRAIN:.0%}  WD: {WD}\n\n"
        f"Embed top-5: {t5e:.0%}\nUnembed top-5: {t5u:.0%}\n"
        f"Tuned neurons: {(sel>0.15).sum()}\nActive freqs: {len(af)}",
        fontsize=10,fontfamily='monospace',va='top',transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.6',fc=CD,ec=ED,alpha=0.95))
    
    plt.savefig("outputs/06_algorithm_summary.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [6/7] algorithm_summary.png")
except Exception as e:
    print(f"  [6/7] FAILED: {e}"); traceback.print_exc()


# ─── FIG 7: Logit Fourier ────────────────────────────────────────
try:
    model.eval()
    with torch.no_grad():
        av=17; lo_=model(torch.full((P,),av,dtype=torch.long),torch.arange(P,dtype=torch.long)).numpy()
    
    fig,axes=plt.subplots(2,2,figsize=(18,13))
    
    # Correct logit
    cl=np.array([lo_[b,target_fn(av,b,P)] for b in range(P)])
    axes[0,0].fill_between(range(P),cl,alpha=0.3,color=BLU)
    axes[0,0].plot(range(P),cl,color=BLU,lw=1.5)
    axes[0,0].set_xlabel("b",fontsize=12); axes[0,0].set_ylabel("Logit",fontsize=12)
    axes[0,0].set_title(f"Logit for correct answer (a={av})",fontsize=14,fontweight='bold',pad=12)
    
    # FFT
    r_=lo_[:,0]; ff=np.fft.rfft(r_); fp=np.abs(ff)**2; fp/=fp.sum()+1e-10
    tfp=np.argsort(fp)[::-1]
    bars=axes[0,1].bar(range(len(fp)),fp,color=GRN,alpha=0.5,width=0.85)
    for k in tfp[:5]: bars[k].set_color(YEL); bars[k].set_alpha(1.0)
    axes[0,1].set_xlim(-0.5,35)
    axes[0,1].set_xlabel("Fourier Frequency",fontsize=12); axes[0,1].set_ylabel("Power",fontsize=12)
    axes[0,1].set_title("FFT of Logit Column",fontsize=14,fontweight='bold',pad=12)
    
    # Reconstruction
    rcolors=[GRY,RED,ORA,YEL,GRN,CYN]
    for i,nf_ in enumerate([1,3,5,10,20]):
        fu=np.fft.rfft(r_); o=np.argsort(np.abs(fu))[::-1]
        fi=np.zeros_like(fu); fi[o[:nf_]]=fu[o[:nf_]]
        axes[1,0].plot(range(P),np.fft.irfft(fi,n=P),label=f"{nf_} modes",alpha=0.85,lw=1.8,color=rcolors[i])
    axes[1,0].plot(range(P),r_,'w--',alpha=0.2,lw=1,label="Full")
    axes[1,0].legend(fontsize=9,framealpha=0.2)
    axes[1,0].set_xlabel("Input b",fontsize=12); axes[1,0].set_ylabel("Logit",fontsize=12)
    axes[1,0].set_title("Reconstruction from Fourier Modes",fontsize=14,fontweight='bold',pad=12)
    
    # Accuracy vs modes
    nr__=list(range(1,40)); ac_=[]
    for nf_ in nr__:
        fl=np.zeros_like(lo_)
        for c in range(P):
            fu=np.fft.rfft(lo_[:,c]); o=np.argsort(np.abs(fu))[::-1]
            fi=np.zeros_like(fu); fi[o[:nf_]]=fu[o[:nf_]]; fl[:,c]=np.fft.irfft(fi,n=P)
        ac_.append(100*np.mean(np.argmax(fl,1)==np.array([target_fn(av,b,P) for b in range(P)])))
    
    axes[1,1].plot(nr__,ac_,'o-',color=PUR,lw=2.2,ms=4,alpha=0.9)
    axes[1,1].fill_between(nr__,ac_,alpha=0.12,color=PUR)
    axes[1,1].axhline(100,color=GRY,ls='--',alpha=0.2)
    axes[1,1].set_xlabel("Fourier Components Kept",fontsize=12)
    axes[1,1].set_ylabel("Accuracy (%)",fontsize=12)
    axes[1,1].set_title(f"Accuracy vs Fourier Complexity (a={av})",fontsize=14,fontweight='bold',pad=12)
    for i,a in enumerate(ac_):
        if a>=95:
            axes[1,1].annotate(f'{i+1} modes -> {a:.0f}%',xy=(i+1,a),xytext=(i+8,a-12),
                              fontsize=11,color=CYN,fontweight='bold',
                              arrowprops=dict(arrowstyle='->',color=CYN,lw=1.5))
            break
    
    plt.tight_layout(pad=2.5)
    plt.savefig("outputs/07_logit_fourier_analysis.png",dpi=200,bbox_inches='tight',facecolor=BG)
    plt.close(); print("  [7/7] logit_fourier_analysis.png")
except Exception as e:
    print(f"  [7/7] FAILED: {e}"); traceback.print_exc()


print(f"\n=== All done! Check outputs/ ===")
# List what we saved
for f in sorted(os.listdir("outputs")):
    sz = os.path.getsize(f"outputs/{f}")
    print(f"  {f:40s} {sz/1024:.1f} kB")
