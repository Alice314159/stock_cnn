# -*- coding: utf-8 -*-
# train_hs300_final.py
import os, sys, math, json, random, warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score

from loguru import logger
import matplotlib.pyplot as plt
from joblib import dump, load as joblib_load

# ---------- Paths / Logging ----------
EPS = 1e-8
RANDOM_SEED = 42
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True, parents=True)
logger.remove()
logger.add(LOG_DIR / "train.log",
           rotation="5 MB", retention=10, compression="zip",
           encoding="utf-8", enqueue=True, backtrace=False, diagnose=False)
logger.add(sys.stderr, level="INFO")

CKPT_PTH   = OUT_DIR / "best_model.pth"
SCALER_PKL = OUT_DIR / "scaler.joblib"
SELECT_PKL = OUT_DIR / "selector.joblib"
MASK_NPY   = OUT_DIR / "feature_mask.npy"
META_JSON  = OUT_DIR / "meta.json"

def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---------- Robust CSV / Header ----------
STANDARD_COL_MAP = {
    "date": ["date","trade_date","交易日期","日期","time","datetime"],
    "open": ["open","open_price","开盘","开盘价"],
    "high": ["high","high_price","最高","最高价"],
    "low" : ["low","low_price","最低","最低价"],
    "close":["close","close_price","收盘","收盘价","endprice","last","last_price"],
    "volume":["volume","vol","成交量","turnover_volume","volume(share)"],
    "transaction_amount":["amount","成交额","turnover","transaction_amount","金额"],
    "amplitude":["amplitude","振幅"],
    "turnover_rate":["turnover_rate","换手率"],
    "change_percentage":["change_percentage","涨跌幅","涨跌%","pct_chg"],
    "change_amount":["change_amount","涨跌额","chg"],
    "secu_code":["secu_code","ts_code","代码","code","ticker"],
    "preclose":["preclose","pre_close","昨收","前收","pre_close_price"]
}
def _clean_col(s): return str(s).strip().lower().replace("\u00a0","").replace("\u3000","").replace("\ufeff","")
def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig","utf-8","gbk","cp936"]:
        for sep in [",","\t",";","|"]:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 2: return df
            except: pass
    return pd.read_csv(path)
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    raw = [_clean_col(c) for c in df.columns]; new = raw.copy()
    for std, aliases in STANDARD_COL_MAP.items():
        aliases = [_clean_col(a) for a in aliases]
        for i, c in enumerate(raw):
            if c in aliases: new[i] = std
    df.columns = new; return df
def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.replace("，","",regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    req = ["open","high","low","close","volume"]
    if all(c in df.columns for c in req): return df
    if "close" not in df.columns:
        pc = next((c for c in ("preclose","pre_close") if c in df.columns), None)
        if pc and "change_amount" in df.columns:
            df["close"] = df[pc] + df["change_amount"]
        elif pc and "change_percentage" in df.columns:
            df["close"] = df[pc] * (1 + df["change_percentage"]/100.0)
    if not all(c in df.columns for c in req):
        missing = [c for c in req if c not in df.columns]
        raise KeyError(f"Missing OHLCV: {missing}")
    return df
def prep(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        for dc in ('trade_date','Date','TRADE_DATE'):
            if dc in df.columns: df['date'] = df[dc]; break
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)

# ---------- Features ----------
def rsi(s: pd.Series, w=14):
    d = s.diff(); up = d.clip(lower=0).rolling(w, min_periods=2).mean()
    dn = (-d.clip(upper=0)).rolling(w, min_periods=2).mean()
    rs = up/(dn+EPS); return 100 - 100/(1+rs)
def add_price_vol(df: pd.DataFrame) -> pd.DataFrame:
    for p in (1,2,3,5,10,20):
        df[f'ret_{p}'] = df['close'].pct_change(p)
        df[f'log_ret_{p}'] = np.log((df['close']+EPS)/(df['close'].shift(p)+EPS))
    df['ret_1'] = df['close'].pct_change()
    df['log_ret_1'] = np.log((df['close']+EPS)/(df['close'].shift(1)+EPS))
    for w in (5,10,20,60):
        df[f'vol_{w}'] = df['ret_1'].rolling(w, min_periods=2).std()
        df[f'realized_vol_{w}'] = np.sqrt((df['log_ret_1']**2).rolling(w, min_periods=2).sum())
    for w in (10,20,50,200):
        sma = df['close'].rolling(w, min_periods=2).mean()
        std = df['close'].rolling(w, min_periods=2).std()
        df[f'sma_{w}'] = sma
        df[f'price_to_sma_{w}'] = df['close']/(sma+EPS) - 1.
        up = sma + 2*std; lo = sma - 2*std
        df[f'bb_pos_{w}'] = (df['close']-lo)/(up-lo+EPS)
    return df
def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    for p in (7,14,21,30): df[f'rsi_{p}'] = rsi(df['close'], p)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    for p in (5,10,20):
        df[f'roc_{p}'] = (df['close']-df['close'].shift(p))/(df['close'].shift(p)+EPS)
    for w in (14,28):
        h = df['high'].rolling(w, min_periods=2).max()
        l = df['low'] .rolling(w, min_periods=2).min()
        df[f'williams_r_{w}'] = (h - df['close'])/(h-l+EPS)*-100
    return df
def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    for w in (5,10,20,50):
        df[f'vol_ma_{w}'] = df['volume'].rolling(w, min_periods=2).mean()
        df[f'vol_ratio_{w}'] = df['volume']/(df[f'vol_ma_{w}']+EPS)
    sign = np.sign(df['close'].diff().fillna(0))
    df['obv'] = (sign*df['volume']).fillna(0).cumsum()
    df['ret_1'] = df['close'].pct_change()
    df['vpt'] = (df['volume']*df['ret_1'].fillna(0)).cumsum()
    tp = (df['high']+df['low']+df['close'])/3.0
    mf = tp*df['volume']
    pos = mf.where(tp>tp.shift(1),0).rolling(14, min_periods=2).sum()
    neg = mf.where(tp<tp.shift(1),0).rolling(14, min_periods=2).sum()
    df['mfi'] = 100 - 100/(1+pos/(neg+EPS))
    return df
def rolling_beta_alpha(sret: pd.Series, mret: pd.Series, w=60):
    d = pd.concat({'s':sret,'m':mret}, axis=1).dropna()
    cov = d['s'].rolling(w, min_periods=10).cov(d['m'])
    var = d['m'].rolling(w, min_periods=10).var()
    beta = cov/(var+EPS); alpha = d['s'] - beta*d['m']
    return beta.reindex(sret.index), alpha.reindex(sret.index)
def add_xs_with_index(stock: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    s = stock.copy(); m = mkt.copy()
    s['ret_1'] = s['close'].pct_change(); m['ret_1'] = m['close'].pct_change()
    s2, m2 = s.set_index('date'), m.set_index('date')
    beta, alpha = rolling_beta_alpha(s2['ret_1'], m2['ret_1'], 60)
    out = pd.DataFrame(index=s2.index)
    out['mkt_beta']  = beta
    out['mkt_alpha'] = alpha
    out['mkt_relative_strength'] = s2['close'].pct_change(20) - m2['close'].pct_change(20).reindex(s2.index)
    def acorr(x, lag):
        if x.size <= lag: return np.nan
        return pd.Series(x).autocorr(lag=lag)
    for lag in (1,2,5):
        out[f'mkt_autocorr_{lag}'] = s2['ret_1'].rolling(20, min_periods=20).apply(lambda arr: acorr(arr, lag))
    return out.reset_index()
def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    r1 = df['close'].pct_change()
    v20 = r1.rolling(20, min_periods=2).std()
    v60 = r1.rolling(60, min_periods=2).std()
    df['vol_regime'] = (v20 > 1.2*(v60+EPS)).astype(int)
    for w in (10,20,50):
        sma = df['close'].rolling(w, min_periods=2).mean()
        df[f'trend_strength_{w}'] = (df['close']-sma)/(sma+EPS)
    return df
def compute_features(df_stock: pd.DataFrame, df_mkt: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = add_price_vol(prep(df_stock))
    df = add_momentum(df); df = add_volume(df); df = add_regime(df)
    if df_mkt is not None:
        xs = add_xs_with_index(df, prep(df_mkt))
        keep = ["date","mkt_beta","mkt_alpha","mkt_relative_strength","mkt_autocorr_1","mkt_autocorr_2","mkt_autocorr_5"]
        xs = xs[[c for c in keep if c in xs.columns]]
        df = df.merge(xs, on='date', how='left')
    df = df.replace([np.inf,-np.inf], 0).fillna(method='ffill').fillna(0)
    return df

# ---------- Config ----------
@dataclass
class Config:
    data_path: str = r"E:\Vesper\data_downloader\data\raw\HS300"
    market_index_path: Optional[str] = r"E:\Vesper\data_downloader\data\raw\indexData\HS300_daily_kline.csv"
    file_glob: str = "*.csv"
    train_split: Tuple[str,str] = ("2020-01-01","2024-12-31")
    val_split:   Tuple[str,str] = ("2025-01-01","2025-12-31")
    window: int = 60
    H: int = 5
    tau: float = 0.010          # 建议降低以缓解不平衡
    min_points: int = 120
    model_type: str = "cnn"     # cnn|transformer|lstm
    dropout: float = 0.2
    hidden: Dict[str,int] = None
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_patience: int = 5
    grad_clip: float = 1.0
    use_cyclical_lr: bool = True
    use_mixed_precision: bool = True
    feature_noise_std: float = 0.01
    use_robust_scaler: bool = True
    use_feature_selection: bool = True
    k_features: int = 128       # 提高选择阈值
    num_workers: int = max((os.cpu_count() or 4)-1, 2)
    prefetch_factor: int = 4
    persistent_workers: bool = True
    loss_type: str = "focal"    # focal|weighted_ce|label_smooth
    focal_gamma: float = 1.5
    def __post_init__(self):
        if self.hidden is None:
            self.hidden = {'conv1':64,'conv2':128,'conv3':256,'fc1':256,'fc2':128,'attn':128}

# ---------- Models ----------
class SelfAttention1D(nn.Module):
    def __init__(self, d, heads=8, drop=0.1):
        super().__init__(); self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); self.norm = nn.LayerNorm(d)
    def forward(self, x): x = x.transpose(1,2); y,_ = self.attn(x,x,x); return self.norm(x+y).transpose(1,2)
class SEBlock1D(nn.Module):
    def __init__(self, c, r=16): super().__init__(); self.gap=nn.AdaptiveAvgPool1d(1); self.fc1=nn.Linear(c,c//r); self.fc2=nn.Linear(c//r,c)
    def forward(self,x): b,c,_=x.size(); s=self.gap(x).view(b,c); s=torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).view(b,c,1); return x*s
class ConvBlk(nn.Module):
    def __init__(self, ci, co, k, drop=0.2, use_se=True, use_attn=False):
        super().__init__(); self.conv=nn.Conv1d(ci,co,k,padding=k//2); self.bn=nn.BatchNorm1d(co)
        self.se=SEBlock1D(co) if use_se else None; self.attn=SelfAttention1D(co,8,drop) if use_attn else None; self.drop=nn.Dropout(drop)
    def forward(self,x): x=F.gelu(self.bn(self.conv(x))); x=self.se(x) if self.se else x; x=self.attn(x) if self.attn else x; return self.drop(x)
class CNN1D(nn.Module):
    def __init__(self, in_c, cfg: Config):
        super().__init__(); h=cfg.hidden
        self.c1=ConvBlk(in_c,h['conv1'],7,cfg.dropout,True,True); self.c2=ConvBlk(h['conv1'],h['conv2'],5,cfg.dropout,True,True); self.c3=ConvBlk(h['conv2'],h['conv3'],3,cfg.dropout,True,False)
        self.gap=nn.AdaptiveAvgPool1d(1); self.gmp=nn.AdaptiveMaxPool1d(1)
        self.fc=nn.Sequential(nn.Dropout(cfg.dropout),nn.Linear(h['conv3']*2,h['fc1']),nn.BatchNorm1d(h['fc1']),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(h['fc1'],h['fc2']),nn.BatchNorm1d(h['fc2']),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(h['fc2'],2))
    def forward(self,x): x=F.max_pool1d(self.c1(x),2); x=F.max_pool1d(self.c2(x),2); x=self.c3(x); z=torch.cat([self.gap(x).squeeze(-1),self.gmp(x).squeeze(-1)],1); return self.fc(z)
class Transformer1D(nn.Module):
    def __init__(self,in_c,cfg:Config):
        super().__init__(); d=cfg.hidden['attn']; self.proj=nn.Linear(in_c,d); self.pe=self._pos(cfg.window,d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=8,dim_feedforward=d*4,dropout=cfg.dropout,batch_first=True,activation='gelu'); self.enc=nn.TransformerEncoder(layer,4)
        self.fc=nn.Sequential(nn.Dropout(cfg.dropout),nn.Linear(d,cfg.hidden['fc1']),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden['fc1'],2))
    def _pos(self,L,D): pe=torch.zeros(L,D); pos=torch.arange(0,L).unsqueeze(1).float(); div=torch.exp(torch.arange(0,D,2).float()*(-math.log(10000.0)/D)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); return nn.Parameter(pe.unsqueeze(0),requires_grad=False)
    def forward(self,x): x=x.transpose(1,2); x=self.proj(x)+self.pe[:, :x.size(1), :]; x=self.enc(x).mean(1); return self.fc(x)
class LSTM1D(nn.Module):
    def __init__(self,in_c,cfg:Config): super().__init__(); self.lstm=nn.LSTM(in_c,128,2,dropout=cfg.dropout,batch_first=True,bidirectional=True); self.head=nn.Sequential(nn.Dropout(cfg.dropout),nn.Linear(256,128),nn.GELU(),nn.Dropout(cfg.dropout),nn.Linear(128,2))
    def forward(self,x): x=x.transpose(1,2); y,_=self.lstm(x); return self.head(y[:,-1,:])

# ---------- Loss ----------
class LossWrap(nn.Module):
    def __init__(self, loss_type="focal", class_weights: Optional[torch.Tensor]=None, focal_alpha=1.0, focal_gamma=1.5, label_smoothing=0.1):
        super().__init__(); self.t=loss_type; self.w=class_weights; self.alpha=focal_alpha; self.gamma=focal_gamma; self.eps=label_smoothing
    def forward(self, logits, y):
        if self.t=="focal":
            ce=F.cross_entropy(logits,y,reduction='none',weight=self.w); pt=torch.exp(-ce); return (self.alpha*(1-pt)**self.gamma*ce).mean()
        if self.t=="label_smooth":
            logp=F.log_softmax(logits,1); y1=F.one_hot(y,2).float(); y1=y1*(1-self.eps)+self.eps/2; return -(y1*logp).sum(1).mean()
        return F.cross_entropy(logits,y,weight=self.w)

# ---------- Dataset ----------
class KlineDatasetHS300(Dataset):
    def __init__(self, cfg: Config, split: Tuple[str,str], scaler=None, selector=None, final_mask=None, is_train=True):
        self.cfg=cfg; self.split=split; self.scaler=scaler; self.selector=selector; self.final_mask=final_mask; self.is_train=is_train
        self.samples: List[tuple]=[]; self._load_market(); self._build()
        if self.is_train: self._fit_preproc()
        self._apply_preproc()
    def _load_market(self):
        self.df_mkt=None
        p=self.cfg.market_index_path
        if p and os.path.exists(p):
            m=normalize_headers(read_csv_safely(p)); m=coerce_numeric(m,["open","high","low","close","volume"])
            if 'close' in m.columns: self.df_mkt=prep(m)[['date','open','high','low','close','volume']]
    def _build(self):
        files=list(Path(self.cfg.data_path).glob(self.cfg.file_glob))
        if not files: raise ValueError("No CSV")
        results=[]; maxw=min(8, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=maxw) as ex:
            for fp in files: results.append(ex.submit(self._process_one, str(fp)))
        feats=[]; labels=[]
        for fut in results:
            r=fut.result()
            if r is None: continue
            f,l,s=r
            if f.size==0: continue
            feats.append(f); labels.append(l); self.samples+=s
        if not feats: raise ValueError("No valid samples.")
        self.all_features=np.vstack(feats); self.all_labels=np.concatenate(labels)
        logger.info(f"{len(self.samples):,} samples; fit rows {self.all_features.shape[0]:,}")
    def _process_one(self, path:str):
        try:
            df=normalize_headers(read_csv_safely(path))
            df=coerce_numeric(df,["open","high","low","close","volume","transaction_amount","amplitude","turnover_rate","change_percentage","change_amount","preclose"])
            cols=set(df.columns)
            if ("secu_code" not in cols) and cols.issubset({"date","close"}): return None
            df=ensure_ohlcv(df); df=prep(df)
            if len(df)<self.cfg.min_points: return None
            feat=compute_features(df, self.df_mkt)
            if "close" not in feat.columns:
                for cand in ("close_x","close_y"):
                    if cand in feat.columns: feat["close"]=feat[cand]
                if "close" not in feat.columns: return None
            future=feat['close'].shift(-self.cfg.H); ret=(future-feat['close'])/(feat['close']+EPS); y=(ret>self.cfg.tau).astype(int)
            mask=(feat['date']>=pd.Timestamp(self.split[0])) & (feat['date']<=pd.Timestamp(self.split[1]))
            feat=feat[mask].reset_index(drop=True); y=y[mask].reset_index(drop=True)
            excl=set(['date','open','high','low','close','volume'])
            cols=[c for c in feat.columns if c not in excl and pd.api.types.is_numeric_dtype(feat[c])]
            if not cols: return None
            X=feat[cols].values.astype(np.float32); win=self.cfg.window; warmup=max(win,60)
            rows=[]; labs=[]; samples=[]
            for t in range(warmup, len(feat)-self.cfg.H):
                if pd.isna(y.iloc[t]): continue
                xw=X[t-win:t].T; yy=int(y.iloc[t])
                samples.append((torch.from_numpy(xw.copy()), yy))
                rows.append(X[t]); labs.append(yy)
            if rows: return np.array(rows), np.array(labs), samples
            return None
        except Exception as e:
            logger.warning(f"Fail {path}: {e}"); return None
    def _fit_preproc(self):
        self.scaler = RobustScaler() if self.cfg.use_robust_scaler else StandardScaler()
        self.scaler.fit(self.all_features)
        if self.cfg.use_feature_selection:
            var=self.all_features.var(0); nz=var>1e-6
            if nz.sum()>0:
                Xf=self.all_features[:, nz]; k=min(self.cfg.k_features, Xf.shape[1])
                self.selector=SelectKBest(f_classif, k=k).fit(Xf, self.all_labels)
                mask=np.zeros(self.all_features.shape[1], dtype=bool); mask[nz]=self.selector.get_support(); self.final_mask=mask
            else:
                self.selector=None; self.final_mask=np.ones(self.all_features.shape[1], dtype=bool)
        else:
            self.selector=None; self.final_mask=np.ones(self.all_features.shape[1], dtype=bool)
        # —— 长期修：保存 sklearn 对象和特征掩码 —— #
        dump(self.scaler, SCALER_PKL)
        if self.selector is not None: dump(self.selector, SELECT_PKL)
        np.save(MASK_NPY, self.final_mask.astype(np.bool_))
    def _apply_preproc(self):
        if self.scaler is None: return
        mask=getattr(self, "final_mask", None)
        new=[]
        for x,y in self.samples:
            xnp=x.numpy().T                               # [T,C]
            x_scaled=self.scaler.transform(xnp)           # [T,C]
            if mask is not None: x_scaled=x_scaled[:, mask]
            new.append((torch.from_numpy(x_scaled.T.astype(np.float32)), y))
        self.samples=new
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

# ---------- Trainer / Eval ----------
class Trainer:
    def __init__(self, model: nn.Module, cfg: Config, device: str):
        self.m=model; self.cfg=cfg; self.dev=device
        self.opt=optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler=GradScaler(enabled=(cfg.use_mixed_precision and device=='cuda'))
        self.sched=None
        self.hist={'tr_loss':[],'val_loss':[],'val_auc':[],'val_acc':[]}
    def set_steps(self, steps:int):
        if self.cfg.use_cyclical_lr:
            self.sched=optim.lr_scheduler.OneCycleLR(self.opt, max_lr=self.cfg.lr, epochs=self.cfg.epochs, steps_per_epoch=steps, pct_start=0.3, anneal_strategy='cos')
        else:
            self.sched=optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=3)
    def train_epoch(self, loader, criterion):
        self.m.train(); tot=0; corr=0; loss_sum=0.0
        for X,y in loader:
            X=X.to(self.dev, non_blocking=True); y=y.to(self.dev, non_blocking=True)
            if self.cfg.feature_noise_std>0: X=X+torch.randn_like(X)*self.cfg.feature_noise_std
            self.opt.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                with autocast(): out=self.m(X); loss=criterion(out,y)
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.m.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()
            else:
                out=self.m(X); loss=criterion(out,y); loss.backward()
                nn.utils.clip_grad_norm_(self.m.parameters(), self.cfg.grad_clip); self.opt.step()
            if self.sched and self.cfg.use_cyclical_lr: self.sched.step()
            bs=X.size(0); loss_sum+=loss.item()*bs; tot+=bs; corr+=(out.argmax(1)==y).sum().item()
        return loss_sum/tot, corr/tot
    @torch.no_grad()
    def validate(self, loader, criterion):
        self.m.eval(); tot=0; corr=0; loss_sum=0.0; probs=[]; labs=[]
        for X,y in loader:
            X=X.to(self.dev, non_blocking=True); y=y.to(self.dev, non_blocking=True)
            if self.scaler.is_enabled():
                with autocast(): out=self.m(X); loss=criterion(out,y)
            else:
                out=self.m(X); loss=criterion(out,y)
            p=F.softmax(out,1)[:,1].detach().cpu().numpy(); probs.append(p); labs.append(y.detach().cpu().numpy())
            bs=X.size(0); loss_sum+=loss.item()*bs; tot+=bs; corr+=(out.argmax(1)==y).sum().item()
        probs=np.concatenate(probs); labs=np.concatenate(labs)
        auc=roc_auc_score(labs, probs) if len(np.unique(labs))>1 else 0.0
        return loss_sum/tot, corr/tot, auc, probs, labs

def collate_fn(batch):
    X=torch.stack([b[0] for b in batch]); y=torch.tensor([b[1] for b in batch], dtype=torch.long); return X,y

def plot_curves(hist: Dict, path=OUT_DIR/"learning_curves.png"):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(hist['tr_loss'],label='train'); plt.plot(hist['val_loss'],label='val'); plt.title("Loss"); plt.legend()
    plt.subplot(1,2,2); plt.plot(hist['val_auc'],label='val AUC'); plt.title("Val AUC"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); logger.info(f"Curves saved -> {path}")

def tune_threshold_by_f1(probs: np.ndarray, labels: np.ndarray, lo=0.2, hi=0.8, steps=61):
    best_t=0.5; best_f1=-1
    for t in np.linspace(lo, hi, steps):
        preds=(probs>=t).astype(int); f1=f1_score(labels, preds, zero_division=0)
        if f1>best_f1: best_f1, best_t = f1, t
    return best_t, best_f1

# ---------- Main ----------
def main(cfg: Config):
    set_seed(); device="cuda" if torch.cuda.is_available() else "cpu"; logger.info(f"Device: {device}")
    # Datasets
    train_ds = KlineDatasetHS300(cfg, cfg.train_split, is_train=True)
    # 兼容：若存在 joblib/npy 则优先使用（推理/复现）
    scaler = joblib_load(SCALER_PKL) if SCALER_PKL.exists() else train_ds.scaler
    selector= joblib_load(SELECT_PKL) if SELECT_PKL.exists() else train_ds.selector
    final_mask = np.load(MASK_NPY) if MASK_NPY.exists() else getattr(train_ds, "final_mask", None)

    val_ds = KlineDatasetHS300(cfg, cfg.val_split, scaler=scaler, selector=selector, final_mask=final_mask, is_train=False)

    nw=cfg.num_workers; pin=(device=='cuda')
    train_loader=DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=pin,
                            prefetch_factor=(cfg.prefetch_factor if nw>0 else None),
                            persistent_workers=(cfg.persistent_workers if nw>0 else False),
                            collate_fn=collate_fn)
    val_loader=DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=nw, pin_memory=pin,
                          prefetch_factor=(cfg.prefetch_factor if nw>0 else None),
                          persistent_workers=(cfg.persistent_workers if nw>0 else False),
                          collate_fn=collate_fn)
    # Model
    in_c = train_ds[0][0].shape[0]
    if   cfg.model_type=="cnn":        model=CNN1D(in_c, cfg).to(device)
    elif cfg.model_type=="transformer":model=Transformer1D(in_c, cfg).to(device)
    elif cfg.model_type=="lstm":       model=LSTM1D(in_c, cfg).to(device)
    else: raise ValueError("model_type must be cnn|transformer|lstm")

    # Class weights
    labels=[s[1] for s in train_ds.samples]; cnt=np.bincount(labels, minlength=2); cnt=np.where(cnt==0,1,cnt)
    w = len(labels)/(len(cnt)*cnt); class_weights=torch.tensor(w, dtype=torch.float32, device=device)

    # Loss
    if cfg.loss_type=="weighted_ce":
        criterion=LossWrap(loss_type="weighted_ce", class_weights=class_weights)
    elif cfg.loss_type=="label_smooth":
        criterion=LossWrap(loss_type="label_smooth")
    else:
        criterion=LossWrap(loss_type="focal", class_weights=class_weights, focal_gamma=cfg.focal_gamma)

    # Train
    t=Trainer(model,cfg,device); t.set_steps(len(train_loader))
    best_auc=0.0; patience=0
    for ep in range(cfg.epochs):
        tr_loss, tr_acc = t.train_epoch(train_loader, criterion)
        val_loss, val_acc, val_auc, probs, labs = t.validate(val_loader, criterion)
        if not cfg.use_cyclical_lr and t.sched is not None: t.sched.step(val_loss)
        t.hist['tr_loss'].append(tr_loss); t.hist['val_loss'].append(val_loss); t.hist['val_auc'].append(val_auc); t.hist['val_acc'].append(val_acc)
        logger.info(f"Epoch {ep+1:03d}/{cfg.epochs} | Train {tr_loss:.4f}/{tr_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}/{val_auc:.4f}")
        if val_auc>best_auc:
            best_auc=val_auc; patience=0
            torch.save({'state_dict': model.state_dict()}, CKPT_PTH)  # 仅权重
        else:
            patience+=1
        if patience>=cfg.early_patience: logger.info(f"Early stopping at epoch {ep+1}"); break

    plot_curves(t.hist, OUT_DIR/"learning_curves.png")

    # Threshold tuning on validation
    # 先确保使用最佳权重
    try:
        ckpt=torch.load(CKPT_PTH, map_location=device, weights_only=True)
    except TypeError:
        ckpt=torch.load(CKPT_PTH, map_location=device)  # 旧版本兼容
    model.load_state_dict(ckpt['state_dict'])
    val_loss, val_acc, val_auc, probs, labs = t.validate(val_loader, criterion)
    best_t, best_f1 = tune_threshold_by_f1(probs, labs, lo=0.2, hi=0.8, steps=61)

    meta = {
        "best_threshold": float(best_t),
        "best_f1": float(best_f1),
        "val_auc": float(val_auc),
        "config": asdict(cfg)
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Meta saved -> {META_JSON} | threshold={best_t:.3f} F1={best_f1:.4f}")

    # Final report with tuned threshold
    preds=(probs>=best_t).astype(int)
    logger.info(f"FINAL  Val Acc={val_acc:.4f}  AUC={val_auc:.4f}  Thr={best_t:.3f}")
    print("Classification Report:\n", classification_report(labs, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(labs, preds))

if __name__ == "__main__":
    cfg = Config(
        data_path=r"E:\Vesper\data_downloader\data\raw\HS300",
        market_index_path=r"E:\Vesper\data_downloader\data\raw\indexData\HS300_daily_kline.csv",
        model_type="cnn",
        batch_size=256, epochs=30, lr=1e-3, early_patience=5,
        use_cyclical_lr=True, use_mixed_precision=(torch.cuda.is_available()),
        use_feature_selection=True, k_features=128,
        num_workers=max((os.cpu_count() or 4)-1, 2), prefetch_factor=4, persistent_workers=True,
        tau=0.010, focal_gamma=1.5, loss_type="focal"
    )
    main(cfg)
