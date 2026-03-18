#!/usr/bin/env python3
"""
Generate C++ submission for CodinGame from INT8 quantized weights.
Uses zlib compression + ASCII85 encoding. Fixed 64×64 spatial resolution.

Usage:
    python generate_submission.py
    python generate_submission.py --weights exported_weights_int8.npz --output submission.cpp
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "exported_weights_int8.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "submission.cpp"
MAX_SOURCE_CHARS = 100_000


def encode_ascii85(data_bytes):
    """Custom base-85 encoding using 85 safe printable ASCII chars."""
    table = []
    for c in range(33, 127):
        if c in (34, 92):  # skip " and backslash
            continue
        table.append(chr(c))
        if len(table) == 85:
            break

    padded = bytearray(data_bytes)
    orig_len = len(padded)
    while len(padded) % 4 != 0:
        padded.append(0)

    result = []
    for i in range(0, len(padded), 4):
        val = struct.unpack('>I', bytes(padded[i:i + 4]))[0]
        chars = []
        for _ in range(5):
            chars.append(table[val % 85])
            val //= 85
        result.extend(reversed(chars))

    return ''.join(result), orig_len, table


def generate_cpp(data):
    """Generate complete C++ submission source."""
    layer_names = str(data["_layer_order"][0]).split(';')
    weight_layers = []
    bias_layers = []
    for name in layer_names:
        shape = tuple(data[f"{name}.shape"])
        if 'bias' in name:
            bias_layers.append((name, data[f"{name}.float"], shape))
        else:
            weight_layers.append((name, data[f"{name}.int8"], float(data[f"{name}.scale"][0]), shape))

    total_int8 = sum(w[1].size for w in weight_layers)
    total_bias = sum(b[1].size for b in bias_layers)
    print(f"Weights: {len(weight_layers)} layers, {total_int8} INT8 values")
    print(f"Biases: {len(bias_layers)} layers, {total_bias} floats")

    # Build binary blob: INT8 weights + float32 biases
    all_uint8 = []
    w_offsets = {}
    off = 0
    for name, int8_data, scale, shape in weight_layers:
        w_offsets[name] = (off, int8_data.size)
        all_uint8.append((int8_data.flatten().astype(np.int16) + 128).astype(np.uint8))
        off += int8_data.size

    weight_bytes = np.concatenate(all_uint8).tobytes()
    bias_start = len(weight_bytes)

    all_bias_f32 = []
    b_offsets = {}
    b_off = 0
    for name, float_data, shape in bias_layers:
        b_offsets[name] = (b_off, float_data.size)
        all_bias_f32.append(float_data.flatten().astype(np.float32))
        b_off += float_data.size

    bias_bytes = np.concatenate(all_bias_f32).tobytes()
    total_blob = weight_bytes + bias_bytes
    orig_blob_len = len(total_blob)
    compressed_blob = zlib.compress(total_blob, level=9)
    encoded, comp_len, table = encode_ascii85(compressed_blob)
    print(f"Blob: {orig_blob_len} bytes "
          f"-> {len(compressed_blob)} bytes (zlib, {100*len(compressed_blob)/orig_blob_len:.1f}%) "
          f"-> {len(encoded)} chars (ascii85)")

    # Layer index maps
    w_idx = {name: i for i, (name, _, _, _) in enumerate(weight_layers)}
    b_idx = {name: i for i, (name, _, _) in enumerate(bias_layers)}

    def wi(name):
        return w_idx[name]

    def bi(name):
        return b_idx[name]

    P = []

    # ---- Headers ----
    P.append('#pragma GCC optimize("O3,unroll-loops")\n')
    P.append('#include<iostream>\n#include<string>\n#include<cstring>\n#include<cmath>\n#include<algorithm>\nusing namespace std;\n')
    P.append('static const int DX[4]={0,1,0,-1},DY[4]={-1,0,1,0},OP[4]={2,3,0,1};\n')
    P.append('static const char*DN[4]={"UP","RIGHT","DOWN","LEFT"};\n')
    P.append('static int W,H,myId,nA,aX[2000],aY[2000],nB;\n')
    P.append('static bool wG[1426],wl[1426],aG[1426],mB[1024];\n')
    P.append('struct Bot{int id,ow,bx[200],by[200],ln;\n')
    P.append('int hx()const{return bx[0];}int hy()const{return by[0];}')
    P.append('int fc()const{if(ln<2)return-1;int dx=bx[0]-bx[1],dy=by[0]-by[1];for(int d=0;d<4;d++)if(DX[d]==dx&&DY[d]==dy)return d;return-1;}};\n')
    P.append('static Bot bt[16];\n')
    P.append('static void pB(Bot&b,const string&s){b.ln=0;const char*p=s.c_str();while(*p){int x=0,y=0;while(*p>=\'0\'&&*p<=\'9\')x=x*10+(*p++-\'0\');if(*p==\',\')p++;while(*p>=\'0\'&&*p<=\'9\')y=y*10+(*p++-\'0\');b.bx[b.ln]=x;b.by[b.ln]=y;b.ln++;if(*p==\':\')p++;}}\n')
    P.append('static inline bool iB(int x,int y){return(unsigned)x<(unsigned)W&&(unsigned)y<(unsigned)H;}\n')

    # ---- Self-contained DEFLATE inflate (public domain, no -lz needed) ----
    P.append(r"""
static struct{const unsigned char*src;int bit;unsigned char buf;}gz;
static unsigned gbits(int n){unsigned v=0;for(int i=0;i<n;i++){if(gz.bit==8){gz.buf=*gz.src++;gz.bit=0;}v|=((gz.buf>>gz.bit)&1)<<i;gz.bit++;}return v;}
static unsigned grev(int n){unsigned v=0;for(int i=0;i<n;i++)v=(v<<1)|gbits(1);return v;}
static void mktree(const int*lens,int n,int*ht,int*hc,int ml){
  int cnt[17]={};for(int i=0;i<n;i++)if(lens[i])cnt[lens[i]]++;
  int nc[17]={};int c=0;for(int i=1;i<=ml;i++){c=(c+cnt[i-1])<<1;nc[i]=c;}
  for(int i=0;i<n;i++){int l=lens[i];if(!l)continue;
    int cd=nc[l]++;int rv=0;for(int b=0;b<l;b++)rv=(rv<<1)|((cd>>b)&1);
    for(int s=rv;s<(1<<ml);s+=(1<<l))ht[s]=i,hc[s]=l;}}
static int decode(const int*ht,const int*hc,int ml){int v=grev(ml);return ht[v+(((1<<ml)-1-v)&~((1<<hc[v])-1))];}
static void zinflate(const unsigned char*src,unsigned char*dst,int dlen){
  gz.src=src+2;gz.bit=8;// skip zlib header
  unsigned char*d=dst;
  static int ht[32768],hc[32768],dt[512],dc[512];
  static const int clen[]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
  static const int blen[]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
  static const int bext[]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
  static const int dbase[]={1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
  static const int dext[]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
  for(;;){int bfinal=gbits(1),btype=gbits(2);
    if(btype==0){gz.bit=8;int ln=*gz.src|(gz.src[1]<<8);gz.src+=4;memcpy(d,gz.src,ln);d+=ln;gz.src+=ln;}
    else{int lml=7,dml=5;
      if(btype==1){// fixed huffman
        static int fht[128],fhc[128],fdt[32],fdc[32];static bool fi=false;
        if(!fi){int fl[288];for(int i=0;i<144;i++)fl[i]=8;for(int i=144;i<256;i++)fl[i]=9;for(int i=256;i<280;i++)fl[i]=7;for(int i=280;i<288;i++)fl[i]=8;int dl[32];for(int i=0;i<32;i++)dl[i]=5;mktree(fl,288,fht,fhc,lml=9);mktree(dl,32,fdt,fdc,dml=5);fi=true;}
        memcpy(ht,fht,sizeof(fht));memcpy(hc,fhc,sizeof(fhc));memcpy(dt,fdt,sizeof(fdt));memcpy(dc,fdc,sizeof(fdc));lml=9;}
      else{// dynamic huffman
        int hlit=gbits(5)+257,hdist=gbits(5)+1,hclen=gbits(4)+4;
        int cl[19]={};for(int i=0;i<hclen;i++)cl[clen[i]]=gbits(3);
        static int clt[128],clc[128];mktree(cl,19,clt,clc,7);
        int ll[320]={};int i=0;while(i<hlit+hdist){int s=decode(clt,clc,7);if(s<16)ll[i++]=s;else if(s==16){int r=ll[i-1],n=gbits(2)+3;while(n--)ll[i++]=r;}else if(s==17){int n=gbits(3)+3;i+=n;}else{int n=gbits(7)+11;i+=n;}}
        mktree(ll,hlit,ht,hc,lml=15);mktree(ll+hlit,hdist,dt,dc,dml=15);}
      for(;;){int s=decode(ht,hc,lml);if(s<256)*d++=s;else if(s==256)break;else{s-=257;int ln=blen[s]+gbits(bext[s]);int ds=decode(dt,dc,dml);int dist=dbase[ds]+gbits(dext[ds]);unsigned char*back=d-dist;while(ln--)*d++=*back++;}}}
    if(bfinal)break;}
}
""")

    # ---- Decode table ----
    decode_table = [0] * 128
    for i, ch in enumerate(table):
        decode_table[ord(ch)] = i
    P.append('static const int D[128]={')
    P.append(','.join(str(v) for v in decode_table))
    P.append('};\n')

    # ---- Encoded data ----
    P.append('static const char*ED=\n')
    for i in range(0, len(encoded), 10000):
        P.append(f'"{encoded[i:i + 10000]}"\n')
    P.append(';\n')

    # ---- Metadata arrays ----
    P.append('static const float SC[]={' + ','.join(f'{s:.10f}f' for _, _, s, _ in weight_layers) + '};\n')
    P.append('static const int WO[]={' + ','.join(str(w_offsets[n][0]) for n, _, _, _ in weight_layers) + '};\n')
    P.append('static const int WZ[]={' + ','.join(str(w_offsets[n][1]) for n, _, _, _ in weight_layers) + '};\n')
    P.append('static const int BO[]={' + ','.join(str(b_offsets[n][0]) for n, _, _ in bias_layers) + '};\n')

    # ---- Weight storage ----
    # comp_len = size of the compressed (zlib) blob in bytes (padded to multiple of 4 by ascii85)
    # orig_blob_len = size of the uncompressed blob
    comp_buf_size = ((len(compressed_blob) + 3) // 4) * 4  # ascii85 always decodes to multiple-of-4
    P.append(f'static float AW[{total_int8 + 10}],AB[{total_bias + 10}];\n')
    P.append(f'static float*Wp[{len(weight_layers)}],*Bp[{len(bias_layers)}];\n')

    # ---- Decode function ----
    P.append('static void decW(){\n')
    # Step 1: ASCII85 decode into compressed buffer
    P.append(f'static unsigned char cbuf[{comp_buf_size + 4}];\n')
    P.append('const char*p=ED;int n=0;\n')
    P.append('while(*p){unsigned v=0;for(int i=0;i<5;i++){v=v*85+D[(unsigned char)*p];p++;}cbuf[n++]=(v>>24)&0xFF;cbuf[n++]=(v>>16)&0xFF;cbuf[n++]=(v>>8)&0xFF;cbuf[n++]=v&0xFF;}\n')
    # Step 2: zinflate into raw buffer (self-contained, no -lz needed)
    P.append(f'static unsigned char r[{orig_blob_len + 4}];\n')
    P.append(f'zinflate(cbuf,r,{orig_blob_len});\n')
    # Step 3: unpack weights and biases from raw buffer (same as before)
    P.append(f'for(int i=0;i<{total_int8};i++)AW[i]=((int)r[i]-128);\n')
    P.append(f'for(int i=0;i<{len(weight_layers)};i++){{Wp[i]=AW+WO[i];for(int j=0;j<WZ[i];j++)Wp[i][j]*=SC[i];}}\n')
    P.append(f'memcpy(AB,r+{bias_start},{total_bias * 4});\n')
    P.append(f'for(int i=0;i<{len(bias_layers)};i++)Bp[i]=AB+BO[i];\n')
    P.append('}\n')

    # ---- NN kernels ----
    P.append('static float ba[400000],bb[400000],s1[50000],s2[25000],s3[15000],obs[30000];\n')

    # conv3x3
    P.append('static void c33(const float*i,float*o,int H,int W,int ic,int oc,const float*w,const float*b){\n')
    P.append('int HW=H*W;for(int p=0;p<oc;p++){float B=b[p];float*op=o+p*HW;for(int j=0;j<HW;j++)op[j]=B;}\n')
    P.append('for(int c=0;c<ic;c++){const float*ip=i+c*HW;for(int p=0;p<oc;p++){float*op=o+p*HW;const float*wp=w+p*ic*9+c*9;\n')
    P.append('for(int y=0;y<H;y++)for(int x=0;x<W;x++){float s=0;for(int ky=-1;ky<=1;ky++){int iy=y+ky;if((unsigned)iy>=(unsigned)H)continue;for(int kx=-1;kx<=1;kx++){int ix=x+kx;if((unsigned)ix>=(unsigned)W)continue;s+=ip[iy*W+ix]*wp[(ky+1)*3+(kx+1)];}}op[y*W+x]+=s;}}}}\n')

    # depthwise 3x3
    P.append('static void dw(const float*i,float*o,int H,int W,int ch,const float*w,const float*b){\n')
    P.append('int HW=H*W;for(int c=0;c<ch;c++){float B=b[c];for(int y=0;y<H;y++)for(int x=0;x<W;x++){float s=B;for(int ky=-1;ky<=1;ky++){int iy=y+ky;if((unsigned)iy>=(unsigned)H)continue;for(int kx=-1;kx<=1;kx++){int ix=x+kx;if((unsigned)ix>=(unsigned)W)continue;s+=i[c*HW+iy*W+ix]*w[c*9+(ky+1)*3+(kx+1)];}}o[c*HW+y*W+x]=s;}}}\n')

    # pointwise 1x1
    P.append('static void pw(const float*i,float*o,int H,int W,int ic,int oc,const float*w,const float*b){\n')
    P.append('int HW=H*W;for(int p=0;p<oc;p++){float B=b[p];float*op=o+p*HW;for(int j=0;j<HW;j++)op[j]=B;}\n')
    P.append('for(int c=0;c<ic;c++){const float*ip=i+c*HW;for(int p=0;p<oc;p++){float wt=w[p*ic+c];float*op=o+p*HW;for(int j=0;j<HW;j++)op[j]+=ip[j]*wt;}}}\n')

    # relu
    P.append('static void rl(float*d,int n){for(int i=0;i<n;i++)if(d[i]<0)d[i]=0;}\n')

    # maxpool2x2
    P.append('static void mp(const float*i,float*o,int H,int W,int C){\n')
    P.append('int oH=H/2,oW=W/2,iHW=H*W,oHW=oH*oW;for(int c=0;c<C;c++)for(int y=0;y<oH;y++)for(int x=0;x<oW;x++){int iy=y*2,ix=x*2;float v=i[c*iHW+iy*W+ix];v=max(v,i[c*iHW+iy*W+ix+1]);v=max(v,i[c*iHW+(iy+1)*W+ix]);v=max(v,i[c*iHW+(iy+1)*W+ix+1]);o[c*oHW+y*oW+x]=v;}}\n')

    # conv_transpose 2x2
    P.append('static void ct(const float*i,float*o,int iH,int iW,int ic,int oc,const float*w,const float*b){\n')
    P.append('int oH=iH*2,oW=iW*2,oHW=oH*oW,iHW=iH*iW;for(int p=0;p<oc;p++)for(int j=0;j<oHW;j++)o[p*oHW+j]=b[p];for(int ci=0;ci<ic;ci++)for(int iy=0;iy<iH;iy++)for(int ix=0;ix<iW;ix++){float v=i[ci*iHW+iy*iW+ix];int oy=iy*2,ox=ix*2;for(int p=0;p<oc;p++)for(int ky=0;ky<2;ky++)for(int kx=0;kx<2;kx++)o[p*oHW+(oy+ky)*oW+(ox+kx)]+=v*w[ci*oc*4+p*4+ky*2+kx];}}\n')

    # adaptive avg pool
    P.append('static void ap(const float*i,float*o,int H,int W,int C,int tH,int tW){\n')
    P.append('for(int c=0;c<C;c++)for(int oh=0;oh<tH;oh++){int h0=oh*H/tH,h1=(oh+1)*H/tH;for(int ow=0;ow<tW;ow++){int w0=ow*W/tW,w1=(ow+1)*W/tW;float s=0;int n=0;for(int y=h0;y<h1;y++)for(int x=w0;x<w1;x++){s+=i[c*H*W+y*W+x];n++;}o[c*tH*tW+oh*tW+ow]=s/n;}}}\n')

    # channel concat
    P.append('static void cc(const float*a,const float*b,float*o,int H,int W,int ca,int cb){int HW=H*W;memcpy(o,a,ca*HW*4);memcpy(o+ca*HW,b,cb*HW*4);}\n')

    # linear
    P.append('static void ln(const float*i,float*o,int ic,int oc,const float*w,const float*b){for(int p=0;p<oc;p++){float s=b[p];for(int j=0;j<ic;j++)s+=i[j]*w[p*ic+j];o[p]=s;}}\n')

    # sep_conv helper
    P.append('static void sp(const float*i,float*t,float*o,int H,int W,int ic,int oc,int di,int dbi,int pi,int pbi){dw(i,t,H,W,ic,Wp[di],Bp[dbi]);pw(t,o,H,W,ic,oc,Wp[pi],Bp[pbi]);rl(o,oc*H*W);}\n')

    # ---- Forward pass (64x64 fixed) ----
    P.append('static float ft[128],ph1[128],ph2[128],al[16];\nstatic void fwd(){static float t1[400000],t2[400000];\n')

    # enc1: 64x64
    P.append(f'c33(obs,t1,64,64,7,8,Wp[{wi("enc1.conv1.weight")}],Bp[{bi("enc1.conv1.bias")}]);rl(t1,8*4096);\n')
    P.append(f'c33(t1,s1,64,64,8,8,Wp[{wi("enc1.conv2.weight")}],Bp[{bi("enc1.conv2.bias")}]);rl(s1,8*4096);mp(s1,ba,64,64,8);\n')

    # enc2: 32x32
    P.append(f'sp(ba,t1,t2,32,32,8,16,{wi("enc2.conv1.depthwise.weight")},{bi("enc2.conv1.depthwise.bias")},{wi("enc2.conv1.pointwise.weight")},{bi("enc2.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,s2,32,32,16,16,{wi("enc2.conv2.depthwise.weight")},{bi("enc2.conv2.depthwise.bias")},{wi("enc2.conv2.pointwise.weight")},{bi("enc2.conv2.pointwise.bias")});mp(s2,ba,32,32,16);\n')

    # enc3: 16x16
    P.append(f'sp(ba,t1,t2,16,16,16,32,{wi("enc3.conv1.depthwise.weight")},{bi("enc3.conv1.depthwise.bias")},{wi("enc3.conv1.pointwise.weight")},{bi("enc3.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,s3,16,16,32,32,{wi("enc3.conv2.depthwise.weight")},{bi("enc3.conv2.depthwise.bias")},{wi("enc3.conv2.pointwise.weight")},{bi("enc3.conv2.pointwise.bias")});mp(s3,ba,16,16,32);\n')

    # enc4: 8x8
    P.append(f'sp(ba,t1,t2,8,8,32,64,{wi("enc4.conv1.depthwise.weight")},{bi("enc4.conv1.depthwise.bias")},{wi("enc4.conv1.pointwise.weight")},{bi("enc4.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,bb,8,8,64,64,{wi("enc4.conv2.depthwise.weight")},{bi("enc4.conv2.depthwise.bias")},{wi("enc4.conv2.pointwise.weight")},{bi("enc4.conv2.pointwise.bias")});\n')

    # decoder
    P.append(f'ct(bb,ba,8,8,64,32,Wp[{wi("up4.weight")}],Bp[{bi("up4.bias")}]);cc(ba,s3,bb,16,16,32,32);\n')
    P.append(f'sp(bb,t1,t2,16,16,64,32,{wi("dec4.conv1.depthwise.weight")},{bi("dec4.conv1.depthwise.bias")},{wi("dec4.conv1.pointwise.weight")},{bi("dec4.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,ba,16,16,32,32,{wi("dec4.conv2.depthwise.weight")},{bi("dec4.conv2.depthwise.bias")},{wi("dec4.conv2.pointwise.weight")},{bi("dec4.conv2.pointwise.bias")});\n')

    P.append(f'ct(ba,bb,16,16,32,16,Wp[{wi("up3.weight")}],Bp[{bi("up3.bias")}]);cc(bb,s2,ba,32,32,16,16);\n')
    P.append(f'sp(ba,t1,t2,32,32,32,16,{wi("dec3.conv1.depthwise.weight")},{bi("dec3.conv1.depthwise.bias")},{wi("dec3.conv1.pointwise.weight")},{bi("dec3.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,bb,32,32,16,16,{wi("dec3.conv2.depthwise.weight")},{bi("dec3.conv2.depthwise.bias")},{wi("dec3.conv2.pointwise.weight")},{bi("dec3.conv2.pointwise.bias")});\n')

    P.append(f'ct(bb,ba,32,32,16,8,Wp[{wi("up2.weight")}],Bp[{bi("up2.bias")}]);cc(ba,s1,bb,64,64,8,8);\n')
    P.append(f'sp(bb,t1,t2,64,64,16,8,{wi("dec2.conv1.depthwise.weight")},{bi("dec2.conv1.depthwise.bias")},{wi("dec2.conv1.pointwise.weight")},{bi("dec2.conv1.pointwise.bias")});\n')
    P.append(f'sp(t2,t1,ba,64,64,8,8,{wi("dec2.conv2.depthwise.weight")},{bi("dec2.conv2.depthwise.bias")},{wi("dec2.conv2.pointwise.weight")},{bi("dec2.conv2.pointwise.bias")});\n')

    # final
    P.append(f'pw(ba,bb,64,64,8,4,Wp[{wi("final_conv.weight")}],Bp[{bi("final_conv.bias")}]);\n')
    P.append('static float pl[64];ap(bb,pl,64,64,4,4,4);\n')
    P.append(f'ln(pl,ft,64,128,Wp[{wi("feature_proj.weight")}],Bp[{bi("feature_proj.bias")}]);rl(ft,128);\n')
    P.append(f'ln(ft,ph1,128,128,Wp[{wi("policy_net.0.weight")}],Bp[{bi("policy_net.0.bias")}]);rl(ph1,128);\n')
    P.append(f'ln(ph1,ph2,128,128,Wp[{wi("policy_net.2.weight")}],Bp[{bi("policy_net.2.bias")}]);rl(ph2,128);\n')
    P.append(f'ln(ph2,al,128,16,Wp[{wi("action_net.weight")}],Bp[{bi("action_net.bias")}]);}}\n')

    # ---- Obs extraction (64x64) ----
    P.append('static void exO(){memset(obs,0,7*64*64*4);\n')
    P.append('for(int y=0;y<H;y++)for(int x=0;x<W;x++)if(wG[y*W+x])obs[y*64+x]=1.0f;\n')
    P.append('for(int i=0;i<nA;i++)if(aX[i]<64&&aY[i]<64)obs[4096+aY[i]*64+aX[i]]=1.0f;\n')
    P.append('for(int b=0;b<nB;b++){if(!mB[bt[b].id]||bt[b].ln<=0)continue;int hx=bt[b].hx(),hy=bt[b].hy();if(hx<64&&hy<64)obs[8192+hy*64+hx]=1.0f;for(int j=1;j<bt[b].ln;j++){int x=bt[b].bx[j],y=bt[b].by[j];if(x<64&&y<64){float s=1.0f-j*0.1f;if(s<0.1f)s=0.1f;obs[12288+y*64+x]=s;}}}\n')
    P.append('for(int b=0;b<nB;b++){if(mB[bt[b].id]||bt[b].ln<=0)continue;int hx=bt[b].hx(),hy=bt[b].hy();if(hx<64&&hy<64)obs[16384+hy*64+hx]=1.0f;for(int j=1;j<bt[b].ln;j++){int x=bt[b].bx[j],y=bt[b].by[j];if(x<64&&y<64){float s=1.0f-j*0.1f;if(s<0.1f)s=0.1f;obs[20480+y*64+x]=s;}}}\n')
    P.append('float hm=max(H-1,1)*1.0f;for(int y=0;y<H;y++)for(int x=0;x<W;x++)if(!wG[y*W+x])obs[24576+y*64+x]=(float)y/hm;}\n')

    # ---- Main ----
    P.append('int main(){ios_base::sync_with_stdio(false);cin.tie(nullptr);decW();\n')
    P.append('cin>>myId;cin.ignore();cin>>W;cin.ignore();cin>>H;cin.ignore();\n')
    P.append('for(int y=0;y<H;y++){string r;getline(cin,r);for(int x=0;x<W;x++)wG[y*W+x]=(r[x]==\'#\');}\n')
    P.append('memcpy(wl,wG,sizeof(wl));int bp;cin>>bp;cin.ignore();\n')
    P.append('for(int i=0;i<bp;i++){int id;cin>>id;cin.ignore();mB[id]=true;}\n')
    P.append('for(int i=0;i<bp;i++){int id;cin>>id;cin.ignore();}\n')
    P.append('while(true){cin>>nA;cin.ignore();if(cin.eof())break;\n')
    P.append('for(int i=0;i<nA;i++){cin>>aX[i]>>aY[i];cin.ignore();}\n')
    P.append('cin>>nB;cin.ignore();for(int i=0;i<nB;i++){string bd;cin>>bt[i].id>>bd;cin.ignore();pB(bt[i],bd);bt[i].ow=mB[bt[i].id]?0:1;}\n')
    P.append('for(int c=0;c<W*H;c++){wl[c]=wG[c];aG[c]=false;}\n')
    P.append('for(int b=0;b<nB;b++)for(int j=0;j<bt[b].ln;j++){int x=bt[b].bx[j],y=bt[b].by[j];if(iB(x,y))wl[y*W+x]=true;}\n')
    P.append('for(int i=0;i<nA;i++)if(iB(aX[i],aY[i])){aG[aY[i]*W+aX[i]]=true;wl[aY[i]*W+aX[i]]=true;}\n')
    P.append('exO();fwd();\n')
    P.append('int mi[4],nm=0;for(int i=0;i<nB;i++)if(mB[bt[i].id])mi[nm++]=i;bool fr=true;\n')
    P.append('for(int bi=0;bi<nm;bi++){Bot&bot=bt[mi[bi]];\n')
    P.append('float*lg=al+bi*4;int best=0;float bS=lg[0];for(int d=1;d<4;d++){if(lg[d]>bS){bS=lg[d];best=d;}}\n')
    P.append('if(!fr)cout<<\';\';cout<<bot.id<<\' \'<<DN[best];fr=false;}\n')
    P.append('if(nm==0)cout<<"WAIT";cout<<endl;}}\n')

    return ''.join(P)


def main():
    parser = argparse.ArgumentParser(description="Generate C++ submission from INT8 weights")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    print("=" * 70)
    print("Generating C++ submission...")
    print("=" * 70)

    data = np.load(args.weights, allow_pickle=True)
    cpp_source = generate_cpp(data)

    char_count = len(cpp_source)
    byte_count = len(cpp_source.encode('utf-8'))
    print(f"\nGenerated: {char_count:,} characters, {byte_count:,} bytes")

    if char_count > MAX_SOURCE_CHARS:
        print(f"WARNING: Exceeds {MAX_SOURCE_CHARS:,} char limit by {char_count - MAX_SOURCE_CHARS:,}")
    else:
        print(f"Within limit ({MAX_SOURCE_CHARS - char_count:,} chars remaining)")

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(cpp_source)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()