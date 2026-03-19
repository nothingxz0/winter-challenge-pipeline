#!/usr/bin/env python3
"""
Generate C++ submission for CodinGame from INT8 quantized weights.
Uses ASCII85 encoding. Fixed 64×64 spatial resolution.

Usage:
    python generate_submission.py
    python generate_submission.py --weights exported_weights_int8.npz --output submission.cpp
"""

import argparse
import struct
import sys
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

    all_bias_f16 = []
    b_offsets = {}
    b_off = 0
    for name, float_data, shape in bias_layers:
        b_offsets[name] = (b_off, float_data.size)
        all_bias_f16.append(float_data.flatten().astype(np.float16)) # Changed to float16
        b_off += float_data.size

    bias_bytes = np.concatenate(all_bias_f16).tobytes()
    total_blob = weight_bytes + bias_bytes
    encoded, orig_blob_len, table = encode_ascii85(total_blob)
    print(f"Blob: {orig_blob_len} bytes -> {len(encoded)} chars (ascii85)")

    # Layer index maps
    w_idx = {name: i for i, (name, _, _, _) in enumerate(weight_layers)}
    b_idx = {name: i for i, (name, _, _) in enumerate(bias_layers)}

    def wi(name):
        return w_idx[name]

    def bi(name):
        return b_idx[name]

    P = []

    # ---- Headers ----
    # ---- Headers ----
    P.append('#pragma GCC optimize("O3,inline,omit-frame-pointer,unroll-loops")\n')
    P.append('#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")\n')
    P.append('#include<iostream>\n#include<string>\n#include <stdint.h>\n#include<cstring>\n#include<cmath>\n#include<algorithm>\nusing namespace std;\n')
    P.append('static const int DX[4]={0,1,0,-1},DY[4]={-1,0,1,0};\n')
    P.append('static const char*DN[4]={"UP","RIGHT","DOWN","LEFT"};\n')
    P.append('static int W,H,myId,nA,aX[2000],aY[2000],nB;\n')
    P.append('static bool wG[4096],mB[1024];\n')
    P.append('static int my_bird_ids[4], n_my_birds = 0;\n')  # NEW: Track bird identities
    P.append('struct Bot{int id,ow,bx[200],by[200],ln;\n')
    P.append('int hx()const{return bx[0];}int hy()const{return by[0];}')
    P.append('int fc()const{if(ln<2)return-1;int dx=bx[0]-bx[1],dy=by[0]-by[1];for(int d=0;d<4;d++)if(DX[d]==dx&&DY[d]==dy)return d;return-1;}};\n')
    P.append('static Bot bt[16];\n')
    # NEW: Safe parser that ignores "DEAD" strings and handles negative coordinates
    P.append('static void pB(Bot&b,const string&s){b.ln=0;if(s.find("DEAD")!=string::npos)return;const char*p=s.c_str();while(*p){int x=0,y=0,sx=1,sy=1;if(*p==\'-\'){sx=-1;p++;}while(*p>=\'0\'&&*p<=\'9\')x=x*10+(*p++-\'0\');x*=sx;if(*p==\',\')p++;if(*p==\'-\'){sy=-1;p++;}while(*p>=\'0\'&&*p<=\'9\')y=y*10+(*p++-\'0\');y*=sy;b.bx[b.ln]=x;b.by[b.ln]=y;b.ln++;if(*p==\':\')p++;else if(*p)p++;}}\n')
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
    P.append('static const float SC[]={' + ','.join(f'{s:.5g}f' for _, _, s, _ in weight_layers) + '};\n')
    P.append('static const int WO[]={' + ','.join(str(w_offsets[n][0]) for n, _, _, _ in weight_layers) + '};\n')
    P.append('static const int WZ[]={' + ','.join(str(w_offsets[n][1]) for n, _, _, _ in weight_layers) + '};\n')
    P.append('static const int BO[]={' + ','.join(str(b_offsets[n][0]) for n, _, _ in bias_layers) + '};\n')

    # ---- Weight storage ----
    # comp_len = size of the compressed (zlib) blob in bytes (padded to multiple of 4 by ascii85)
    # orig_blob_len = size of the uncompressed blob
    raw_buf_size = ((orig_blob_len + 3) // 4) * 4  # ascii85 decodes to multiple of 4
    P.append(f'static float AW[{total_int8 + 10}],AB[{total_bias + 10}];\n')
    P.append(f'static float*Wp[{len(weight_layers)}],*Bp[{len(bias_layers)}];\n')

    # ---- Decode function ----
    P.append('static void decW(){\n')
    P.append(f'static unsigned char r[{raw_buf_size + 4}];\n')
    P.append('const char*p=ED;int n=0;\n')
    P.append('while(*p){unsigned v=0;for(int i=0;i<5;i++){v=v*85+D[(unsigned char)*p];p++;}r[n++]=(v>>24)&0xFF;r[n++]=(v>>16)&0xFF;r[n++]=(v>>8)&0xFF;r[n++]=v&0xFF;}\n')
    P.append(f'for(int i=0;i<{total_int8};i++)AW[i]=((int)r[i]-128);\n')
    P.append(f'for(int i=0;i<{len(weight_layers)};i++){{Wp[i]=AW+WO[i];for(int j=0;j<WZ[i];j++)Wp[i][j]*=SC[i];}}\n')
    # Tiny inline lambda to convert Float16 bytes to Float32 safely
    P.append('auto h2f=[](unsigned short h){uint32_t e=(h>>10)&31,m=h&1023,s=(h&32768)<<16;if(!e)return 0.0f;uint32_t f=s|((e+112)<<23)|(m<<13);float v;memcpy(&v,&f,4);return v;};\n')
    # Read two bytes at a time (handles alignment safety) and decode
    P.append(f'for(int i=0;i<{total_bias};i++){{unsigned short h=r[{bias_start}+i*2]|(r[{bias_start}+i*2+1]<<8);AB[i]=h2f(h);}}\n')
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
    P.append('int bp;cin>>bp;cin.ignore();\n')
    
    # NEW: Lock in initial IDs during startup
    P.append('for(int i=0;i<bp;i++){int id;cin>>id;cin.ignore();mB[id]=true;my_bird_ids[n_my_birds++]=id;}\n')
    
    P.append('for(int i=0;i<bp;i++){int id;cin>>id;cin.ignore();}\n')
    P.append('while(true){cin>>nA;cin.ignore();if(cin.eof())break;\n')
    P.append('for(int i=0;i<nA;i++){cin>>aX[i]>>aY[i];cin.ignore();}\n')
    P.append('cin>>nB;cin.ignore();for(int i=0;i<nB;i++){string bd;cin>>bt[i].id>>bd;cin.ignore();pB(bt[i],bd);bt[i].ow=mB[bt[i].id]?0:1;}\n')
    P.append('exO();fwd();\n')
    
    # NEW: Only queue alive birds to prevent crashes
    P.append('int mi[4],nm=0;for(int i=0;i<nB;i++)if(mB[bt[i].id]&&bt[i].ln>0)mi[nm++]=i;bool fr=true;\n')

    # NEW: The Brain-Swap fix (matches nn_idx to initial ID)
    P.append('for(int bi=0;bi<nm;bi++){Bot&bot=bt[mi[bi]];int nn_idx=0;for(int j=0;j<n_my_birds;j++){if(my_bird_ids[j]==bot.id){nn_idx=j;break;}}float*lg=al+nn_idx*4;int bd=-1,fc=bot.fc();if(fc!=-1)bd=(fc+2)%4;\n')
    
    # NEW: The C++ Ceiling fix (ny < 0)
    P.append('int hx=bot.hx(),hy=bot.hy();auto is_fatal=[&](int d){int nx=hx+DX[d],ny=hy+DY[d];if(nx<0||nx>=W||ny<0||ny>=H||wG[ny*W+nx])return true;for(int b=0;b<nB;b++)for(int j=0;j<bt[b].ln;j++)if(bt[b].bx[j]==nx&&bt[b].by[j]==ny)return true;return false;};\n')
    
    P.append('int best=-1;float bS=-1e9;for(int d=0;d<4;d++){if(d==bd||is_fatal(d))continue;if(best==-1||lg[d]>bS){bS=lg[d];best=d;}}if(best==-1){bS=-1e9;for(int d=0;d<4;d++){if(d==bd)continue;if(best==-1||lg[d]>bS){bS=lg[d];best=d;}}}if(best==-1)best=0;\n')
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