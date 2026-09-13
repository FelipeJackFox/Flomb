#include <cstdint>
#include <cmath>

template<class T> void edge_grad(const int32_t* ptr, const int32_t* col,
    const T* x, const T* dy, T* out, int begin, int end, int batch) {
    for (int r=begin; r<end; ++r) {
        const T* row=dy+(int64_t)r*batch;
        bool active=false;
        for(int b=0;b<batch;++b) if(row[b]!=0){active=true;break;}
        if(!active){for(int e=ptr[r];e<ptr[r+1];++e) out[e]=0;continue;}
        for (int e=ptr[r]; e<ptr[r+1]; ++e) {
            const T* source=x+(int64_t)col[e]*batch;
            T sum=0;
            for (int b=0; b<batch; ++b) sum+=row[b]*source[b];
            out[e]=sum;
        }
    }
}
extern "C" void edge_grad_f32(const int32_t* p,const int32_t* c,const float* x,
    const float* dy,float* out,int a,int b,int batch) {edge_grad(p,c,x,dy,out,a,b,batch);}
extern "C" void edge_grad_f64(const int32_t* p,const int32_t* c,const double* x,
    const double* dy,double* out,int a,int b,int batch) {edge_grad(p,c,x,dy,out,a,b,batch);}

static uint64_t next_random(uint64_t& x) {x^=x<<13;x^=x>>7;x^=x<<17;return x;}
extern "C" int64_t rewire(const int32_t* ptr,int32_t* col,const int32_t* rows,
    const float* signs,int64_t edges,int64_t attempts,uint64_t seed) {
    int64_t accepted=0;
    for (int64_t k=0;k<attempts;++k) {
        int a=next_random(seed)%edges,b=next_random(seed)%edges;
        int ra=rows[a],rb=rows[b],ca=col[a],cb=col[b];
        if(ra==rb || ca==cb || signs[ca]!=signs[cb]) continue;
        // Preserve existing self-edges and do not introduce new ones.
        if(ra==ca || rb==cb || ra==cb || rb==ca) continue;
        bool duplicate=false;
        for(int e=ptr[ra];e<ptr[ra+1];++e) if(col[e]==cb){duplicate=true;break;}
        if(duplicate) continue;
        for(int e=ptr[rb];e<ptr[rb+1];++e) if(col[e]==ca){duplicate=true;break;}
        if(duplicate) continue;
        col[a]=cb;col[b]=ca;++accepted;
    }
    return accepted;
}
