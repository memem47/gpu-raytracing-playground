import cupy as cp

gpu_kernel = cp.RawKernel(r'''
extern "C" __global__
void raytrace(unsigned char* img,
              int width, int height,
              float aspect, float radius)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int i = idx % width;
    int j = idx / width;
                          
    float u = (i + 0.5f) / width * 2.f - 1.f;
    float v = (j + 0.5f) / height * 2.f - 1.f;

    // ray direction
    float3 dir = make_float3(u*aspect, -v, -1.f);
    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;

    // sphere at (0,0,-1)
    float3 oc = make_float3(0.f,0.f,1.f);
                          
    float b = dir.x*oc.x + dir.y*oc.y + dir.z*oc.z;
    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - radius*radius;
    float disc = b*b - c;
                          
    int offset = idx * 3;
    unsigned char r = 0, g = 0, b_out = 0;
    if (disc >= 0.0f) r = 255;

    img[offset]     = r;
    img[offset + 1] = g;
    img[offset + 2] = b_out;
}
''', 'raytrace')

def render(width=800, height=450):
    aspect = width/height
    img = cp.zeros((height*width*3,), dtype=cp.uint8)
    threads = 256
    blocks = (width * height + threads - 1)//threads
    gpu_kernel((blocks,), (threads,),
               (img, width, height, cp.float32(aspect), cp.float32(0.5)))
    return cp.asnumpy(img.reshape(height, width, 3))

if __name__ == "__main__":
    from PIL import Image
    img = render()
    Image.fromarray(img).save("sphere_gpu.png")
