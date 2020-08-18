#include "util.cuh"

#include "Camera.cuh"

#include "Hittable.cuh"
#include "Material.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Color color(const Ray& r, Hittable** world, curandState* local_rand_state) {

    Ray cur_ray = r;
    Color cur_attenuation = Color(1.0f, 1.0f, 1.0f);

    for(int i = 0; i < 50; i++){

        hit_record rec;

        if((*world)->hit(cur_ray, 0.001f, infinity, rec)){

            Ray scattered;
            Color attenuation;

            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)){

                cur_attenuation *= attenuation;
                cur_ray = scattered;

            } else {

                return Color(0.0f, 0.0f, 0.0f);

            }



        } else {

            Vector3 unit_direction = unit_vector(cur_ray.direction());

            float t = 0.5f*(unit_direction.y() + 1.0f);

            Vector3 c = (1.0f-t)*Color(1.0f, 1.0f, 1.0f) + t*Color(0.5f, 0.7f, 1.0f);

            return cur_attenuation * c;

        }


    }


    return Vector3(0.0f, 0.0f, 0.0f);



}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Hittable** d_world, Hittable** d_list, int list_size, Camera** d_camera, int nx, int ny){

    if (threadIdx.x == 0 && blockIdx.x == 0){

        *(d_list) = new Sphere(Point3(0.0f, 0.0f, -1.0f), 0.5f, new Lambertian(Color(0.1f, 0.2f, 0.5f)));

        *(d_list + 1) = new Sphere(Point3(0.0f,-1000.5f,-1.0f), 1000.0f, new Lambertian(Color(0.5f, 0.5f, 0.5f)));

        *(d_list + 2) = new Sphere(Point3(1.0f, 0.0f, -1.0f), 0.5f, new Metal(Color(0.8f, 0.6f, 0.2f), 0.0f));

        *(d_list + 3) = new Sphere(Point3(-1.0f, 0.0f, -1.0f), 0.5f, new Dielectric(1.5f));

        *(d_list + 4) = new Sphere(Point3(-1.0f, 0.0f, -1.0f), -0.45f, new Dielectric(1.5f));


        *d_world = new HittableList(d_list, list_size);

        Vector3 lookfrom(3.0f,3.0f,2.0f);
        Vector3 lookat(0.0f,0.0f,-1.0f);

        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 0.1f;

        *d_camera = new Camera(lookfrom, lookat, Vector3(0.0f, 1.0f, 0.0f), 20.0f, float(nx)/float(ny), aperture, dist_to_focus);

    }

}

__global__ void destroy_world(Hittable** d_world, Hittable** d_list, int list_size, Camera** d_camera){

    if (threadIdx.x == 0 && blockIdx.x == 0){

        for(int i = 0; i < list_size; i++){

            delete *(d_list + i);

        }

        delete *d_world;

        delete *d_camera;

    }

}

__global__ void render_init(int max_x, int max_y, curandState* rand_state){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)){

        return;

    };

    int pixel_index = j*max_x + i;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);

}

__global__ void render(Color* frame_buffer,
                       int max_x,
                       int max_y,
                       int ns,
                       Camera** cam,
                       Hittable** world,
                       curandState* rand_state) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)){

        return;

    };

    int pixel_index = j*max_x + i;

    curandState local_rand_state = rand_state[pixel_index];

    Color col(0.0f, 0.0f, 0.0f);

    //Antialiasing by taking the avg of the samples
    for(int s = 0; s < ns; s++){

        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

        Ray r = (*cam)->get_ray(u,v, &local_rand_state);

        col += color(r, world, &local_rand_state);

    }


    frame_buffer[pixel_index] = col/float(ns);

}

int main() {

    int nx = 1200;
    int ny = 600;
    int ns = 100;
    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;

    // allocate FB
    Vector3* fb;
    size_t fb_size = num_pixels*sizeof(Vector3);
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));

    Hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hittable*)));

    Hittable** d_list;
    int list_size = 5;
    checkCudaErrors(cudaMalloc((void**)&d_list, list_size*sizeof(Hittable*)));

    create_world<<<1,1>>>(d_world, d_list, list_size, d_camera, nx, ny);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    render_init<<<blocks,threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks,threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();

    destroy_world<<<1,1>>>(d_world, d_list, list_size, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    write_buffer(fb, nx, ny);

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_rand_state));

    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_camera));

    cudaDeviceReset();
}
