/*
 *  Simple OpenCL demo program
 *
 *  Copyright (C) 2009  Clifford Wolf <clifford@clifford.at>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  gcc -o cldemo -std=gnu99 -Wall -I/usr/include/nvidia-current cldemo.c -lOpenCL
 *
 */
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <functional>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <set>
#include <list>

#include "kernels.inl"
#include "kernels2.inl"

#include "va/va.h"
#include "va/va_drm.h"
//#include "va/va_x11.h"
#include "va_ext.h"

#include "args.hxx"
#include "helper.h"

// "/dev/dri/card0"
#define DRI_NODE "/dev/dri/renderD128"

class _Config_
{
public:
	int			Width;
	int			Height;
	int 		thread_cnt;
	int 		numberTest;
	int 		TestCode;
	bool 		bUseOCLBinCache;
	bool 		bStopTest = false;


	VADisplay display(const char * dev_node = DRI_NODE){

		if(va_display != NULL) return va_display;

	    VAStatus va_st;
	    int fd_dri = open(dev_node, O_RDWR);
	    if (fd_dri < 0){
	        printf("open(%s) failed: errno is %s\n", dev_node, strerror(errno));
	        exit(-1);
	    }

	    va_display = vaGetDisplayDRM(fd_dri);
	    if (!va_display) {
	        close(fd_dri);
	        fd_dri = -1;
	        printf("vaGetDisplayDRM(%s) failed: errno is %s\n", dev_node, strerror(errno));
	        exit(-1);
	    }

	    va_st = vaInitialize(va_display, &major_version, &minor_version);
	    if (VA_STATUS_SUCCESS != va_st) {
	        close(fd_dri);
	        fd_dri = -1;
	        printf("vaInitialize(%s) failed: return code is 0x%x\n", dev_node, va_st);
	        exit(-1);
	    }
	    printf("vaInitialize(%s) version=%d.%d\n", dev_node, major_version,  minor_version);

	    return va_display;
	}

	~_Config_(){
		if(fd_dri >= 0)
			close(fd_dri);
	}
private:
	int 		fd_dri = -1;
	VADisplay 	va_display = NULL;
	int 		major_version;
	int 		minor_version;
} gConfig;

static VADisplay getVADisplay(void){ return gConfig.display();}
static void abortTest(void){ gConfig.bStopTest=true; abort(); }


#define NUM_DATA (1280*720*3)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       gConfig.bStopTest=true; 													\
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })
   
#define EXIT_ON_COND(cond, ...) do{\
    if(cond){\
        printf("error on %s:%d\n",__FILE__, __LINE__);\
        printf(__VA_ARGS__);\
        gConfig.bStopTest=true; \
        abort();  \
    }}while(0);



void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}


cl_program load_source(const char * psrc, cl_context &context, const cl_device_id device)
{
    cl_int status = 0;
    cl_program program = 0;
    bool cache_hit = false;
    std::string cache_filename;

    if(gConfig.bUseOCLBinCache){
    	std::size_t str_hash = std::hash<std::string>{}(std::string(psrc));
    	cache_filename = std::to_string(str_hash)+".clbin";
		std::ifstream cachefile(cache_filename, std::ios::ate |std::ios::binary);
		if (cachefile.is_open()) {
			std::streamsize size = cachefile.tellg();
			cachefile.seekg(0, std::ios::beg);

			std::vector<char> buffer(size);
			if (cachefile.read(buffer.data(), size))
			{
				cache_hit = true;
				size_t lengths[1]={size};
				const unsigned char *binaries[1] = { (unsigned char *)&buffer[0]};
				cl_int binary_status[1];
				program = clCreateProgramWithBinary(context, 1, &device,lengths, binaries, binary_status, &status); CL_CHECK(status);
				printf("clCreateProgramWithBinary from cached binary file %s\n", cache_filename.c_str());
			}
		}
    }

    if(!program)
    	program = clCreateProgramWithSource(context, 1, &psrc, NULL, &status); CL_CHECK(status);

	status = clBuildProgram(program, 1, &device, "", NULL, NULL);
	if (status != CL_SUCCESS) {
		char buffer[10240];
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", status == CL_SUCCESS?buffer:"clGetProgramBuildInfo() failed");
		abort();
	}

	if(gConfig.bUseOCLBinCache){
		if(!cache_hit){
			std::vector<char> buf;
			size_t sz = 0;
			CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(sz), &sz, NULL));
			buf.resize(sz);
			uint8_t* ptr = (uint8_t*)&buf[0];
			CL_CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(ptr), &ptr, NULL));

			std::ofstream fout(cache_filename, std::ios::out | std::ios::binary);
			fout.write((char*)&buf[0], buf.size() * sizeof(buf[0]));
			fout.close();
		}
	}

    //status = clUnloadCompiler();
    //EXIT_ON_COND(status != CL_SUCCESS, "openCL: clUnloadCompiler failed");
    
    return program;
}

void testOCL_RunKernel(cl_context &context, const cl_device_id *devices)
{
    cl_int clerr = CL_INVALID_VALUE;
    cl_program program = load_source((const char *)kernels_cl, context, devices[0]);
    cl_mem input_buffer;
    input_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*NUM_DATA, NULL, &_err));

    cl_mem output_buffer;
    char * pout_data = (char*)malloc(sizeof(int)*NUM_DATA + 1024);
    char * pout_align = pout_data+1;// + (((unsigned long long)pout_data) & 63);

    printf(">>> Output buffer: %p => %p\n",pout_data, pout_align);

    //output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*NUM_DATA, NULL, &_err));
    output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_USE_HOST_PTR, sizeof(int)*NUM_DATA, pout_align, &_err));
    //output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, sizeof(int)*NUM_DATA,NULL, &_err));
    //output_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR|CL_MEM_COPY_HOST_PTR, sizeof(int)*NUM_DATA,pout_align, &_err));

    int factor = 2;

    cl_kernel kernel;
    kernel = CL_CHECK_ERR(clCreateKernel(program, "simple_demo", &_err));
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(input_buffer), &input_buffer));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(output_buffer), &output_buffer));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(factor), &factor));

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, devices[0], 0, &clerr);
    EXIT_ON_COND(clerr != CL_SUCCESS, "clCreateCommandQueue failed: 0x%x\n", clerr);
    
    for (int i=0; i<NUM_DATA; i++) {
        CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, i*sizeof(int), sizeof(int), &i, 0, NULL, NULL));
    }

    cl_event kernel_completion;
    size_t global_work_size[1] = { NUM_DATA };
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &kernel_completion));
    CL_CHECK(clWaitForEvents(1, &kernel_completion));
    CL_CHECK(clReleaseEvent(kernel_completion));

    printf("Result:");
    for (int i=0; i<NUM_DATA && i<100; i++) {
        int data;
        CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, i*sizeof(int), sizeof(int), &data, 0, NULL, NULL));
        printf(" %d", data);
    }
    printf("\n");

    CL_CHECK(clReleaseMemObject(input_buffer));
    CL_CHECK(clReleaseMemObject(output_buffer));

    CL_CHECK(clReleaseKernel(kernel));
    CL_CHECK(clReleaseProgram(program));
    free(pout_data);
}

int testOCL_basic_info()
{
    cl_platform_id platforms[100];
    cl_uint platforms_n = 0;
    CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

    printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
    for (int i=0; i<platforms_n; i++)
    {
        char buffer[10240];
        printf("  -- %d --\n", i);
        CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
        printf("  PROFILE = %s\n", buffer);
        CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
        printf("  VERSION = %s\n", buffer);
        CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
        printf("  NAME = %s\n", buffer);
        CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
        printf("  VENDOR = %s\n", buffer);
        CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
        printf("  EXTENSIONS = %s\n", buffer);

        cl_device_id devices[100];
        cl_uint devices_n = 0;
        // CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
        CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));

        printf("=== %d OpenCL device(s) found on platform %d:\n", devices_n, i);
        for (int i=0; i<devices_n; i++)
        {
            char buffer[10240];
            cl_uint buf_uint;
            cl_ulong buf_ulong;
            printf("  -- %d --\n", i);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
            printf("    DEVICE_NAME = %s\n", buffer);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
            printf("    DEVICE_VENDOR = %s\n", buffer);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
            printf("    DEVICE_VERSION = %s\n", buffer);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
            printf("    DRIVER_VERSION = %s\n", buffer);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
            printf("    DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
            printf("    DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
            printf("    DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
            CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL));
            printf("    DEVICE_EXTENSIONS = %s\n", buffer);
        }
    }

    if (platforms_n == 0)
        return 1;

    cl_device_id devices[100];
    cl_uint devices_n = 0;
    // CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));
    if (devices_n == 0)
        return 1;

    cl_context context;
    context = CL_CHECK_ERR(clCreateContext(NULL, 1, devices, &pfn_notify, NULL, &_err));

    testOCL_RunKernel(context, devices);

    CL_CHECK(clReleaseContext(context));

    printf("============testOCL_basic_info() END===============\n");
    return 0;
}







#ifdef USE_OPENCV
void assert_matBGR24(cv::Mat m, 
                    int x0, int x1, int y0, int y1, int B=0xff,int G=0xe3, int R=0x00)
{
    bool bfail = false;
    for(int y=y0;y<y1;y++)
	for (int x=x0;x<x1;x++){
		cv::Vec3b v = m.at<cv::Vec3b>(y,x);
		if(v[0]!= B || v[1]!=G || v[2] != R){
    		printf(" ERROR in assert_matBGR24 @(%d,%d)  (%02X,%02X,%02X)\n",x, y,v[0],v[1],v[2]);
    		bfail = true;
    	}
	}
	if(bfail) exit(1);
}

void VASurfaceTest_CV(VASurfaceID surface, int W, int H,  int channel)
{
	cv::UMat mRGBOrig(H,W,CV_8UC3);
	cv::va_intel::convertFromVASurface(gConfig.display(), surface, cv::Size(W, H), mRGBOrig);

	cv::UMat mRGBResize(H,W,CV_8UC3);
	cv::resize(mRGBOrig, mRGBResize, cv::Size(W,H),0,0, cv::INTER_NEAREST);

	assert_matBGR24(mRGBResize.getMat(0), 0,16,0,16, 0xff, 0xe3, 0x00);
}
#endif



#define VERBOSE 0



#if 0


int testOCL_VA_Intel()
{
    cv::va_intel::ocl::initializeContextFromVA(gConfig.display(), true);

    cv::ocl::Context& ctx = cv::ocl::Context::getDefault();
    cl_context context = (cl_context)ctx.ptr();
    size_t n = ctx.ndevices();
    if(n != 1){
    	printf("Context has %d devices ?\n", n);
    	exit(2);
    }
    cl_device_id device =  (cl_device_id)(ctx.device(0).ptr());

    //this do not help
    //load_dummy_binary_prog(context, &device);

    //this is the cue
    load_source("__kernel void test(){}\n",context, device);

    //cv::ocl::ProgramSource cvocl_prog_src("__kernel void test(){}\n");
    //cv::String buildflags, builderr;
    //cv::ocl::Program cvocl_prog(cvocl_prog_src, buildflags, builderr);

    printf("initializeContextFromVA success!\n");

    auto testor_ocvfunc1 = [&](int channel){
        VASurfaceID surfaces[8];
        int surfaces_num = sizeof(surfaces)/sizeof(surfaces[0]);
        int Width = 1280;
        int Height = 736;
        createVASurfaces(surfaces, surfaces_num, Width, Height, 0xA0, 0xD0, 0x10,0);

        printf("vaCreateSurfaces success!\n");


        while(1){
        	printf("INQUEUE=%d CNT = %d,%d,%d,   %d\n",getCNT(1)-getCNT(0), getCNT(0),getCNT(1),getCNT(2),getCNT(3));
            for(int i=0;i<surfaces_num;i++){
                //{VASurfaceLocker vl(display, surfaces[i]);vl.brief();}
                printf("[%02d] convertFromVASurface %d  ...\n", channel, i);

                VASurfaceTest_CV(surfaces[0], Width, Height, channel);
            }
        }
    };
    
    std::vector<std::thread> test_ths;
    for(int t=0;t<gConfig.thread_cnt;t++) test_ths.emplace_back(testor_ocvfunc1, t);
    for(int t=0;t<gConfig.thread_cnt;t++) test_ths[t].join();
}





#else



class OCLBufferPool
{
public:
	cl_context m_context;
	int m_buffer_size;
	int m_pool_limit;

	OCLBufferPool(cl_context clc, int size, int pool_limit = 128*1024*1024):
		m_context(clc),
		m_buffer_size(size){
		m_pool_limit = pool_limit;
	}

	std::vector<cl_mem> m_reserve;
	std::set<cl_mem> m_used;

	std::mutex m_mpool;

	cl_mem allocate(void){
		std::lock_guard<std::mutex> lk(m_mpool);
		cl_int status;
		cl_mem clBuffer;

		if(!m_reserve.empty()){
			clBuffer = m_reserve.back();
			m_reserve.pop_back();
		}else{
			clBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_buffer_size, NULL, &status); CL_CHECK(status);
			//printf("clCreateBuffer %p  m_used=%d\n", clBuffer, m_used.size());
		}

		m_used.insert(clBuffer);
		return clBuffer;
	}

	void deallocate(cl_mem buff){
		std::lock_guard<std::mutex> lk(m_mpool);

		auto it = m_used.find(buff);
		if(it == m_used.end()){
			printf("OCLBufferPool::deallocate(%p) Free unknow buff\n", buff);
			exit(1);
		}

		m_used.erase(it);
		m_reserve.push_back(buff);
	}

	void checkReserveSize(){
		std::lock_guard<std::mutex> lk(m_mpool);

		if(m_reserve.size()*m_buffer_size > m_pool_limit){
			std::ostringstream ss;
			ss << "clReleaseMemObject:";
			while(m_reserve.size()*m_buffer_size > m_pool_limit){
				cl_mem b = m_reserve.back();
				ss << " " << b;
				CL_CHECK(clReleaseMemObject(b));
				m_reserve.pop_back();
			}
			ss << "\n";
			//printf("%s\n", ss.str().c_str());
		}
	}
};

struct OCLThreadContext{
	cl_context 						context;
	cl_device_id 					device;
	cl_command_queue 				queue;
	std::vector<cl_program> 		programs;
	std::shared_ptr<OCLBufferPool>  localBufferPool;
	std::shared_ptr<OCLBufferPool>  globalBufferPool;

	OCLThreadContext(VAOCLContext &v){
		context = v.context;
		device = v.device;
		if(v.useOpenCV){
#ifndef USE_OPENCV
		NO_OPENCV_ERROR();
#else
			queue = (cl_command_queue)cv::ocl::Queue::getDefault().ptr();
#endif
		}else{
			cl_int status;
			queue = clCreateCommandQueue(context, device, 0, &status);  CL_CHECK(status);
		}
	}

	void add_prog(const char * src){
		cl_program prog = load_source(src, context, device);
		programs.push_back(prog);

        //cv::ocl::ProgramSource cvocl_prog_src(kernels_cl);
        //cv::String buildflags, builderr;
        //cv::ocl::Program cvocl_prog(cvocl_prog_src, buildflags, builderr);
        //if(builderr.length() > 0){
		// 	printf("Build Error: %s\n", builderr.c_str());
        //}
	}

	void assert_clmem(cl_mem clbuff, int W, int H,
					 int x0, int x1, int y0, int y1,
					 int valB,int valG,int valR, int xIMPRINT, int yIMPRINT, int sx=1, int sy=1)
	{
		cl_int status;
		uint8_t * pData = (uint8_t*)clEnqueueMapBuffer(queue, clbuff, CL_TRUE, CL_MAP_READ, 0, sizeof(uint8_t)*W*H*3, 0,NULL, NULL, &status); CL_CHECK(status);
		for(int y=y0;y<H && y<y1;y++){
			for(int x=x0;x<W && x<x1*3;x+=3){
				uint8_t &b = pData[y*W*3 + x + 0];
				uint8_t &g = pData[y*W*3 + x + 1];
				uint8_t &r = pData[y*W*3 + x + 2];
				int addY = y*sy*yIMPRINT + (x/3)*sx*xIMPRINT;
				int addUV = ((y*sy)>>1)*yIMPRINT + (((x/3)*sx)>>1)*xIMPRINT;

				int expB = (valB+addY) & 0xFF;
				int expG = (valG+addUV) & 0xFF;
				int expR = (valR+addUV) & 0xFF;

				if(b != expB || g != expG || r != expR){
					printf("\nERROR in assert_clmem: @(%d,%d) in image(%dx%d)  BGR=(0x%X,0x%X,0x%X) expecting(0x%X,0x%X,0x%X)  clbuff=%p\n",
							x, y, W, H, b,g,r,expB, expG, expR, clbuff);
					exit(1);
				}
			}
		}
		CL_CHECK(clEnqueueUnmapMemObject(queue, clbuff, pData, 0, NULL, NULL));
	}
};


void VASurfaceTest(OCLThreadContext &ctx, VASurfaceID surface, int W, int H, int channel);

#define VALUE_Y 0xA0
#define VALUE_U 0xD0
#define VALUE_V 0x10

//#define VALUE_B 0xFF
//#define VALUE_G 0xE3
//#define VALUE_R 0x00

#define VALUE_B 0xA0
#define VALUE_G 0xD0
#define VALUE_R 0x10

#define X_IMPRINT 1
#define Y_IMPRINT 1

int testOCL_VA_Intel()
{
	int Width = gConfig.Width;
	int Height = gConfig.Height;
    bool bUseOpenCV = false;
    VAOCLContext vaOCL(gConfig.display(), bUseOpenCV);
    std::shared_ptr<OCLBufferPool>  pGlobalBufferPool(new OCLBufferPool(vaOCL.context, Width*Height*3, 128*1024*1024));

	printf("Created context %p matches VADisplay!\n", vaOCL.context);
	//testOCL_RunKernel(context, &device);

	auto testor_func1 = [&](int channel){
		VASurfaceID surfaces[1];
		cl_int  status;
		int surfaces_num = sizeof(surfaces)/sizeof(surfaces[0]);
		createVASurfaces(surfaces, surfaces_num, Width, Height, VALUE_Y,VALUE_U,VALUE_V, X_IMPRINT, Y_IMPRINT);

		OCLThreadContext ctx(vaOCL);

		//ctx.localBufferPool.reset(new OCLBufferPool(ctx.context, Width*Height*3, 4*1024*1024));
		ctx.globalBufferPool = pGlobalBufferPool;

		ctx.add_prog((const char *)kernels_cl);
		//ctx.add_prog((const char *)kernels2_cl);

		while(!gConfig.bStopTest && gConfig.numberTest > 0){
			for(int i=0;i<surfaces_num;i++){
				if(VERBOSE){VASurfaceLocker vl(surfaces[i]); vl.brief();}
				printf("%02d.%02d ", channel, i); fflush(stdout);
				VASurfaceTest(ctx, surfaces[i], Width, Height, channel);
				//VASurfaceTest_CV(surfaces[0], Width, Height, channel);

				gConfig.numberTest --;
			}
		}
	};

	std::vector<std::thread> test_ths;
	for(int t=0;t<gConfig.thread_cnt;t++) test_ths.emplace_back(testor_func1, t);
	for(int t=0;t<gConfig.thread_cnt;t++) test_ths[t].join();

	printf("\n\n\n  ****   Congratulations! Test done w/o Error   ****\n\n");
    return 0;
}


extern "C" void CL_CALLBACK OCLKernelCleanupCallback(cl_event e, cl_int status, void *p);

class OCLKernel{
public:
	OCLKernel(cl_program program, const char * kernel_name, OCLThreadContext  &ctx): _ctx(ctx){
		cl_int status;
		_kernel = clCreateKernel(program, kernel_name, &status); CL_CHECK(status);
		_argi = 0;
	}
	~OCLKernel(){
        for(auto clbuff: _vCLbuffs){
        	//_ctx.localBufferPool->deallocate(clbuff);
        	_ctx.globalBufferPool->deallocate(clbuff);
        }

        CL_CHECK(clReleaseKernel(_kernel));
	}

	template<class T>
	void addArg(T * p){ CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(T), p)); }

	std::vector<cl_mem> _vCLbuffs;
	int 				_argi;
	cl_kernel 			_kernel;
	OCLThreadContext  & _ctx;

	void addArgCLMem(cl_mem clBuff, int step, int offset, int H, int W, bool bAutoRelease){

		if(bAutoRelease)
			_vCLbuffs.push_back(clBuff);

		CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(clBuff), &clBuff));
		CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(step), &step));
		CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(offset), &offset));
		CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(H), &H));
		CL_CHECK(clSetKernelArg(_kernel, _argi++, sizeof(W), &W));
	}

	void run(cl_command_queue &queue, int cols, int rows){
		cl_event asyncEvent;
	    size_t global_work_size[2] = { (size_t)cols, (size_t)rows };
	    CL_CHECK(clEnqueueNDRangeKernel(queue, _kernel, 2, NULL, global_work_size, NULL, 0, NULL, &asyncEvent));
	    CL_CHECK(clSetEventCallback(asyncEvent, CL_COMPLETE, OCLKernelCleanupCallback, this));
	    CL_CHECK(clReleaseEvent(asyncEvent));
	}
};

extern "C" void CL_CALLBACK OCLKernelCleanupCallback(cl_event e, cl_int status, void *p)
{
    if(status != CL_COMPLETE) {
        printf("oclCleanupCallback status=%d\n", status);
        exit(1);
    }
    OCLKernel * pk = (OCLKernel *)p;
    delete pk;
}





void VASurfaceTest(OCLThreadContext &ctx, VASurfaceID surface, int W, int H,  int channel)
{
	int bgrStep = W*3;
	cl_int status;

	//cl_mem clBuffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(uint8_t)*W*H*3, NULL, &status); CL_CHECK(status);

	//gOCLBufferPool1->checkReserveSize();
	//cl_mem clBuffer = gOCLBufferPool1->allocate();

	ctx.globalBufferPool->checkReserveSize();

	cl_mem clBuffer = ctx.globalBufferPool->allocate();

    cl_mem clImageY = clCreateFromVA_APIMediaSurfaceINTEL(ctx.context, CL_MEM_READ_WRITE, &surface, 0, &status); CL_CHECK(status);
    cl_mem clImageUV = clCreateFromVA_APIMediaSurfaceINTEL(ctx.context, CL_MEM_READ_WRITE, &surface, 1, &status); CL_CHECK(status);

    //printCLNV12(ctx.queue, clImageY, clImageUV);

    cl_mem images[2] = { clImageY, clImageUV };
    status = clEnqueueAcquireVA_APIMediaSurfacesINTEL(ctx.queue, 2, images, 0, NULL, NULL); CL_CHECK(status);

    //printCLNV12(queue, clImageY, clImageUV);

	//const char * kernel_name = "YUV2BGR_NV12_8u";
	const char * kernel_name = "YUV2BGR_NV12_8u_DEBUG";

	cl_kernel kernel = clCreateKernel((cl_program)ctx.programs[0], kernel_name, &status); CL_CHECK(status);

	CL_CHECK(clSetKernelArg(kernel, 0, sizeof(clImageY), &clImageY));
	CL_CHECK(clSetKernelArg(kernel, 1, sizeof(clImageUV), &clImageUV));
	CL_CHECK(clSetKernelArg(kernel, 2, sizeof(clBuffer), &clBuffer));
	CL_CHECK(clSetKernelArg(kernel, 3, sizeof(bgrStep), &bgrStep));
	CL_CHECK(clSetKernelArg(kernel, 4, sizeof(W), &W));
	CL_CHECK(clSetKernelArg(kernel, 5, sizeof(H), &H));

	size_t global_work_size[2] = { (size_t)W, (size_t)H };
	CL_CHECK(clEnqueueNDRangeKernel(ctx.queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
	//CL_CHECK(clWaitForEvents(1, &asyncEvent1));
	//CL_CHECK(clReleaseEvent(asyncEvent1));

	if(0)
	{
		printf("clEnqueueReadBuffer %dx%d region from clBuffer:\n", NY, NX);
		for(int y=0;y<NY;y++){
			uint8_t data[NX*3];
			int offset = y*bgrStep;
			CL_CHECK(  clEnqueueReadBuffer(ctx.queue, clBuffer, CL_TRUE, offset, sizeof(data), data,0, NULL, NULL));

			printf("\t");
			for(int x=0;x<NX;x++){
				printf("(%02X,%02X,%02X),", data[x*3+0], data[x*3+1], data[x*3+2]);
			}
			printf("\n");
		}
	}

	CL_CHECK(clEnqueueReleaseVA_APIMediaSurfacesINTEL(ctx.queue, 2, images, 0, NULL, NULL));
	CL_CHECK(clFinish(ctx.queue));
	CL_CHECK(clReleaseKernel(kernel));
	CL_CHECK(clReleaseMemObject(clImageY));
	CL_CHECK(clReleaseMemObject(clImageUV));

	if(gConfig.TestCode == 0){
		ctx.assert_clmem(clBuffer, W, H, 0,W,0,H, 		VALUE_B, VALUE_G, VALUE_R, X_IMPRINT, Y_IMPRINT);
		ctx.globalBufferPool->deallocate(clBuffer);
		return;
	}

	//resize the result and check result
	float ifx = 4;
	float ify = 4;
	int W2=W/ifx;
	int H2=H/ify;
	int srcStep = W*3;
	int srcOffset = 0;
	int dstStep = W2*3;
	int dstOffset = 0;

#if 1
	//cv::UMat mRGBResized(H2,W2,CV_8UC3);
	//cl_mem clBufferResize = (cl_mem)mRGBResized.handle(cv::ACCESS_WRITE);//clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t)*W2*H2*3, NULL, &status); CL_CHECK(status);

	cl_mem clBufferResize = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(uint8_t)*W2*H2*3, NULL, &status); CL_CHECK(status);

	OCLKernel * pk = new OCLKernel((cl_program)ctx.programs[0], "resizeNN", ctx);

	//pk->addArgUMAT(&mRGBOrig);//pk->addArg(&clBuffer);       pk->addArg(&srcStep); pk->addArg(&srcOffset); pk->addArg(&H); pk->addArg(&W);
	//pk->addArgUMAT(&mRGBResized);
	pk->addArgCLMem(clBuffer, W*3, 0, H, W, true);
	pk->addArgCLMem(clBufferResize, W2*3, 0, H2, W2, false);
	pk->addArg(&ifx); pk->addArg(&ify);
	pk->run(ctx.queue, W2, H2);

#else
	int argi=0;
	cl_kernel kernel2 = clCreateKernel(program, "resizeNN", &status); CL_CHECK(status);
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(clBuffer), &clBuffer));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(srcStep), &srcStep));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(srcOffset), &srcOffset));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(H), &H));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(W), &W));

	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(clBufferResize), &clBufferResize));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(dstStep), &dstStep));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(dstOffset), &dstOffset));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(H2), &H2));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(W2), &W2));

	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(ifx), &ifx));
	CL_CHECK(clSetKernelArg(kernel2, argi++, sizeof(ify), &ify));

	cl_event kernel2_completion;
	size_t global_work_size[2] = { (size_t)W2, (size_t)H2 };
	CL_CHECK(clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
	//CL_CHECK(clWaitForEvents(1, &kernel2_completion));
	//CL_CHECK(clReleaseEvent(kernel2_completion));
	//CL_CHECK(clFinish(queue));



#endif

	//assert_matBGR24(mRGBResized.getMat(0), 0,16,0,16, 0xff, 0xe3, 0x00);
	ctx.assert_clmem(clBufferResize, W2, H2, 0,W2,0,H2, VALUE_B, VALUE_G, VALUE_R, X_IMPRINT, Y_IMPRINT, ifx, ify);

	CL_CHECK(clReleaseMemObject(clBufferResize));
	//CL_CHECK(clReleaseKernel(kernel2));
}

#endif






#include <signal.h>
void my_sigint(int sig){ // can be called asynchronously
	printf("my_sigint is called with sig=%d\n", sig);
	gConfig.bStopTest = true;
}

int main(int argc, char **argv)
{
	args::ArgumentParser parser("OpenCL/OpenCV Test program.", "HDDL projects");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<int> thr_cnt(parser, "integer", "The Threads for testing", {'t'});
    args::ValueFlag<int> height(parser, "integer", "height of vaSurface:default 720", {'H'});
    args::ValueFlag<int> width(parser, "integer", "height of vaSurface:default 1280", {'W'});
    args::ValueFlag<int> numberTest(parser, "integer", "number of tests:default 100", {'n'});
    args::ValueFlag<int> TestCode(parser, "integer", "TestCode: default 0", {'d'});
    args::ValueFlag<bool> ocl_cache(parser, "bool", "use cache file so binary is compiled", {'c'});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    gConfig.thread_cnt = 1;
    gConfig.bUseOCLBinCache  = false;
    gConfig.Height = 720;
    gConfig.Width = 1280;
    gConfig.numberTest = 100;
    gConfig.TestCode = 0;

    if(thr_cnt) 	gConfig.thread_cnt 		= args::get(thr_cnt);
    if(ocl_cache) 	gConfig.bUseOCLBinCache = args::get(ocl_cache);
    if(height) 		gConfig.Height 			= args::get(height);
    if(width) 		gConfig.Width 			= args::get(width);
    if(numberTest)  gConfig.numberTest      = args::get(numberTest);
    if(TestCode)    gConfig.TestCode        = args::get(TestCode);

    printf("\n===========================================================\n");
    printf("thread_cnt:        %d\n", gConfig.thread_cnt);
    printf("bUseOCLBinCache:   %d\n", gConfig.bUseOCLBinCache);
    printf("thread_cnt:        %d\n", gConfig.thread_cnt);
    printf("Width x Height:    %d x %d\n", gConfig.Width, gConfig.Height);
    printf("numberTest:        %d\n", gConfig.numberTest);
    printf("vaSurface YUV:     0x%X,0x%X,0x%X\n", VALUE_Y,VALUE_U,VALUE_V);
    printf("Expected  BGR:     0x%X,0x%X,0x%X\n", VALUE_B,VALUE_G,VALUE_R);
    printf("X_IMPRINT  :       %d\n", X_IMPRINT);
    printf("Y_IMPRINT  :       %d\n", Y_IMPRINT);

    printf("\n===========================================================\n");

    signal(SIGINT, my_sigint);

    //testOCL_basic_info();
    testOCL_VA_Intel();
}


