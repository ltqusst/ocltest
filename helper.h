

#include "va/va.h"
#include "va/va_drm.h"
//#include "va/va_x11.h"
#include "va_ext.h"

#ifdef USE_OPENCV
#include "opencv2/cvconfig.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cvconfig.h"
#ifdef WIN32
	#include "opencv2/core/directx.hpp"
#else
	#include "opencv2/core/va_intel.hpp"
#endif
#endif

static VADisplay getVADisplay(void);
static void abortTest(void);


#define CALL(st, ret, expect, ...) do{\
    st;\
    if(ret != expect){\
        printf("!!!! error on %s:%d\n!!!! %s:return value is %d(0x%x)\n",__FILE__, __LINE__, #st, ret, ret);\
        printf(__VA_ARGS__);\
        abortTest();\
    }\
    }while(0)

#define CALL_OCL(st) CALL(st, status, CL_SUCCESS, "\n");
#define CL_CHECK(call) do{cl_int ret = (call);  if(CL_SUCCESS != ret) {printf("%s:%d\n%s return %d(0x%x)\n",__FILE__,__LINE__, #call, ret, ret); fflush(stdout); abortTest();} }while(0)
#define NO_OPENCV_ERROR() 	printf("USE_OPENCV is undefined while trying to execute OpenCV code\n"); abortTest();


class VASurfaceLocker
{
public:
#define CALL_VA(st) CALL(st, status, VA_STATUS_SUCCESS, "\n");
    VASurfaceLocker(VASurfaceID &surface): m_display(getVADisplay()), m_surface(surface)
    {
        VAStatus status;
        CALL_VA(status = vaSyncSurface(m_display, surface));

        CALL_VA(status = vaDeriveImage(m_display, surface, &image));

        CALL_VA(status = vaMapBuffer(m_display, image.buf, (void **)&buffer));
    }
    void info(){
        #define PRINT_IMAGE_INFO(f)  printf("\t%s=%d\n", #f, image.f);
        printf("\tformat.fourcc=%c%c%c%c\n",image.format.fourcc&0xFF,(image.format.fourcc>>8)&0xFF,(image.format.fourcc>>16)&0xFF,(image.format.fourcc>>24)&0xFF);
        PRINT_IMAGE_INFO(format.bits_per_pixel);
        PRINT_IMAGE_INFO(width);
        PRINT_IMAGE_INFO(height);
        PRINT_IMAGE_INFO(data_size);
        PRINT_IMAGE_INFO(num_planes);
        PRINT_IMAGE_INFO(offsets[0]);
        PRINT_IMAGE_INFO(offsets[1]);
        PRINT_IMAGE_INFO(offsets[2]);
        PRINT_IMAGE_INFO(pitches[0]);
        PRINT_IMAGE_INFO(pitches[1]);
        PRINT_IMAGE_INFO(pitches[2]);
        PRINT_IMAGE_INFO(num_palette_entries);
        PRINT_IMAGE_INFO(component_order[0]);
        PRINT_IMAGE_INFO(component_order[1]);
        PRINT_IMAGE_INFO(component_order[2]);
        PRINT_IMAGE_INFO(component_order[3]);
    }
    ~VASurfaceLocker(){
        VAStatus status;
        CALL_VA(status = vaUnmapBuffer(m_display, image.buf));
        CALL_VA(status = vaDestroyImage(m_display, image.image_id));
        CALL_VA(status = vaSyncSurface(m_display, m_surface));
    }

    int W(){return image.width;}
    int H(){return image.height;}
    uint8_t * Y (int line){return buffer + image.offsets[0] + image.pitches[0] * line;}
    uint8_t * UV(int line){return buffer + image.offsets[1] + image.pitches[1] * line;}

    void brief(int YC = 16, int XC = 16){
		printf("VAsurface content: Y, UV\n");
		for(int y=0;y<YC;y++){
			printf("\t");
			for(int x=0;x<XC;x++) printf("%02X,", Y(y)[x]);
			printf("...\t");
			if(y<YC/2)
				for(int x=0;x<XC;x++)  printf("%02X,", UV(y)[x]);
			printf("...\n");
		}
    }

    VAImage image;
    uint8_t * buffer;
    VADisplay   m_display;
    VASurfaceID m_surface;
};

//========================================================================================================
static void createVASurfaces(
		VASurfaceID *surfaces, int surfaces_num,
		int Width, int Height,
		int VALUE_Y = 0x10,
		int VALUE_U = 0x80,
		int VALUE_V = 0xA0,
		int xImprint = 1,
		int yImprint = 1)
{
	VADisplay display = getVADisplay();

	VAStatus va_st;
	VASurfaceAttrib attrib;
	attrib.type             = VASurfaceAttribPixelFormat;
	attrib.value.type       = VAGenericValueTypeInteger;
	attrib.value.value.i    = VA_FOURCC_NV12;
	attrib.flags            = VA_SURFACE_ATTRIB_SETTABLE;

	CALL(va_st = vaCreateSurfaces(display,
							  	  VA_RT_FORMAT_YUV420,
								  Width,
								  Height,
								  surfaces, surfaces_num,
								  &attrib, 1),
			va_st, VA_STATUS_SUCCESS,
			"vaCreateSurfaces() %dx(%dx%d) failed: return code is 0x%x\n", surfaces_num, Width,Height,va_st);

	for(int i=0;i<surfaces_num;i++)
	{
		VASurfaceLocker vl(surfaces[i]);
		printf("surfaces[%d]:\n", i);
		vl.info();
		for(int y=0;y<vl.H();y++)
			for(int x=0;x<vl.W();x++)
				vl.Y(y)[x] = VALUE_Y + yImprint*y + xImprint*x;

		for(int y=0;y<vl.H()/2;y++) {
			for(int x=0;x<vl.W();x+=2) {
				vl.UV(y)[x+0] = VALUE_U + yImprint*y + xImprint*(x>>1);
				vl.UV(y)[x+1] = VALUE_V + yImprint*y + xImprint*(x>>1);
			}
		}
	}
}
//========================================================================================================
static void printCLNV12(cl_command_queue queue, cl_mem clImageY,cl_mem clImageUV)
{
#define NX 16
#define NY 16
	printf("clEnqueueReadImage %dx%d region from clImageY:\n", NY, NX);
	uint8_t data[NX*2];

	for(int y=0;y<NY;y++){
		size_t origin[3] = {0,(size_t)y,0};
		size_t region[3] = {NX,1,1};
		int row_pitch = NX*sizeof(uint8_t);
		CL_CHECK( clEnqueueReadImage(queue, clImageY, CL_TRUE, origin, region, row_pitch, 0, data,0, NULL, NULL));

		printf("\t");
		for(int x=0;x<NX;x++) printf("%02X,", data[x]);
		printf("...\t");

		if(y < NY/2){
			size_t origin[3] = {0,(size_t)y,0};
			size_t region[3] = {(size_t)NX,1,1};
			int row_pitch = NX*sizeof(uint8_t);
			CL_CHECK( clEnqueueReadImage(queue, clImageUV, CL_TRUE, origin, region, row_pitch, 0, data,0, NULL, NULL));
			for(int x=0;x<NX;x++)  printf("%02X,", data[x]);
		}
		printf("...\n");
	}
}

//========================================================================================================
// create OCL context with  VA intel
static clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn    clGetDeviceIDsFromVA_APIMediaAdapterINTEL;
static clCreateFromVA_APIMediaSurfaceINTEL_fn          clCreateFromVA_APIMediaSurfaceINTEL;
static clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn     clEnqueueAcquireVA_APIMediaSurfacesINTEL;
static clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn     clEnqueueReleaseVA_APIMediaSurfacesINTEL;

struct VAOCLContext{
    cl_context context = 0;
    cl_device_id device = 0;
    cl_platform_id platform = 0;

    bool useOpenCV;

    VAOCLContext(VADisplay display, bool bUseOpenCV =false){

    	VAOCLContext &ret = *this;

    	ret.useOpenCV = bUseOpenCV;

    	if(bUseOpenCV){
    #ifndef USE_OPENCV
    		NO_OPENCV_ERROR();
    #else
    		cv::va_intel::ocl::initializeContextFromVA(gConfig.display(), true);

    		cv::ocl::Context& ctx = cv::ocl::Context::getDefault();
    		ret.context = (cl_context)ctx.ptr();

    		size_t n = ctx.ndevices();
    		if(n != 1){
    			printf("Context has %d devices ?\n", n);
    			exit(2);
    		}
    		ret.device =  (cl_device_id)(ctx.device(0).ptr());

    		cv::ocl::Platform& p = cv::ocl::Platform::getDefault();
    		ret.platform = (cl_platform_id)p.ptr();

    		clGetDeviceIDsFromVA_APIMediaAdapterINTEL = (clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)clGetExtensionFunctionAddressForPlatform(platform, "clGetDeviceIDsFromVA_APIMediaAdapterINTEL");
    		clCreateFromVA_APIMediaSurfaceINTEL       = (clCreateFromVA_APIMediaSurfaceINTEL_fn)      clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromVA_APIMediaSurfaceINTEL");
    		clEnqueueAcquireVA_APIMediaSurfacesINTEL  = (clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueAcquireVA_APIMediaSurfacesINTEL");
    		clEnqueueReleaseVA_APIMediaSurfacesINTEL  = (clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueReleaseVA_APIMediaSurfacesINTEL");
    #endif
    	}
    	else
    	{
    		cl_platform_id platforms[100];
    		cl_uint numPlatforms = 0;
    		CL_CHECK(clGetPlatformIDs(100, platforms, &numPlatforms));

    			// For CL-VA interop, we must find platform/device with "cl_intel_va_api_media_sharing" extension.
    			// With standard initialization procedure, we should examine platform extension string for that.
    			// But in practice, the platform ext string doesn't contain it, while device ext string does.
    			// Follow Intel procedure (see tutorial), we should obtain device IDs by extension call.
    			// Note that we must obtain function pointers using specific platform ID, and can't provide pointers in advance.
    			// So, we iterate and select the first platform, for which we got non-NULL pointers, device, and CL context.

    		int found = -1;
    		cl_int status = 0;
    		for (int i = 0; i < (int)numPlatforms; ++i)
    		{
    			// Get extension function pointers
    			ret.platform = platforms[i];
    			clGetDeviceIDsFromVA_APIMediaAdapterINTEL = (clGetDeviceIDsFromVA_APIMediaAdapterINTEL_fn)clGetExtensionFunctionAddressForPlatform(ret.platform, "clGetDeviceIDsFromVA_APIMediaAdapterINTEL");
    			clCreateFromVA_APIMediaSurfaceINTEL       = (clCreateFromVA_APIMediaSurfaceINTEL_fn)      clGetExtensionFunctionAddressForPlatform(ret.platform, "clCreateFromVA_APIMediaSurfaceINTEL");
    			clEnqueueAcquireVA_APIMediaSurfacesINTEL  = (clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn) clGetExtensionFunctionAddressForPlatform(ret.platform, "clEnqueueAcquireVA_APIMediaSurfacesINTEL");
    			clEnqueueReleaseVA_APIMediaSurfacesINTEL  = (clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn) clGetExtensionFunctionAddressForPlatform(ret.platform, "clEnqueueReleaseVA_APIMediaSurfacesINTEL");

    			if (((void*)clGetDeviceIDsFromVA_APIMediaAdapterINTEL == NULL) ||
    				((void*)clCreateFromVA_APIMediaSurfaceINTEL == NULL) ||
    				((void*)clEnqueueAcquireVA_APIMediaSurfacesINTEL == NULL) ||
    				((void*)clEnqueueReleaseVA_APIMediaSurfacesINTEL == NULL))
    			{
    				continue;
    			}

    			// Query device list

    			cl_uint numDevices = 0;

    			status = clGetDeviceIDsFromVA_APIMediaAdapterINTEL(ret.platform, CL_VA_API_DISPLAY_INTEL, display,
    															   CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, 0, NULL, &numDevices);
    			if ((status != CL_SUCCESS) || !(numDevices > 0))
    				continue;
    			numDevices = 1; // initializeContextFromHandle() expects only 1 device
    			status = clGetDeviceIDsFromVA_APIMediaAdapterINTEL(ret.platform, CL_VA_API_DISPLAY_INTEL, display,
    															   CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, numDevices, &ret.device, NULL);
    			if (status != CL_SUCCESS)
    				continue;

    			// Creating CL-VA media sharing OpenCL context

    			cl_context_properties props[] = {
    				CL_CONTEXT_VA_API_DISPLAY_INTEL, (cl_context_properties) display,
    				CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE, // no explicit sync required
    				0
    			};

    			ret.context = clCreateContext(props, numDevices, &ret.device, NULL, NULL, &status);
    			if (status != CL_SUCCESS)
    			{
    				clReleaseDevice(ret.device);
    			}
    			else
    			{
    				found = i;

    				break;
    			}
    		}
    		if(found < 0){
    			printf("Cannot create OpenCL context from VA\n");
    			exit(1);
    		}
    	}
    }
};

