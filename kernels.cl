/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL FP_FAST_FMAF ON
#pragma OPENCL FP_FAST_FMA ON

static
__constant
float c_YUV2RGBCoeffs_420[5] =
{
     1.163999557f,
     2.017999649f,
    -0.390999794f,
    -0.812999725f,
     1.5959997177f
};

static const __constant float CV_8U_MAX         = 255.0f;
static const __constant float CV_8U_HALF        = 128.0f;
static const __constant float BT601_BLACK_RANGE = 16.0f;
static const __constant float CV_8U_SCALE       = 1.0f / 255.0f;
static const __constant float d1                = BT601_BLACK_RANGE / CV_8U_MAX;
static const __constant float d2                = CV_8U_HALF / CV_8U_MAX;

#define NCHANNELS 3

__kernel
void YUV2BGR_NV12_8u(
    read_only image2d_t imgY,
    read_only image2d_t imgUV,
    __global unsigned char* pBGR,
   int bgrStep,
   int cols,
   int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x + 1 < cols)
    {
        if (y + 1 < rows)
        {
            __global uchar* pDstRow1 = pBGR + mad24(y, bgrStep, mad24(x, NCHANNELS, 0));
            __global uchar* pDstRow2 = pDstRow1 + bgrStep;

            float4 Y1 = read_imagef(imgY, (int2)(x+0, y+0));
            float4 Y2 = read_imagef(imgY, (int2)(x+1, y+0));
            float4 Y3 = read_imagef(imgY, (int2)(x+0, y+1));
            float4 Y4 = read_imagef(imgY, (int2)(x+1, y+1));

            float4 UV = read_imagef(imgUV, (int2)(x/2, y/2)) - d2;

            __constant float* coeffs = c_YUV2RGBCoeffs_420;

            Y1 = max(0.f, Y1 - d1) * coeffs[0];
            Y2 = max(0.f, Y2 - d1) * coeffs[0];
            Y3 = max(0.f, Y3 - d1) * coeffs[0];
            Y4 = max(0.f, Y4 - d1) * coeffs[0];

            float ruv = fma(coeffs[4], UV.y, 0.0f);
            float guv = fma(coeffs[3], UV.y, fma(coeffs[2], UV.x, 0.0f));
            float buv = fma(coeffs[1], UV.x, 0.0f);

            float R1 = (Y1.x + ruv) * CV_8U_MAX;
            float G1 = (Y1.x + guv) * CV_8U_MAX;
            float B1 = (Y1.x + buv) * CV_8U_MAX;

            float R2 = (Y2.x + ruv) * CV_8U_MAX;
            float G2 = (Y2.x + guv) * CV_8U_MAX;
            float B2 = (Y2.x + buv) * CV_8U_MAX;

            float R3 = (Y3.x + ruv) * CV_8U_MAX;
            float G3 = (Y3.x + guv) * CV_8U_MAX;
            float B3 = (Y3.x + buv) * CV_8U_MAX;

            float R4 = (Y4.x + ruv) * CV_8U_MAX;
            float G4 = (Y4.x + guv) * CV_8U_MAX;
            float B4 = (Y4.x + buv) * CV_8U_MAX;

            pDstRow1[0*NCHANNELS + 0] = convert_uchar_sat(B1);
            pDstRow1[0*NCHANNELS + 1] = convert_uchar_sat(G1);
            pDstRow1[0*NCHANNELS + 2] = convert_uchar_sat(R1);

            pDstRow1[1*NCHANNELS + 0] = convert_uchar_sat(B2);
            pDstRow1[1*NCHANNELS + 1] = convert_uchar_sat(G2);
            pDstRow1[1*NCHANNELS + 2] = convert_uchar_sat(R2);

            pDstRow2[0*NCHANNELS + 0] = convert_uchar_sat(B3);
            pDstRow2[0*NCHANNELS + 1] = convert_uchar_sat(G3);
            pDstRow2[0*NCHANNELS + 2] = convert_uchar_sat(R3);

            pDstRow2[1*NCHANNELS + 0] = convert_uchar_sat(B4);
            pDstRow2[1*NCHANNELS + 1] = convert_uchar_sat(G4);
            pDstRow2[1*NCHANNELS + 2] = convert_uchar_sat(R4);
        }
    }
}

__kernel
void YUV2BGR_NV12_8u_DEBUG(
   image2d_t imgY,
   image2d_t imgUV,
   __global unsigned char* pBGR,
   int bgrStep,
   int cols,
   int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    __global uchar* pDstRow1 = pBGR + mad24(y, bgrStep, mad24(x, NCHANNELS, 0));
    
    float4 Y1 = read_imagef(imgY, (int2)(x, y));
    float4 UV1 = read_imagef(imgUV, (int2)(x/2, y/2));
    
    pDstRow1[0*NCHANNELS + 0] = convert_uchar_sat(Y1[0]*255);
    pDstRow1[0*NCHANNELS + 1] = convert_uchar_sat(UV1[0]*255);
    pDstRow1[0*NCHANNELS + 2] = convert_uchar_sat(UV1[1]*255);
}

__kernel void simple_demo(__global int *src, __global int *dst, int factor)
{
    int i = get_global_id(0);
    dst[i] = src[i] * factor;
}




#define T1 uchar
#define cn 3

#define loadpix(addr)  vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE (int)sizeof(T1)*cn

__kernel void resizeNN(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                       __global       uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                       float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float s1 = dx * ifx;
        float s2 = dy * ify;
        int sx = min(convert_int_rtz(s1), src_cols - 1);
        int sy = min(convert_int_rtz(s2), src_rows - 1);
        
        storepix(loadpix(srcptr + mad24(sy, src_step, mad24(sx, TSIZE, src_offset))),
                 dstptr + mad24(dy, dst_step, mad24(dx, TSIZE, dst_offset)));
    }
}


