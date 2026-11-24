@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright (c) 2026 the YASAKANI authors
rem Builds the engine library for Inkling-2 Mini (see BUILDING.md, "The Inkling-2 Mini library").
rem Run it in the "x64 Native Tools Command Prompt for VS 2022" with the CUDA Toolkit 13 installed.
rem   build.cmd cuda   ->  build\inkling2\yskn_inkling2.dll   (the released library: CUDA kernels for RTX 30, 40 and 50 and the
rem                                                           CPU emulation of the same kernels for its self-test)
rem   build.cmd host   ->  build\inkling2\ink2_host.dll       (CPU emulation only; for the unit tests, not released)
rem The build folder is build\inkling2 in the source folder (the one with CMakeLists.txt). Set INK2_ARCHS to build other GPU
rem generations, for example  set INK2_ARCHS=-gencode arch=compute_120,code=sm_120   for the RTX 50 series only.
setlocal
set SRC=%~dp0
set OUT=%~dp0..\..\build\inkling2
where cl >nul 2>nul
if errorlevel 1 (
  echo build.cmd: cl.exe is not on the path. Open the "x64 Native Tools Command Prompt for VS 2022" and run it again.
  exit /b 1
)
set CLFLAGS=/nologo /O2 /MD /EHsc /arch:AVX2 /openmp /fp:precise /W3 /wd4996
if "%1"=="host" goto host
if "%1"=="cuda" goto cuda
echo usage: build.cmd cuda^|host
exit /b 2

:host
if not exist "%OUT%\host" mkdir "%OUT%\host"
cl %CLFLAGS% /LD "%SRC%ink2_engine.cpp" "%SRC%ink2_selftest.cpp" "%SRC%ink2_emu.cpp" /Fo"%OUT%\host\\" /Fe"%OUT%\ink2_host.dll" /link /IMPLIB:"%OUT%\host\ink2_host.lib"
if errorlevel 1 exit /b 1
echo built %OUT%\ink2_host.dll
exit /b 0

:cuda
set NVCC=nvcc
if not "%CUDA_PATH%"=="" set NVCC="%CUDA_PATH%\bin\nvcc.exe"
set CUDART=cudart_static.lib
if not "%CUDA_PATH%"=="" set CUDART="%CUDA_PATH%\lib\x64\cudart_static.lib"
rem machine code for compute capability 8.6 (RTX 30), 8.9 (RTX 40) and 12.0 (RTX 50), PTX for compute_120 (as yskn.dll)
if "%INK2_ARCHS%"=="" set INK2_ARCHS=-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=[sm_120,compute_120]
set NVFLAGS=-c -O3 %INK2_ARCHS% -fmad=false -std=c++17 -Xcompiler "/MD,/EHsc,/O2,/wd4996,/wd4819,/Zc:preprocessor"
if not exist "%OUT%\cuda" mkdir "%OUT%\cuda"
rem The NVFP4 tensor-core prompt products (ink2_nvfp4.cu) need sm_120a: machine code for the RTX 50 series only, used
rem there when the option nvfp4_prefill asks for it. set INK2_NVFP4_TC=0 builds the library without them.
set TCOBJ=
if "%INK2_NVFP4_TC%"=="0" goto notc
%NVCC% -c -O3 -gencode arch=compute_120a,code=sm_120a -fmad=false -std=c++17 -Xcompiler "/MD,/EHsc,/O2,/Zc:preprocessor" "%SRC%ink2_nvfp4.cu" -o "%OUT%\cuda\ink2_nvfp4.obj"
if errorlevel 1 exit /b 1
set TCOBJ="%OUT%\cuda\ink2_nvfp4.obj"
set NVFLAGS=%NVFLAGS% -DINK2_WITH_NVFP4_TC
:notc
%NVCC% %NVFLAGS% "%SRC%ink2_cuda.cu" -o "%OUT%\cuda\ink2_cuda.obj"
if errorlevel 1 exit /b 1
%NVCC% %NVFLAGS% "%SRC%ink2_micro.cu" -o "%OUT%\cuda\ink2_micro.obj"
if errorlevel 1 exit /b 1
rem the image encoder (ink2_vision.cu: patches, 23 blocks and the projector of the image tower); the engine reaches it
rem with INK2_WITH_VISION (without it, the ink2_vision_* calls refuse with a message)
%NVCC% %NVFLAGS% "%SRC%ink2_vision.cu" -o "%OUT%\cuda\ink2_vision.obj"
if errorlevel 1 exit /b 1
rem the fast kernels (ink2_fast.cu, default -fmad=true: the fused forms are written out in the source)
%NVCC% -c -O3 %INK2_ARCHS% -std=c++17 -Xcompiler "/MD,/EHsc,/O2,/wd4996,/wd4819,/Zc:preprocessor" "%SRC%ink2_fast.cu" -o "%OUT%\cuda\ink2_fast.obj"
if errorlevel 1 exit /b 1
cl %CLFLAGS% /c /DINK2_WITH_CUDA /DINK2_WITH_VISION "%SRC%ink2_engine.cpp" "%SRC%ink2_selftest.cpp" "%SRC%ink2_emu.cpp" /Fo"%OUT%\cuda\\"
if errorlevel 1 exit /b 1
link /nologo /DLL /OUT:"%OUT%\yskn_inkling2.dll" /IMPLIB:"%OUT%\cuda\yskn_inkling2.lib" "%OUT%\cuda\ink2_engine.obj" "%OUT%\cuda\ink2_selftest.obj" "%OUT%\cuda\ink2_emu.obj" "%OUT%\cuda\ink2_cuda.obj" "%OUT%\cuda\ink2_micro.obj" "%OUT%\cuda\ink2_vision.obj" "%OUT%\cuda\ink2_fast.obj" %TCOBJ% %CUDART% /NODEFAULTLIB:LIBCMT
if errorlevel 1 exit /b 1
echo built %OUT%\yskn_inkling2.dll
exit /b 0
