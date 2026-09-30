//-
// =============================================================================
// Copyright 2022 Autodesk, Inc. All rights reserved.
//
// Use of this software is subject to the terms of the Autodesk license
// agreement provided at the time of installation or download, or which
// otherwise accompanies this software in either electronic or hard copy form.
// =============================================================================
//+

#ifndef PF_FLIP_SPIKE_EXPORT_H
#define PF_FLIP_SPIKE_EXPORT_H

#if defined(_WIN32)
#define PF_FLIP_SPIKE_EXPORT __declspec(dllexport)
#define PF_FLIP_SPIKE_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define PF_FLIP_SPIKE_EXPORT __attribute__((visibility("default")))
#define PF_FLIP_SPIKE_IMPORT __attribute__((visibility("default")))
#else
#define PF_FLIP_SPIKE_EXPORT
#define PF_FLIP_SPIKE_IMPORT
#endif

#if defined(PF_FLIP_SPIKE_BUILD_NODEDEF_DLL)
#define PF_FLIP_SPIKE_DECL PF_FLIP_SPIKE_EXPORT
#else
#define PF_FLIP_SPIKE_DECL PF_FLIP_SPIKE_IMPORT
#endif

#endif
