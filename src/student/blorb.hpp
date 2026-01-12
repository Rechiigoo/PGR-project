#pragma once
#include <cstdint>

struct BlorbVertex {
	float position[3];
	float normal[3];
	float texcoord[2] = { 0,0 };
	float tan[3] = { 0,0,0 };
	float bitan[3] = { 0,0,0 };
	int   boneIDs[4] = { 0,0,0,0 };
	float weights[4] = { 0,0,0,0 };
};

extern BlorbVertex blorbVertices[6695];
extern const uint32_t blorbIndices[10594][3];
//WITCH HAT
extern const BlorbVertex witchVertices[225];
extern const uint32_t witchIndices[416][3];
//ROBE
extern const BlorbVertex robeVertices[902];
extern const uint32_t robeIndices[696][3];
//WINTER HAT
extern const BlorbVertex gorroVertices[1377];
extern const uint32_t gorroIndices[2464][3];
//WINTER COAT
extern const BlorbVertex coatVertices[1100];
extern const uint32_t coatIndices[1312][3];