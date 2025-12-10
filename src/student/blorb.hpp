#pragma once
#include <cstdint>

struct BlorbVertex {
	float position[3];
	float normal[3];
	float texcoord[2] = { 0,0 };
	float tan[3] = { 0,0,0 };
	float bitan[3] = { 0,0,0 };
};

extern const BlorbVertex blorbVertices[19736];
extern const uint32_t blorbIndices[17231][3];
