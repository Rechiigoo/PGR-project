#pragma once
#include <cstdint>

struct HatblorbVertex {
  float position[3];
  float normal[3];
};

extern const HatblorbVertex hatblorbVertices[640];
extern const uint32_t hatblorbIndices[1216][3];
