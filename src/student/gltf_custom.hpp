#pragma once
#include "tiny_gltf.h"
static constexpr int MAX_BONES = 100;

struct Bone {
    int parent = -1;
    glm::mat4 inverseBind;
    glm::mat4 localTransform;
};

struct Skeleton {
    std::vector<Bone> bones;
    std::vector<glm::mat4> finalMatrices;
};

struct Keyframe {
    float time;
    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
};

struct BoneAnimation {
    int boneIndex;
    std::vector<Keyframe> keys;
};

struct Animation {
    float duration;
    std::vector<BoneAnimation> boneAnimations;
};

struct Vec3Key {
    float time;
    glm::vec3 value;
};

struct QuatKey {
    float time;
    glm::quat value;
};

static std::vector<float> readFloatAccessor(
    const tinygltf::Model& model,
    int accessorIndex
) {
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];

    const float* data = reinterpret_cast<const float*>(
        &buffer.data[view.byteOffset + accessor.byteOffset]
        );

    return std::vector<float>(data, data + accessor.count);
}

static std::vector<glm::vec3> readVec3Accessor(
    const tinygltf::Model& model,
    int accessorIndex
) {
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];

    const float* data = reinterpret_cast<const float*>(
        &buffer.data[view.byteOffset + accessor.byteOffset]
        );

    std::vector<glm::vec3> result(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        result[i] = glm::vec3(
            data[i * 3 + 0],
            data[i * 3 + 1],
            data[i * 3 + 2]
        );
    }
    return result;
}

static std::vector<glm::quat> readQuatAccessor(
    const tinygltf::Model& model,
    int accessorIndex
) {
    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];

    const float* data = reinterpret_cast<const float*>(
        &buffer.data[view.byteOffset + accessor.byteOffset]
        );

    std::vector<glm::quat> result(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        result[i] = glm::quat(
            data[i * 4 + 3], // w
            data[i * 4 + 0], // x
            data[i * 4 + 1], // y
            data[i * 4 + 2]  // z
        );
    }
    return result;
}

