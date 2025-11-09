// obj_loader.h  (single-header style for clarity)
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include<SDL.h>
#include<Vars/Vars.h>
#include<geGL/geGL.h>
#include<glm/glm.hpp>

using namespace glm;

static void normalize_inplace(vec3& v) {
    float len = std::sqrt(dot(v, v));
    if (len > 1e-9f) { v = v * (1.0f / len); }
}

struct Vertex {
    vec3 pos;
    vec2 uv;
    vec3 normal;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

struct ObjIndex {
    int v = 0;
    int vt = 0;
    int vn = 0;
    bool operator==(ObjIndex const& o) const {
        return v == o.v && vt == o.vt && vn == o.vn;
    }
};

struct ObjIndexHash {
    size_t operator()(ObjIndex const& k) const noexcept {
        // combine three ints; simple mix
        uint64_t a = (uint32_t)k.v, b = (uint32_t)k.vt, c = (uint32_t)k.vn;
        uint64_t res = a;
        res = (res * 73856093u) ^ (b * 19349663u) ^ (c * 83492791u);
        return (size_t)res;
    }
};

static int fix_index(int idx, int vec_size) {
    // OBJ: 1-based; negative indexes count from end
    if (idx > 0) return idx - 1;
    if (idx < 0) return vec_size + idx; // idx is negative
    return -1;
}

Mesh loadOBJ(const std::string& path, bool computeNormalsIfMissing = true) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open: " + path);
    }

    std::vector<vec3> positions;
    std::vector<vec2> texcoords;
    std::vector<vec3> normals;
    std::vector<std::vector<ObjIndex>> faces; // each face is a list of ObjIndex

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // trim left
        size_t p = 0;
        while (p < line.size() && isspace((unsigned char)line[p])) p++;
        if (p >= line.size()) continue;
        if (line[p] == '#') continue;

        std::istringstream ss(line.substr(p));
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(vec3(x, y, z));
        }
        else if (tag == "vt") {
            float u, v; ss >> u >> v;
            texcoords.push_back(vec2{ u,v });
        }
        else if (tag == "vn") {
            float nx, ny, nz; ss >> nx >> ny >> nz;
            normals.push_back(vec3(nx, ny, nz));
        }
        else if (tag == "f") {
            std::vector<ObjIndex> face;
            std::string vertToken;
            while (ss >> vertToken) {
                // parse forms: v, v/vt, v//vn, v/vt/vn
                ObjIndex idx{};
                int a = 0, b = 0, c = 0;
                size_t firstSlash = vertToken.find('/');
                if (firstSlash == std::string::npos) {
                    a = std::stoi(vertToken);
                    idx.v = a;
                }
                else {
                    std::string s1 = vertToken.substr(0, firstSlash);
                    size_t secondSlash = vertToken.find('/', firstSlash + 1);
                    if (secondSlash == std::string::npos) {
                        // v/vt
                        a = s1.empty() ? 0 : std::stoi(s1);
                        b = std::stoi(vertToken.substr(firstSlash + 1));
                        idx.v = a;
                        idx.vt = b;
                    }
                    else {
                        // v//vn or v/vt/vn
                        a = s1.empty() ? 0 : std::stoi(s1);
                        std::string s2 = vertToken.substr(firstSlash + 1, secondSlash - firstSlash - 1);
                        std::string s3 = vertToken.substr(secondSlash + 1);
                        if (s2.empty()) {
                            // v//vn
                            b = 0;
                        }
                        else {
                            b = std::stoi(s2);
                        }
                        c = s3.empty() ? 0 : std::stoi(s3);
                        idx.v = a; idx.vt = b; idx.vn = c;
                    }
                }
                face.push_back(idx);
            }
            if (face.size() >= 3) faces.push_back(std::move(face));
        }
        else {
            // ignore other tags (o, g, usemtl, mtllib, s, etc.) for now
        }
    }

    // Build vertex/index lists with deduplication
    Mesh mesh;
    mesh.vertices.clear();
    mesh.indices.clear();
    std::unordered_map<ObjIndex, uint32_t, ObjIndexHash> indexMap;
    indexMap.reserve(faces.size() * 3);

    auto addVertex = [&](const ObjIndex& oi)->uint32_t {
        auto it = indexMap.find(oi);
        if (it != indexMap.end()) return it->second;

        Vertex v{};
        // positions
        int ip = fix_index(oi.v, (int)positions.size());
        if (ip >= 0 && ip < (int)positions.size()) v.pos = positions[ip];
        else v.pos = vec3(0, 0, 0);

        // texcoords
        if (oi.vt != 0) {
            int itc = fix_index(oi.vt, (int)texcoords.size());
            if (itc >= 0 && itc < (int)texcoords.size()) v.uv = texcoords[itc];
            else v.uv = vec2{ 0,0 };
        }
        else v.uv = vec2{ 0,0 };

        // normals
        if (oi.vn != 0) {
            int in = fix_index(oi.vn, (int)normals.size());
            if (in >= 0 && in < (int)normals.size()) v.normal = normals[in];
            else v.normal = vec3(0, 0, 0);
        }
        else v.normal = vec3(0, 0, 0);

        uint32_t newIndex = (uint32_t)mesh.vertices.size();
        indexMap.emplace(oi, newIndex);
        mesh.vertices.push_back(v);
        return newIndex;
        };

    // Triangulate faces (fan)
    for (auto& face : faces) {
        if (face.size() == 3) {
            uint32_t i0 = addVertex(face[0]);
            uint32_t i1 = addVertex(face[1]);
            uint32_t i2 = addVertex(face[2]);
            mesh.indices.push_back(i0);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);
        }
        else {
            // polygon -> fan triangulation
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                uint32_t i0 = addVertex(face[0]);
                uint32_t i1 = addVertex(face[i]);
                uint32_t i2 = addVertex(face[i + 1]);
                mesh.indices.push_back(i0);
                mesh.indices.push_back(i1);
                mesh.indices.push_back(i2);
            }
        }
    }

    // Compute normals if missing
    bool anyNormMissing = false;
    for (const auto& v : mesh.vertices) {
        if (std::abs(v.normal.x) < 1e-6f && std::abs(v.normal.y) < 1e-6f && std::abs(v.normal.z) < 1e-6f) {
            anyNormMissing = true;
            break;
        }
    }
    if (anyNormMissing && computeNormalsIfMissing) {
        // zero-out normals
        for (auto& v : mesh.vertices) v.normal = vec3(0, 0, 0);
        // accumulate face normals
        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
            Vertex& A = mesh.vertices[mesh.indices[i + 0]];
            Vertex& B = mesh.vertices[mesh.indices[i + 1]];
            Vertex& C = mesh.vertices[mesh.indices[i + 2]];
            vec3 e1 = B.pos - A.pos;
            vec3 e2 = C.pos - A.pos;
            vec3 faceN = cross(e1, e2);
            // accumulate
            A.normal = A.normal + faceN;
            B.normal = B.normal + faceN;
            C.normal = C.normal + faceN;
        }
        // normalize
        for (auto& v : mesh.vertices) normalize_inplace(v.normal);
    }

    return mesh;
}
