#define STBI_ONLY_JPEG
#include "stb_image.h"
#include<glm/geometric.hpp>
#include<glm/glm.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<SDL.h>
#include<Vars/Vars.h>
#include<geGL/geGL.h>
#include<geGL/StaticCalls.h>
#include<imguiDormon/imgui.h>
#include<imguiVars/addVarsLimits.h>
#include<framework/FunctionPrologue.h>
#include<framework/methodRegister.hpp>
#include<framework/makeProgram.hpp>
 //additional libraries
#include <PGR/02/model.hpp> //camera model
#include<BasicCamera/OrbitCamera.h>
#include<BasicCamera/PerspectiveCamera.h>
#include<PGR/01/compileShaders.hpp>
#include<framework/bunny.hpp>
#include "blorb.hpp"
#include "hatblorb.hpp"
#include<PGR/03/phong.hpp>
#include "gltf_custom.hpp"
#include <cstddef>
#include <cmath>
#include <iostream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <filesystem>



using namespace ge::gl;
using namespace std;
using namespace compileShaders;

namespace student::project {
    GLuint groundPlaneVAO = 0;
    GLuint groundPlaneVBO = 0;
    GLuint groundPlaneEBO = 0;
    GLuint groundPlaneProgram = 0;

    // Ground plane vertices - a large flat plane at Y=0
    float groundPlaneVertices[] = {
        // Position (X, Y, Z)     // Normal (pointing up)
        -20.0f, 0.0f, -20.0f,     0.0f, 1.0f, 0.0f,
         20.0f, 0.0f, -20.0f,     0.0f, 1.0f, 0.0f,
         20.0f, 0.0f,  20.0f,     0.0f, 1.0f, 0.0f,
        -20.0f, 0.0f,  20.0f,     0.0f, 1.0f, 0.0f
    };

    unsigned int groundPlaneIndices[] = {
        0, 1, 2,
        2, 3, 0
    };

    struct Mesh {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ebo = 0;
        GLsizei indexCount = 0;
    };

    struct GLBModel {
        tinygltf::Model model;

        GLuint diffuseTex = 0;

        Skeleton skeleton;
        Animation animation;

        bool loaded = false;
    };

    GLuint glbDiffuseTex = 0;
    GLBModel glbModel;
    Mesh glbMesh;
    // GL objects
    GLuint prg = 0;
    Mesh blorb;
    Mesh hat;
    Mesh robe;
    Mesh winthat;
    //witch hat
    GLuint hatPrg = 0;
    GLuint hatTex = 0;
    //robe
    GLuint robeprg = 0;
    GLuint robetex = 0;
    //winter hat
    GLuint wintprg = 0;
    GLuint winttex = 0;
    //coat
    GLuint coatprg = 0;
    GLuint coattex = 0;
    Mesh coat;
    //skybox 
    GLuint skyboxvao, skyboxvbo;
    GLuint skyboxProgram, skyboxTex;

    GLuint bonesUBO = 0;
    GLuint texID = 0;
    GLuint diffuseTex; //leather
    GLuint normalTex;
    float animTime = 0.0f;

    //shadowmapping variables
    GLuint shadowFBO = 0;
    GLuint shadowMap = 0;
    GLuint shadowProgram = 0;
    GLuint shadowProgramGLTF = 0;

    const unsigned int SHADOW_WIDTH = 2048;
    const unsigned int SHADOW_HEIGHT = 2048;

    glm::mat4 lightSpaceMatrix;
    glm::vec3 lightPos = glm::vec3(5.0f, 10.0f, 5.0f);


    // keep track of last compiled shader toggle
    static int lastShaderToggle = -1;

    // transforms & uniforms
    glm::mat4 proj = glm::mat4(1.f);
    glm::mat4 view = glm::mat4(1.f);
    glm::mat4 model = glm::mat4(1.f);

    GLint viewUniform = -1;
    GLint projUniform = -1;
    GLint modelUniform = -1;
    int headBoneIndex = -1;
    int torsoBoneIndex = -1;

    // forward declarations
    void computeProjectionMatrix(vars::Vars& vars);
    void computeViewMatrix(vars::Vars& vars);

    std::string getTexturePath(const std::string& filename) {
        // Get the path to the source file
        std::filesystem::path sourcePath = __FILE__;
        std::filesystem::path sourceDir = sourcePath.parent_path();

        // Combine with filename
        std::filesystem::path fullPath = sourceDir / filename;

        // Convert to string and print for debugging
        std::string pathStr = fullPath.string();
        std::cout << "Loading texture from: " << pathStr << std::endl;

        return pathStr;
    }

    void initGroundPlane() {
        glGenVertexArrays(1, &groundPlaneVAO);
        glGenBuffers(1, &groundPlaneVBO);
        glGenBuffers(1, &groundPlaneEBO);

        glBindVertexArray(groundPlaneVAO);

        glBindBuffer(GL_ARRAY_BUFFER, groundPlaneVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(groundPlaneVertices), groundPlaneVertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, groundPlaneEBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(groundPlaneIndices), groundPlaneIndices, GL_STATIC_DRAW);

        // Position attribute
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

        // Normal attribute
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

        glBindVertexArray(0);

        std::cout << "Ground plane initialized" << std::endl;
    }
    GLuint loadCubemap(const std::vector<std::string>& faces)
    {
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texID);

        int width, height, channels;

        for (GLuint i = 0; i < faces.size(); i++) {
            unsigned char* data = stbi_load(
                faces[i].c_str(),
                &width,
                &height,
                &channels,
                0
            );

            if (!data) {
                std::cerr << "Failed to load cubemap face: "
                    << faces[i] << std::endl;
                continue;
            }

            GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;

            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                format,
                width,
                height,
                0,
                format,
                GL_UNSIGNED_BYTE,
                data
            );

            stbi_image_free(data);
        }

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        return texID;
    }
    std::string const groundPlaneShader = R".(
#ifdef VERTEX_SHADER
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 vNormal;
out vec3 vPosition;
out vec4 vFragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 lightSpaceMatrix;

void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    
    vPosition = worldPos.xyz;
    vNormal = normalize(mat3(model) * aNormal);
    vFragPosLightSpace = lightSpaceMatrix * worldPos;
    
    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vNormal;
in vec3 vPosition;
in vec4 vFragPosLightSpace;

out vec4 FragColor;

uniform sampler2D shadowMap;
uniform vec3 lightPos = vec3(5.0, 10.0, 5.0);

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    
    float bias = max(0.015 * (1.0 - dot(normal, lightDir)), 0.005);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    
    // Very subtle ground color
    vec3 groundColor = vec3(0.15, 0.15, 0.15);
    
    // Ambient lighting
    vec3 ambient = 0.3 * groundColor;
    
    // Diffuse lighting
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * groundColor * 0.5;
    
    // Shadow
    float shadow = ShadowCalculation(vFragPosLightSpace, N, L);
    
    // Make shadows more visible on the ground
    vec3 result = ambient + (1.0 - shadow) * diffuse;
    
    // Add slight transparency so it doesn't obscure too much
    FragColor = vec4(result, 0.95);
}
#endif
).";
    std::string const leatherWithShadows = R".(
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out VS_OUT {
    vec2 texCoord;
    vec3 fragPos;
    vec3 normal;
    vec4 fragPosLightSpace;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 lightSpaceMatrix;

void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);

    vs_out.fragPos = worldPos.xyz;
    vs_out.normal  = normalize(mat3(model) * aNormal);
    vs_out.texCoord = aTexCoord;
    vs_out.fragPosLightSpace = lightSpaceMatrix * worldPos;

    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in VS_OUT {
    vec2 texCoord;
    vec3 fragPos;
    vec3 normal;
    vec4 fragPosLightSpace;
} fs_in;

out vec4 FragColor;

uniform sampler2D alb;
uniform sampler2D nor;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 cameraPos;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // Bias to prevent shadow acne
    float bias = max(0.015 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF (Percentage Closer Filtering) for soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    // Keep the shadow at 0.0 when outside the far_plane region
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

void main()
{
    //-----------------------------------------------------------
    // Build TBN per pixel using derivatives
    //-----------------------------------------------------------
    vec3 N = normalize(fs_in.normal);

    vec3 dp1 = dFdx(fs_in.fragPos);
    vec3 dp2 = dFdy(fs_in.fragPos);
    vec2 duv1 = dFdx(fs_in.texCoord);
    vec2 duv2 = dFdy(fs_in.texCoord);

    vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 B = normalize(-dp1 * duv2.x + dp2 * duv1.x);

    mat3 TBN = mat3(T, B, N);

    //-----------------------------------------------------------
    // Sample and transform normal
    //-----------------------------------------------------------
    vec3 normalMap = texture(nor, fs_in.texCoord).rgb;
    normalMap = normalize(normalMap * 2.0 - 1.0);
    vec3 normal = normalize(TBN * normalMap);

    //-----------------------------------------------------------
    // Lighting vectors
    //-----------------------------------------------------------
    vec3 lightDir = normalize(lightPos - fs_in.fragPos);
    vec3 viewDir  = normalize(cameraPos - fs_in.fragPos);

    //-----------------------------------------------------------
    // Ambient
    //-----------------------------------------------------------
    vec3 albedo = texture(alb, fs_in.texCoord).rgb;
    vec3 ambient = 0.2 * albedo;

    //-----------------------------------------------------------
    // Diffuse
    //-----------------------------------------------------------
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo;

    //-----------------------------------------------------------
    // Soft specular (leather)
    //-----------------------------------------------------------
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(reflectDir, viewDir), 0.0), 16.0);
    vec3 specular = spec * 0.1 * vec3(1.0);

    //-----------------------------------------------------------
    // Shadow calculation
    //-----------------------------------------------------------
    float shadow = ShadowCalculation(fs_in.fragPosLightSpace, normal, lightDir);

    //-----------------------------------------------------------
    // Final color (ambient not affected by shadow)
    //-----------------------------------------------------------
    vec3 color = ambient + (1.0 - shadow) * (diffuse + specular);

    FragColor = vec4(color * lightColor, 1.0);
}
#endif
).";
    // Add these shader strings with your other shaders
    std::string const shadowVertexShader = R".(
#ifdef VERTEX_SHADER
layout(location = 0) in vec3 aPos;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    gl_Position = lightSpaceMatrix * model * vec4(aPos, 1.0);
}
#endif

#ifdef FRAGMENT_SHADER
void main()
{
    // Depth is written automatically
}
#endif
).";

    std::string const shadowVertexShaderGLTF = R".(
#ifdef VERTEX_SHADER
layout(location = 0) in vec3 aPos;
layout(location = 5) in ivec4 aBoneIDs;
layout(location = 6) in vec4 aWeights;

uniform mat4 bones[100];
uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main()
{
    // Normalize weights
    vec4 w = aWeights;
    float sum = w.x + w.y + w.z + w.w;
    if (sum > 0.001)
        w /= sum;
    else
        w = vec4(1.0, 0.0, 0.0, 0.0);
    
    // Clamp bone IDs
    ivec4 ids = clamp(aBoneIDs, ivec4(0), ivec4(99));
    
    // Calculate skinning matrix
    mat4 skinMatrix = 
        w.x * bones[ids.x] +
        w.y * bones[ids.y] +
        w.z * bones[ids.z] +
        w.w * bones[ids.w];
    
    vec4 skinnedPos = skinMatrix * vec4(aPos, 1.0);
    gl_Position = lightSpaceMatrix * model * skinnedPos;
}
#endif

#ifdef FRAGMENT_SHADER
void main()
{
    // Depth is written automatically
}
#endif
).";

    std::string const gltfWithShadows = R".(
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 5) in ivec4 aBoneIDs;
layout (location = 6) in vec4  aWeights;

uniform mat4 bones[100];
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 lightSpaceMatrix;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vTexCoord;
out vec4 vFragPosLightSpace;

void main()
{
    vec4 w = aWeights;
    float sum = w.x + w.y + w.z + w.w;
    if (sum > 0.001)
        w /= sum;
    else
        w = vec4(1.0, 0.0, 0.0, 0.0);
    
    ivec4 ids = clamp(aBoneIDs, ivec4(0), ivec4(99));
    
    mat4 skinMatrix = 
        w.x * bones[ids.x] +
        w.y * bones[ids.y] +
        w.z * bones[ids.z] +
        w.w * bones[ids.w];
    
    vec4 skinnedPos = skinMatrix * vec4(aPos, 1.0);
    vec4 worldPos = model * skinnedPos;
    
    vec3 skinnedNormal = mat3(skinMatrix) * aNormal;
    vNormal = normalize(mat3(model) * skinnedNormal);
    
    vPosition = worldPos.xyz;
    vTexCoord = aTexCoord;
    vFragPosLightSpace = lightSpaceMatrix * worldPos;
    
    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vNormal;
in vec3 vPosition;
in vec2 vTexCoord;
in vec4 vFragPosLightSpace;

out vec4 FragColor;

uniform vec3 lightPos = vec3(5.0, 10.0, 5.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 baseColor = vec3(0.8, 0.6, 0.4);
uniform sampler2D diffuseTexture;
uniform sampler2D shadowMap;
uniform int useTexture = 0;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    // Perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Get closest depth value from light's perspective
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    
    // INCREASED BIAS to fix shadow acne on top of head
    // Use a larger base bias and increase it based on surface angle
    float bias = max(0.015 * (1.0 - dot(normal, lightDir)), 0.005);
    
    // PCF (Percentage Closer Filtering) for soft shadows
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    // Keep the shadow at 0.0 when outside the far_plane region of the light's frustum
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

void main()
{
    vec3 color = baseColor;
    if (useTexture == 1) {
        color = texture(diffuseTexture, vTexCoord).rgb;
    }
    
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    
    // Ambient
    vec3 ambient = 0.3 * color;
    
    // Diffuse
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * color * lightColor;
    
    // Shadow
    float shadow = ShadowCalculation(vFragPosLightSpace, N, L);
    
    // Final color (ambient is not affected by shadow)
    vec3 result = ambient + (1.0 - shadow) * diffuse;
    
    FragColor = vec4(result, 1.0);
}
#endif
).";
    std::string const sourcetexWithShadows = R".(
#ifdef VERTEX_SHADER
uniform mat4 view  = mat4(1.0);
uniform mat4 proj  = mat4(1.0);
uniform mat4 model = mat4(1.0);
uniform mat4 lightSpaceMatrix = mat4(1.0);

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec2 vTexCoord;
out vec3 vNormal;
out vec3 vPosition;
out vec4 vFragPosLightSpace;

void main(){
    vec4 worldPos = model * vec4(position, 1.0);
    
    vTexCoord = texCoord;
    vPosition = worldPos.xyz;
    vNormal = mat3(transpose(inverse(model))) * normal;
    vFragPosLightSpace = lightSpaceMatrix * worldPos;

    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vPosition;
in vec3 vNormal;
in vec2 vTexCoord;
in vec4 vFragPosLightSpace;

uniform vec3  lightPosition = vec3(5,10,5);
uniform vec3  lightColor    = vec3(1,1,1);
uniform vec3  lightAmbient  = vec3(0.3,0.1,0.0);
uniform float shininess     = 60.0;

uniform sampler2D tex;
uniform sampler2D shadowMap;
uniform vec3 baseColor = vec3(1.0);
uniform int  useTexture = 1;

layout(location = 0) out vec4 fColor;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    
    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.001);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;
    
    if(projCoords.z > 1.0)
        shadow = 0.0;
    
    return shadow;
}

void main(){
    vec3 diffuseColor = baseColor;

    if (useTexture == 1) {
        diffuseColor = texture(tex, vTexCoord).rgb;
    }

    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPosition - vPosition);
    
    // Simple lighting
    vec3 ambient = 0.3 * diffuseColor;
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * diffuseColor * lightColor;
    
    // Shadow
    float shadow = ShadowCalculation(vFragPosLightSpace, N, L);
    
    vec3 finalColor = ambient + (1.0 - shadow) * diffuse;

    fColor = vec4(finalColor, 1.0);
}
#endif
).";
    std::string const gltf_noskin = R".(
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 5) in ivec4 aBoneIDs;
layout (location = 6) in vec4  aWeights;

uniform mat4 bones[100];
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vTexCoord;

void main()
{
    // TEST: Render without skinning
    vec4 worldPos = model * vec4(aPos, 1.0);
    
    vNormal = normalize(mat3(model) * aNormal);
    vPosition = worldPos.xyz;
    vTexCoord = aTexCoord;
    
    gl_Position = proj * view * worldPos;
}
#endif
#ifdef FRAGMENT_SHADER
in vec3 vNormal;
in vec3 vPosition;
in vec2 vTexCoord;

out vec4 FragColor;

uniform vec3 lightPos = vec3(5.0, 5.0, 5.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 baseColor = vec3(0.8, 0.6, 0.4);  // Default tan color
uniform sampler2D diffuseTexture;
uniform int useTexture = 0;

void main()
{
    // Get base color
    vec3 color = baseColor;
    if (useTexture == 1) {
        color = texture(diffuseTexture, vTexCoord).rgb;
    }
    
    // Simple lighting
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    
    // Ambient
    vec3 ambient = 0.3 * color;
    
    // Diffuse
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * color * lightColor;
    
    // Final color
    vec3 result = ambient + diffuse;
    
    FragColor = vec4(result, 1.0);
}
#endif
).";
    std::string const gltf = R".(
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 5) in ivec4 aBoneIDs;
layout (location = 6) in vec4  aWeights;

uniform mat4 bones[100];
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vTexCoord;

void main()
{
    // Normalize weights
    vec4 w = aWeights;
    float sum = w.x + w.y + w.z + w.w;
    if (sum > 0.001)
        w /= sum;
    else
        w = vec4(1.0, 0.0, 0.0, 0.0);
    
    // Clamp bone IDs
    ivec4 ids = clamp(aBoneIDs, ivec4(0), ivec4(99));
    
    // Calculate skinning matrix
    mat4 skinMatrix = 
        w.x * bones[ids.x] +
        w.y * bones[ids.y] +
        w.z * bones[ids.z] +
        w.w * bones[ids.w];
    
    // CRITICAL FIX: Apply skinning in LOCAL space, THEN model transform
    // Old (wrong): model * skinMatrix * vec4(aPos, 1.0)
    // New (correct): model * (skinMatrix * vec4(aPos, 1.0))
    vec4 skinnedPos = skinMatrix * vec4(aPos, 1.0);
    vec4 worldPos = model * skinnedPos;
    
    // Transform normal: extract 3x3 from skinMatrix, then apply model
    vec3 skinnedNormal = mat3(skinMatrix) * aNormal;
    vNormal = normalize(mat3(model) * skinnedNormal);
    
    vPosition = worldPos.xyz;
    vTexCoord = aTexCoord;
    
    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vNormal;
in vec3 vPosition;
in vec2 vTexCoord;

out vec4 FragColor;

uniform vec3 lightPos = vec3(5.0, 5.0, 5.0);
uniform vec3 lightColor = vec3(1.0, 1.0, 1.0);
uniform vec3 baseColor = vec3(0.8, 0.6, 0.4);
uniform sampler2D diffuseTexture;
uniform int useTexture = 0;

void main()
{
    vec3 color = baseColor;
    if (useTexture == 1) {
        color = texture(diffuseTexture, vTexCoord).rgb;
    }
    
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPosition);
    
    vec3 ambient = 0.3 * color;
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * color * lightColor;
    
    vec3 result = ambient + diffuse;
    
    FragColor = vec4(result, 1.0);
}
#endif
).";

    // --- leather shader string (guarded, no #version inside) ---
    std::string const leather = R".(
#ifdef VERTEX_SHADER
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out VS_OUT {
    vec2 texCoord;
    vec3 fragPos;
    vec3 normal;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);

    vs_out.fragPos = worldPos.xyz;
    vs_out.normal  = normalize(mat3(model) * aNormal);
    vs_out.texCoord = aTexCoord;

    gl_Position = proj * view * worldPos;
}
#endif

#ifdef FRAGMENT_SHADER
in VS_OUT {
    vec2 texCoord;
    vec3 fragPos;
    vec3 normal;
} fs_in;

out vec4 FragColor;

uniform sampler2D alb;
uniform sampler2D nor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 cameraPos;

void main()
{
    //-----------------------------------------------------------
    // Build TBN per pixel using derivatives
    //-----------------------------------------------------------
    vec3 N = normalize(fs_in.normal);

    vec3 dp1 = dFdx(fs_in.fragPos);
    vec3 dp2 = dFdy(fs_in.fragPos);
    vec2 duv1 = dFdx(fs_in.texCoord);
    vec2 duv2 = dFdy(fs_in.texCoord);

    vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 B = normalize(-dp1 * duv2.x + dp2 * duv1.x);

    mat3 TBN = mat3(T, B, N);

    //-----------------------------------------------------------
    // Sample and transform normal
    //-----------------------------------------------------------
    vec3 normalMap = texture(nor, fs_in.texCoord).rgb;
    normalMap = normalize(normalMap * 2.0 - 1.0);
    vec3 normal = normalize(TBN * normalMap);

    //-----------------------------------------------------------
    // Lighting vectors
    //-----------------------------------------------------------
    vec3 lightDir = normalize(lightPos - fs_in.fragPos);
    vec3 viewDir  = normalize(cameraPos - fs_in.fragPos);

    //-----------------------------------------------------------
    // Diffuse
    //-----------------------------------------------------------
    float diff = max(dot(normal, lightDir), 0.0);

    //-----------------------------------------------------------
    // Soft specular (leather)
    //-----------------------------------------------------------
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(reflectDir, viewDir), 0.0), 16.0);

    //-----------------------------------------------------------
    // Final color
    //-----------------------------------------------------------
    vec3 albedo = texture(alb, fs_in.texCoord).rgb;

    vec3 color =
        albedo * (0.2 + diff) +
        spec * 0.1;

    FragColor = vec4(color * lightColor, 1.0);
}
#endif
).";


    std::string const texshader = R".(
      #ifdef VERTEX_SHADER

    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    layout(location=2) in vec2 aTexCoord;


    out vec2 TexCoord;
    uniform mat4 view;
    uniform mat4 proj;
    uniform mat4 model = mat4(1.f);

    void main(){
        gl_Position = proj * view *  model * vec4(position, 1.0);
        TexCoord = aTexCoord;
    }
    #endif

    #ifdef FRAGMENT_SHADER
      in vec2 TexCoord;
      out vec4 fColor;
     uniform sampler2D tex;

    void main(){
        fColor = texture(tex, TexCoord);
        //fColor = vec4(TexCoord, 0.0, 0.0);
    }
    #endif

).";
    std::string const sourcetex = R".(

#ifdef VERTEX_SHADER
uniform mat4 view  = mat4(1.0);
uniform mat4 proj  = mat4(1.0);
uniform mat4 model = mat4(1.0);

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

out vec2 vTexCoord;
out vec3 vNormal;
out vec3 vPosition;
out vec3 vCamPosition;

void main(){
    vTexCoord   = texCoord;
    vPosition   = vec3(model * vec4(position, 1.0));
    vNormal     = mat3(transpose(inverse(model))) * normal;
    vCamPosition = vec3(inverse(view) * vec4(0,0,0,1));

    gl_Position = proj * view * model * vec4(position, 1.0);
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vPosition;
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vCamPosition;

uniform vec3  lightPosition = vec3(30,30,30);
uniform vec3  lightColor    = vec3(1,1,1);
uniform vec3  lightAmbient  = vec3(0.3,0.1,0.0);
uniform float shininess     = 60.0;

uniform sampler2D tex;        
uniform vec3 baseColor = vec3(1.0);
uniform int  useTexture = 1;

layout(location = 0) out vec4 fColor;

void main(){
    vec3 diffuseColor = baseColor;

    if (useTexture == 1) {
        diffuseColor = texture(tex, vTexCoord).rgb;
    }

    vec3 finalColor = phongLighting(
        vPosition,
        normalize(vNormal),
        lightPosition,
        vCamPosition,
        lightColor,
        lightAmbient,
        diffuseColor,
        shininess,
        0.0
    );

    fColor = vec4(finalColor, 1.0);
}
#endif
).";

    std::string const furshader = R".(
    #ifdef VERTEX_SHADER
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal;

    out vec3 vNormal;
    out vec3 vPosition;

    uniform mat4 model = mat4(1.f);; 
    uniform mat4 view = mat4(1.f);;
    uniform mat4 proj = mat4(1.f);;

    void main() {
        vNormal = mat3(transpose(inverse(model))) * normal;
        vPosition = vec3(model * vec4(position, 1.0));
        gl_Position = proj * view * vec4(vPosition, 1.0);
    }
    #endif

#ifdef GEOMETRY_SHADER

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

in vec3 vNormal[];
in vec3 vPosition[];

out vec3 gNormal;
out vec3 gPosition;
out float gLayer;

uniform mat4 view  = mat4(1.0);
uniform mat4 proj  = mat4(1.0);
uniform int furLayers = 15;
uniform float furLength = 0.25;

// simple hash for randomness
float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    for (int layer = 0; layer < furLayers; ++layer) {

        float t = float(layer) / float(furLayers);
        float offset = t * furLength;     

        for (int i = 0; i < 3; ++i) {

            // Per-vertex randomization
            float r1 = hash(float(layer * 13 + i * 17));
            float r2 = hash(float(layer * 19 + i * 23));

            // small sideways jitter
            vec3 tangentJitter = normalize(
                vec3(r1 - 0.5, r2 - 0.5, r1 * r2 - 0.5)
            );

            // mix normal + jitter, more jitter toward the tip
            vec3 spikeDir = normalize(
                mix(vNormal[i], tangentJitter, t * 0.7)
            );

            vec3 displacedPos = vPosition[i] + spikeDir * offset;

            gPosition = displacedPos;
            gNormal   = spikeDir;
            gLayer    = t;

            gl_Position = proj * view * vec4(displacedPos, 1.0);
            EmitVertex();
        }

        EndPrimitive();
    }
}
#endif

    #ifdef FRAGMENT_SHADER
    in vec3 gNormal;
    in vec3 gPosition;
    in float gLayer;

    out vec4 fColor;

    uniform vec3 lightDir = normalize(vec3(-0.5, -1.0, -0.3));
    uniform vec3 baseColor = vec3(0, 0, 0); // black
    uniform vec3 tipColor  = vec3(1, 1, 1); // white tips

    void main() {
        vec3 N = normalize(gNormal);
        float light = max(dot(N, -lightDir), 0.2);

       
        vec3 color = mix(baseColor, tipColor, gLayer);

        
        float alpha = 1.0 - pow(gLayer, 2.0);

        fColor = vec4(color * light, alpha);
    }
    #endif
).";

    //fallback shader
    std::string const source2 = R".(
#ifdef VERTEX_SHADER

uniform mat4 view = mat4(1.f);
uniform mat4 proj = mat4(1.f);

out vec3 vColor;

 layout(location=0)in vec3 position;  
 layout(location=1)in vec3 color   ;

  void main(){
    mat4 pv = proj * view;
    gl_Position = pv*vec4(position,1);
    vColor = color;

  }
#endif

#ifdef FRAGMENT_SHADER
in vec3 vColor;
layout(location=0)out vec4 fColor;
void main(){
  fColor = vec4(vColor,1);
}
#endif
).";
    //phong shader
    std::string const source = R".(

#ifdef VERTEX_SHADER
uniform mat4 view  = mat4(1.f);
uniform mat4 proj  = mat4(1.f);
uniform mat4 model = mat4(1.f);

layout(location = 0)in vec3 position;
layout(location = 1)in vec3 normal  ;
layout(location = 2)in vec2 texCoord;

out vec2 vCoord;
out vec3 vNormal;
out vec3 vPosition;
out vec3 vCamPosition;
void main(){
  vCoord  = texCoord;
  vNormal = normal  ;
  vPosition = vec3(model*vec4(position,1.f));
  vCamPosition = vec3(inverse(view)*vec4(0,0,0,1));
  gl_Position = proj*view*model*vec4(position,1.f);
}
#endif

#ifdef FRAGMENT_SHADER
in vec3 vPosition;
in vec2 vCoord;
in vec3 vNormal;
in vec3 vCamPosition;

uniform vec3  lightPosition = vec3(30,30,30)   ;
uniform vec3  lightColor    = vec3(1,1,1)      ;
uniform vec3  lightAmbient  = vec3(0.3,0.1,0.0);
uniform float shininess     = 60.f             ;


uniform sampler2D diffuseTexture;
uniform vec4      diffuseColor = vec4(1.f);
uniform int       useTexture   = 0;
uniform vec3 difcolor = vec3(1.f);

layout(location=0)out vec4 fColor;
void main(){
  vec3 diffuseColor = difcolor;
  fColor = vec4(vNormal,1);

  if(useTexture == 1)
    diffuseColor = texture(diffuseTexture,vCoord).rgb;
  else
    diffuseColor = diffuseColor.rgb;


  vec3 finalColor = phongLighting(
      vPosition          ,
      normalize(vNormal) ,
      lightPosition      ,
      vCamPosition       ,
      lightColor         ,
      lightAmbient       ,
      diffuseColor       ,
      shininess          ,
      0.f                );

    fColor = vec4(finalColor,1.f);

}
#endif
).";
    std::string const cubemap = R".(
    #ifdef VERTEX_SHADER
layout(location = 0) in vec3 position;

out vec3 TexCoords;

uniform mat4 view;
uniform mat4 proj;

void main()
{
    TexCoords = position;

    mat4 viewNoTranslate = mat4(mat3(view));
    vec4 pos  = proj * viewNoTranslate * vec4(position, 1.0);
    gl_Position = pos.xyww;
}
#endif
#ifdef FRAGMENT_SHADER
in vec3 TexCoords;
out vec4 FragColor;

uniform samplerCube skybox;

void main()
{
    FragColor = texture(skybox, TexCoords);
}
#endif

).";
    //SKYBOX
    float skyboxVertices[] = {
        // +X (right)
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,

         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

         // -X (left)
         -1.0f, -1.0f,  1.0f,
         -1.0f, -1.0f, -1.0f,
         -1.0f,  1.0f, -1.0f,

         -1.0f,  1.0f, -1.0f,
         -1.0f,  1.0f,  1.0f,
         -1.0f, -1.0f,  1.0f,

         // +Y (top)
         -1.0f,  1.0f, -1.0f,
          1.0f,  1.0f, -1.0f,
          1.0f,  1.0f,  1.0f,

          1.0f,  1.0f,  1.0f,
         -1.0f,  1.0f,  1.0f,
         -1.0f,  1.0f, -1.0f,

         // -Y (bottom)
         -1.0f, -1.0f,  1.0f,
          1.0f, -1.0f,  1.0f,
          1.0f, -1.0f, -1.0f,

          1.0f, -1.0f, -1.0f,
         -1.0f, -1.0f, -1.0f,
         -1.0f, -1.0f,  1.0f,

         // +Z (front)
         -1.0f, -1.0f,  1.0f,
          1.0f, -1.0f,  1.0f,
          1.0f,  1.0f,  1.0f,

          1.0f,  1.0f,  1.0f,
         -1.0f,  1.0f,  1.0f,
         -1.0f, -1.0f,  1.0f,

         // -Z (back)
          1.0f, -1.0f, -1.0f,
         -1.0f, -1.0f, -1.0f,
         -1.0f,  1.0f, -1.0f,

         -1.0f,  1.0f, -1.0f,
          1.0f,  1.0f, -1.0f,
          1.0f, -1.0f, -1.0f
    };


    // ------------------------------------------------------------------------
    // Small math helpers for tangent generation (operate on float arrays)
    // ------------------------------------------------------------------------
    static inline void normalize3(float v[3])
    {
        float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 1e-8f) {
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        }
    }
    static inline void cross3(const float a[3], const float b[3], float r[3])
    {
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
    }
    static inline void sub3(const float a[3], const float b[3], float r[3])
    {
        r[0] = a[0] - b[0];
        r[1] = a[1] - b[1];
        r[2] = a[2] - b[2];
    }

    inline void computeSphericalUV(BlorbVertex& v)
    {
        glm::vec3 p(v.position[0], v.position[1], v.position[2]);

        // compute spherical coordinates
        float theta = atan2(p.z, p.x);     // [-pi, +pi]
        float phi = acos(glm::clamp(p.y / glm::length(p), -1.0f, 1.0f)); // [0, pi]

        // convert to 0–1 UVs
        float u = (theta + glm::pi<float>()) / (2.0f * glm::pi<float>());
        float vTex = phi / glm::pi<float>();

        v.texcoord[0] = u;
        v.texcoord[1] = vTex;
    }


    void computeTangentsAndSphericalUVs(
        BlorbVertex* vertices,
        size_t vertexCount,
        const uint32_t indices[][3],
        size_t triangleCount)
    {
        // 1) generate spherical UVs and zero tangents
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            computeSphericalUV(vertices[i]);
        }

        // reset all tangents
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            vertices[i].tan[0] = vertices[i].tan[1] = vertices[i].tan[2] = 0.0f;
            vertices[i].bitan[0] = vertices[i].bitan[1] = vertices[i].bitan[2] = 0.0f;
        }

        // triangle loop: compute tangents using UVs
        for (uint32_t i = 0; i < triangleCount; i += 3)
        {
            uint32_t i0 = indices[i][0];
            uint32_t i1 = indices[i][1];
            uint32_t i2 = indices[i][2];

            BlorbVertex& v0 = vertices[i0];
            BlorbVertex& v1 = vertices[i1];
            BlorbVertex& v2 = vertices[i2];

            glm::vec3 p0 = glm::make_vec3(v0.position);
            glm::vec3 p1 = glm::make_vec3(v1.position);
            glm::vec3 p2 = glm::make_vec3(v2.position);

            glm::vec2 uv0 = glm::make_vec2(v0.texcoord);
            glm::vec2 uv1 = glm::make_vec2(v1.texcoord);
            glm::vec2 uv2 = glm::make_vec2(v2.texcoord);

            glm::vec3 dp1 = p1 - p0;
            glm::vec3 dp2 = p2 - p0;

            glm::vec2 duv1 = uv1 - uv0;
            glm::vec2 duv2 = uv2 - uv0;

            float r = 1.0f / (duv1.x * duv2.y - duv1.y * duv2.x);

            glm::vec3 tangent = (dp1 * duv2.y - dp2 * duv1.y) * r;
            glm::vec3 bitangent = (dp2 * duv1.x - dp1 * duv2.x) * r;

            auto add = [&](BlorbVertex& v) {
                v.tan[0] += tangent.x;
                v.tan[1] += tangent.y;
                v.tan[2] += tangent.z;
                v.bitan[0] += bitangent.x;
                v.bitan[1] += bitangent.y;
                v.bitan[2] += bitangent.z;
                };

            add(v0); add(v1); add(v2);
        }

        // normalize
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            glm::vec3 T = glm::make_vec3(vertices[i].tan);
            glm::vec3 B = glm::make_vec3(vertices[i].bitan);

            T = glm::normalize(T);
            B = glm::normalize(B);

            memcpy(vertices[i].tan, glm::value_ptr(T), sizeof(float) * 3);
            memcpy(vertices[i].bitan, glm::value_ptr(B), sizeof(float) * 3);
        }

    }

   

    void recomputeInverseBindMatrices(Skeleton& skeleton) {

        std::vector<glm::mat4> globalBindPose(skeleton.bones.size());

        // Build global bind pose transforms
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            if (skeleton.bones[i].parent >= 0) {
                globalBindPose[i] = globalBindPose[skeleton.bones[i].parent] *
                    skeleton.bones[i].localTransform;
            }
            else {
                globalBindPose[i] = skeleton.bones[i].localTransform;
            }
        }

        // Compute inverse bind matrices
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            skeleton.bones[i].inverseBind = glm::inverse(globalBindPose[i]);

            if (i < 3) {
                glm::vec3 trans(globalBindPose[i][3]);

                glm::vec3 invTrans(skeleton.bones[i].inverseBind[3]);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Texture loader (stb_image)
    // ------------------------------------------------------------------------
    GLuint loadTexture(const char* path, GLuint& id) {
        int width, height, nrChannels;
        unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
        if (!data) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return 0;
        }
        GLenum format;
        if (nrChannels == 1) format = GL_RED;
        else if (nrChannels == 3) format = GL_RGB;
        else if (nrChannels == 4) format = GL_RGBA;
        std::cout << "number of channels is: " << nrChannels << std::endl;

        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,
            (GLint)std::floor(std::log2(std::max(width, height))));
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(data);

        return id;
    }

    // Load ALL primitives from the GLTF mesh, not just the first one
    void loadMeshFromGLTF(const tinygltf::Model& model, Mesh& mesh, const Skeleton& skeleton) {
        if (model.meshes.empty()) {
            std::cerr << "No meshes in GLTF" << std::endl;
            return;
        }

        std::cout << "Model has " << model.meshes.size() << " meshes" << std::endl;

        // Build node->bone mapping
        std::map<int, int> nodeToBoneIndex;
        if (!model.skins.empty()) {
            const tinygltf::Skin& skin = model.skins[0];
            for (size_t i = 0; i < skin.joints.size(); ++i) {
                nodeToBoneIndex[skin.joints[i]] = (int)i;
            }
        }



        // Accumulate all vertices and indices from ALL meshes and their primitives
        std::vector<BlorbVertex> allVertices;
        std::vector<unsigned int> allIndices;

        int totalInvalidBones = 0;
        int maxBoneID = 0;

        // Loop through ALL meshes in the model
        for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx) {
            const tinygltf::Mesh& gltfMesh = model.meshes[meshIdx];

            std::cout << "Loading mesh " << meshIdx << " ('" << gltfMesh.name
                << "') with " << gltfMesh.primitives.size() << " primitives" << std::endl;

            // Loop through ALL primitives in this mesh
            for (size_t primIdx = 0; primIdx < gltfMesh.primitives.size(); ++primIdx) {
                const tinygltf::Primitive& primitive = gltfMesh.primitives[primIdx];

                std::cout << "Loading primitive " << primIdx << "..." << std::endl;

                // Get accessors
                const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
                const tinygltf::Accessor& normAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];

                // Get buffer views
                const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
                const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::BufferView& indexView = model.bufferViews[indexAccessor.bufferView];

                // Get buffers
                const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
                const tinygltf::Buffer& normBuffer = model.buffers[normView.buffer];
                const tinygltf::Buffer& indexBuffer = model.buffers[indexView.buffer];

                const float* positions = reinterpret_cast<const float*>(
                    &posBuffer.data[posView.byteOffset + posAccessor.byteOffset]);

                const float* normals = reinterpret_cast<const float*>(
                    &normBuffer.data[normView.byteOffset + normAccessor.byteOffset]);

                // Extract index data
                const unsigned short* indices16 = nullptr;
                const unsigned int* indices32 = nullptr;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    indices16 = reinterpret_cast<const unsigned short*>(
                        &indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset]);
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    indices32 = reinterpret_cast<const unsigned int*>(
                        &indexBuffer.data[indexView.byteOffset + indexAccessor.byteOffset]);
                }

                // Get UV data
                const float* uvs = nullptr;
                if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor& uvAccessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                    const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
                    const tinygltf::Buffer& uvBuffer = model.buffers[uvView.buffer];
                    uvs = reinterpret_cast<const float*>(
                        &uvBuffer.data[uvView.byteOffset + uvAccessor.byteOffset]);
                }

                // Get bone IDs and weights
                const unsigned short* joints16 = nullptr;
                const unsigned char* joints8 = nullptr;
                const float* weights = nullptr;

                if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor& jointAccessor = model.accessors[primitive.attributes.at("JOINTS_0")];
                    const tinygltf::BufferView& jointView = model.bufferViews[jointAccessor.bufferView];
                    const tinygltf::Buffer& jointBuffer = model.buffers[jointView.buffer];

                    if (jointAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        joints16 = reinterpret_cast<const unsigned short*>(
                            &jointBuffer.data[jointView.byteOffset + jointAccessor.byteOffset]);
                    }
                    else if (jointAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        joints8 = reinterpret_cast<const unsigned char*>(
                            &jointBuffer.data[jointView.byteOffset + jointAccessor.byteOffset]);
                    }
                }

                if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end()) {
                    const tinygltf::Accessor& weightAccessor = model.accessors[primitive.attributes.at("WEIGHTS_0")];
                    const tinygltf::BufferView& weightView = model.bufferViews[weightAccessor.bufferView];
                    const tinygltf::Buffer& weightBuffer = model.buffers[weightView.buffer];
                    weights = reinterpret_cast<const float*>(
                        &weightBuffer.data[weightView.byteOffset + weightAccessor.byteOffset]);
                }

                // Build vertex data for this primitive
                size_t vertexCount = posAccessor.count;
                size_t vertexOffset = allVertices.size();  // Remember where this primitive starts

                int invalidBoneCount = 0;

                for (size_t i = 0; i < vertexCount; ++i) {
                    BlorbVertex v = {};

                    // Position
                    v.position[0] = positions[i * 3 + 0];
                    v.position[1] = positions[i * 3 + 1];
                    v.position[2] = positions[i * 3 + 2];

                    // Normal
                    v.normal[0] = normals[i * 3 + 0];
                    v.normal[1] = normals[i * 3 + 1];
                    v.normal[2] = normals[i * 3 + 2];

                    // UV
                    if (uvs) {
                        v.texcoord[0] = uvs[i * 2 + 0];
                        v.texcoord[1] = uvs[i * 2 + 1];
                    }

                    // Bone IDs - REMAP THEM!
                    if (joints16 || joints8) {
                        for (int j = 0; j < 4; ++j) {
                            int nodeIndex = 0;

                            if (joints16) {
                                nodeIndex = joints16[i * 4 + j];
                            }
                            else if (joints8) {
                                nodeIndex = joints8[i * 4 + j];
                            }

                            if (nodeToBoneIndex.find(nodeIndex) != nodeToBoneIndex.end()) {
                                v.boneIDs[j] = nodeToBoneIndex[nodeIndex];
                            }
                            else {
                                v.boneIDs[j] = 0;
                                invalidBoneCount++;
                            }

                            maxBoneID = std::max(maxBoneID, v.boneIDs[j]);
                        }
                    }
                    else {
                        v.boneIDs[0] = v.boneIDs[1] = v.boneIDs[2] = v.boneIDs[3] = 0;
                    }

                    // Weights
                    if (weights) {
                        v.weights[0] = weights[i * 4 + 0];
                        v.weights[1] = weights[i * 4 + 1];
                        v.weights[2] = weights[i * 4 + 2];
                        v.weights[3] = weights[i * 4 + 3];
                    }
                    else {
                        v.weights[0] = 1.0f;
                        v.weights[1] = v.weights[2] = v.weights[3] = 0.0f;
                    }

                    allVertices.push_back(v);
                }

                totalInvalidBones += invalidBoneCount;

                // Build index data for this primitive (with offset)
                size_t indexCount = indexAccessor.count;

                for (size_t i = 0; i < indexCount; ++i) {
                    unsigned int idx = 0;
                    if (indices16) {
                        idx = indices16[i];
                    }
                    else if (indices32) {
                        idx = indices32[i];
                    }

                    // Add the vertex offset to make indices point to correct vertices
                    allIndices.push_back(idx + vertexOffset);
                }


                std::cout << "  Mesh " << meshIdx << " Primitive " << primIdx << ": "
                    << vertexCount << " vertices, " << indexCount << " indices" << std::endl;
            }
        } // End mesh loop

        // Create GL buffers with combined data
        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.ebo);

        glBindVertexArray(mesh.vao);

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(GL_ARRAY_BUFFER,
            allVertices.size() * sizeof(BlorbVertex),
            allVertices.data(),
            GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            allIndices.size() * sizeof(unsigned int),
            allIndices.data(),
            GL_STATIC_DRAW);

        const GLsizei stride = sizeof(BlorbVertex);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, position));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, texcoord));

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, tan));

        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, bitan));

        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(5, 4, GL_INT, stride,
            (void*)offsetof(BlorbVertex, boneIDs));

        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, weights));

        glBindVertexArray(0);

        mesh.indexCount = static_cast<GLsizei>(allIndices.size());

        std::cout << "TOTAL: " << allVertices.size() << " vertices, "
            << allIndices.size() << " indices (" << (allIndices.size() / 3) << " triangles)" << std::endl;
    }
    // NEW FUNCTION: Compute rest pose
    void computeRestPose(Skeleton& skeleton) {
        std::vector<glm::mat4> globalTransforms(skeleton.bones.size());

        // Build global transforms from local transforms
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            if (skeleton.bones[i].parent >= 0) {
                globalTransforms[i] = globalTransforms[skeleton.bones[i].parent] *
                    skeleton.bones[i].localTransform;
            }
            else {
                globalTransforms[i] = skeleton.bones[i].localTransform;
            }
        }

        // Make relative to root (same as animation)
        if (!skeleton.bones.empty()) {
            glm::mat4 invRoot = glm::inverse(globalTransforms[0]);
            for (size_t i = 0; i < skeleton.bones.size(); ++i) {
                glm::mat4 relativeToRoot = invRoot * globalTransforms[i];
                skeleton.finalMatrices[i] = relativeToRoot * skeleton.bones[i].inverseBind;
            }
        }
        else {
            // Fallback if no root
            for (size_t i = 0; i < skeleton.bones.size(); ++i) {
                skeleton.finalMatrices[i] = globalTransforms[i] * skeleton.bones[i].inverseBind;
            }
        }
    }
    void buildSkeletonFromGLTF(
        const tinygltf::Model& model,
        Skeleton& skeleton
    ) {
        if (model.skins.empty()) {
            std::cerr << "No skins in model" << std::endl;
            return;
        }

        const tinygltf::Skin& skin = model.skins[0];
        size_t jointCount = skin.joints.size();

        skeleton.bones.resize(jointCount);
        skeleton.finalMatrices.resize(jointCount, glm::mat4(1.0f));

        std::cout << "Building skeleton with " << jointCount << " bones" << std::endl;

        // Step 1: Load inverse bind matrices
        if (skin.inverseBindMatrices >= 0) {
            const tinygltf::Accessor& acc = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[view.buffer];

            const float* data = reinterpret_cast<const float*>(
                &buffer.data[view.byteOffset + acc.byteOffset]);

            for (size_t i = 0; i < jointCount; ++i) {
                skeleton.bones[i].inverseBind = glm::make_mat4(data + i * 16);
            }
        }
        else {
            std::cerr << "ERROR: No inverse bind matrices!" << std::endl;
            return;
        }

        // Step 2: Build parent relationships
        for (size_t i = 0; i < jointCount; ++i) {
            skeleton.bones[i].parent = -1;
            int jointNode = skin.joints[i];

            for (size_t p = 0; p < jointCount; ++p) {
                int parentNode = skin.joints[p];
                const auto& children = model.nodes[parentNode].children;
                if (std::find(children.begin(), children.end(), jointNode) != children.end()) {
                    skeleton.bones[i].parent = (int)p;
                    break;
                }
            }
        }

        // Step 3: Load LOCAL transforms from nodes (this is the bind pose)
        for (size_t i = 0; i < jointCount; ++i) {
            int nodeIdx = skin.joints[i];
            const tinygltf::Node& node = model.nodes[nodeIdx];

            glm::mat4 transform = glm::mat4(1.0f);

            // GLTF stores transforms in this priority: matrix > TRS
            if (node.matrix.size() == 16) {
                // Matrix is in COLUMN-MAJOR order (OpenGL standard)
                transform = glm::make_mat4(node.matrix.data());
            }
            else {
                // Build from Translation, Rotation, Scale
                if (node.translation.size() == 3) {
                    transform = glm::translate(transform,
                        glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
                }
                if (node.rotation.size() == 4) {
                    // GLTF quaternion is [x, y, z, w]
                    glm::quat rot(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
                    transform = transform * glm::mat4_cast(rot);
                }
                if (node.scale.size() == 3) {
                    transform = glm::scale(transform,
                        glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
                }
            }

            skeleton.bones[i].localTransform = transform;

            // Debug: Print first few bones
            if (i < 3) {
                std::cout << "Bone " << i << " parent: " << skeleton.bones[i].parent << std::endl;
            }
        }

        // Step 4: Compute rest pose to verify everything is correct
        computeRestPose(skeleton);

        std::cout << "Skeleton built successfully" << std::endl;
    }


    bool isDefaultKeyframe(const Keyframe& k) {
        bool defaultTranslation = (glm::length(k.translation) < 0.0001f);
        bool defaultRotation = (glm::abs(k.rotation.w - 1.0f) < 0.0001f &&
            glm::length(glm::vec3(k.rotation.x, k.rotation.y, k.rotation.z)) < 0.0001f);
        bool defaultScale = (glm::abs(k.scale.x - 1.0f) < 0.0001f &&
            glm::abs(k.scale.y - 1.0f) < 0.0001f &&
            glm::abs(k.scale.z - 1.0f) < 0.0001f);

        return defaultTranslation && defaultRotation && defaultScale;
    }

    // BETTER: Interpolate with proper checking
    glm::mat4 InterpolateBone(
        const Animation& animation,
        int boneIndex,
        float time
    ) {
        const BoneAnimation* track = nullptr;
        for (const auto& ba : animation.boneAnimations) {
            if (ba.boneIndex == boneIndex) {
                track = &ba;
                break;
            }
        }

        if (!track || track->keys.empty()) {
            return glm::mat4(1.0f);
        }

        float animTime = fmod(time, animation.duration);
        if (animTime < 0.0f) animTime += animation.duration;

        if (track->keys.size() == 1) {
            const Keyframe& k = track->keys[0];
            return glm::translate(glm::mat4(1.0f), k.translation) *
                glm::mat4_cast(k.rotation) *
                glm::scale(glm::mat4(1.0f), k.scale);
        }

        const Keyframe* k0 = nullptr;
        const Keyframe* k1 = nullptr;

        for (size_t i = 0; i < track->keys.size() - 1; ++i) {
            if (animTime >= track->keys[i].time && animTime <= track->keys[i + 1].time) {
                k0 = &track->keys[i];
                k1 = &track->keys[i + 1];
                break;
            }
        }

        if (!k0 || !k1) {
            k0 = k1 = (animTime < track->keys.front().time) ? &track->keys.front() : &track->keys.back();
        }

        if (k0 == k1) {
            return glm::translate(glm::mat4(1.0f), k0->translation) *
                glm::mat4_cast(k0->rotation) *
                glm::scale(glm::mat4(1.0f), k0->scale);
        }

        float dt = k1->time - k0->time;
        float t = (dt > 0.0001f) ? (animTime - k0->time) / dt : 0.0f;
        t = glm::clamp(t, 0.0f, 1.0f);

        glm::vec3 pos = glm::mix(k0->translation, k1->translation, t);
        glm::quat rot = glm::normalize(glm::slerp(k0->rotation, k1->rotation, t));
        glm::vec3 scl = glm::mix(k0->scale, k1->scale, t);

        return glm::translate(glm::mat4(1.0f), pos) *
            glm::mat4_cast(rot) *
            glm::scale(glm::mat4(1.0f), scl);
    }
// FIXED updateSkeleton

    void updateSkeleton(
        Skeleton& skeleton,
        const Animation& animation,
        float time
    ) {
        if (skeleton.bones.empty()) return;

        skeleton.finalMatrices.resize(skeleton.bones.size());
        std::vector<glm::mat4> globalTransforms(skeleton.bones.size());

        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            glm::mat4 animatedLocal = InterpolateBone(animation, (int)i, time);

            glm::mat4 local;
            if (animatedLocal != glm::mat4(1.0f)) {
                local = animatedLocal;
            }
            else {
                local = skeleton.bones[i].localTransform;
            }

            if (skeleton.bones[i].parent >= 0) {
                globalTransforms[i] = globalTransforms[skeleton.bones[i].parent] * local;
            }
            else {
                globalTransforms[i] = local;
            }

            skeleton.finalMatrices[i] = globalTransforms[i] * skeleton.bones[i].inverseBind;
        }
    }

    void updateSkeletonRelativeToRoot(
        Skeleton& skeleton,
        const Animation& animation,
        float time
    ) {
        if (skeleton.bones.empty()) return;

        skeleton.finalMatrices.resize(skeleton.bones.size());
        std::vector<glm::mat4> globalTransforms(skeleton.bones.size());

        // First pass: compute all global transforms
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            glm::mat4 animatedLocal = InterpolateBone(animation, (int)i, time);

            glm::mat4 local;
            if (animatedLocal != glm::mat4(1.0f)) {
                local = animatedLocal;
            }
            else {
                local = skeleton.bones[i].localTransform;
            }

            if (skeleton.bones[i].parent >= 0) {
                globalTransforms[i] = globalTransforms[skeleton.bones[i].parent] * local;
            }
            else {
                globalTransforms[i] = local;
            }
        }

        // Get root bone's global transform
        glm::mat4 rootGlobal = globalTransforms[0];
        glm::mat4 invRootGlobal = glm::inverse(rootGlobal);

        // Second pass: make all transforms relative to root
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            // Transform to root-relative space, then apply inverse bind
            glm::mat4 relativeGlobal = invRootGlobal * globalTransforms[i];
            skeleton.finalMatrices[i] = relativeGlobal * skeleton.bones[i].inverseBind;
        }
    }


    int findBoneByName(const tinygltf::Model& model, const Skeleton& skeleton, const std::string& nameSubstring) {
        if (model.skins.empty()) return -1;

        const tinygltf::Skin& skin = model.skins[0];

        for (size_t i = 0; i < skin.joints.size(); ++i) {
            int nodeIdx = skin.joints[i];
            if (nodeIdx < model.nodes.size()) {
                const std::string& nodeName = model.nodes[nodeIdx].name;

                // Try exact match first
                if (nodeName == nameSubstring) {
                    std::cout << "Found bone '" << nodeName << "' at index " << i << " (exact match)" << std::endl;
                    return (int)i;
                }

                // Then try case-insensitive substring match
                std::string lowerName = nodeName;
                std::string lowerSearch = nameSubstring;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                std::transform(lowerSearch.begin(), lowerSearch.end(), lowerSearch.begin(), ::tolower);

                if (lowerName.find(lowerSearch) != std::string::npos) {
                    std::cout << "Found bone '" << nodeName << "' at index " << i << " (substring match)" << std::endl;
                    return (int)i;
                }
            }
        }

        std::cout << "Could not find bone matching '" << nameSubstring << "'" << std::endl;
        return -1;
    }

    void loadAnimationFromGLTF(
        const tinygltf::Model& model,
        Animation& animation
    ) {
        if (model.animations.empty()) {
            std::cerr << "No animations in GLTF" << std::endl;
            return;
        }

        const tinygltf::Animation& gltfAnim = model.animations[0];

        animation.boneAnimations.clear();
        animation.duration = 0.0f;

        std::map<int, BoneAnimation> boneMap;

        for (const auto& channel : gltfAnim.channels) {
            const auto& sampler = gltfAnim.samplers[channel.sampler];

            int boneIndex = -1;
            int targetNode = channel.target_node;
            for (size_t i = 0; i < model.skins[0].joints.size(); ++i) {
                if (model.skins[0].joints[i] == channel.target_node) {
                    boneIndex = (int)i;
                    break;
                }
            }
            if (boneIndex == -1) continue;


            if (boneMap.find(boneIndex) == boneMap.end()) {
                boneMap[boneIndex].boneIndex = boneIndex;
            }
            BoneAnimation& track = boneMap[boneIndex];

            std::vector<float> times = readFloatAccessor(model, sampler.input);

            for (float t : times) {
                animation.duration = std::max(animation.duration, t);
            }

            if (channel.target_path == "translation") {
                auto values = readVec3Accessor(model, sampler.output);

                for (size_t i = 0; i < times.size(); ++i) {
                    Keyframe* existing = nullptr;
                    for (auto& key : track.keys) {
                        if (std::abs(key.time - times[i]) < 0.001f) {
                            existing = &key;
                            break;
                        }
                    }

                    if (existing) {
                        existing->translation = values[i];
                    }
                    else {
                        Keyframe k;
                        k.time = times[i];
                        k.translation = values[i];
                        k.rotation = glm::quat(1, 0, 0, 0);
                        k.scale = glm::vec3(1.0f);
                        track.keys.push_back(k);
                    }
                }
            }
            else if (channel.target_path == "rotation") {
                auto values = readQuatAccessor(model, sampler.output);

                for (size_t i = 0; i < times.size(); ++i) {
                    Keyframe* existing = nullptr;
                    for (auto& key : track.keys) {
                        if (std::abs(key.time - times[i]) < 0.001f) {
                            existing = &key;
                            break;
                        }
                    }

                    if (existing) {
                        existing->rotation = values[i];
                    }
                    else {
                        Keyframe k;
                        k.time = times[i];
                        k.translation = glm::vec3(0.0f);  // Will be filled in later
                        k.rotation = values[i];
                        k.scale = glm::vec3(1.0f);
                        track.keys.push_back(k);
                    }
                }
            }
            else if (channel.target_path == "scale") {
                auto values = readVec3Accessor(model, sampler.output);

                for (size_t i = 0; i < times.size(); ++i) {
                    Keyframe* existing = nullptr;
                    for (auto& key : track.keys) {
                        if (std::abs(key.time - times[i]) < 0.001f) {
                            existing = &key;
                            break;
                        }
                    }

                    if (existing) {
                        existing->scale = values[i];
                    }
                    else {
                        Keyframe k;
                        k.time = times[i];
                        k.translation = glm::vec3(0.0f);  // Will be filled in later
                        k.rotation = glm::quat(1, 0, 0, 0);
                        k.scale = values[i];
                        track.keys.push_back(k);
                    }
                }
            }
        }
        // If a keyframe has zero translation but an earlier keyframe has non-zero,
        // copy the translation forward
        for (auto& pair : boneMap) {
            std::sort(pair.second.keys.begin(), pair.second.keys.end(),
                [](const Keyframe& a, const Keyframe& b) {
                    return a.time < b.time;
                });

            // Find the first non-zero translation
            glm::vec3 lastTranslation(0.0f);
            for (auto& key : pair.second.keys) {
                if (glm::length(key.translation) > 0.0001f) {
                    lastTranslation = key.translation;
                    break;
                }
            }

            // Fill in all keyframes with zero translation
            for (auto& key : pair.second.keys) {
                if (glm::length(key.translation) < 0.0001f) {
                    key.translation = lastTranslation;
                }
                else {
                    lastTranslation = key.translation;
                }
            }

            animation.boneAnimations.push_back(pair.second);
        }
    }

    void updateColor(vars::Vars& vars) {
        if (notChanged(vars, "all", __FUNCTION__, { "blue", "red", "green" })) return;
        GLuint loc = glGetUniformLocation(prg, "difcolor");
        if (loc != -1) {
            glUniform3f(loc, vars.getFloat("red"), vars.getFloat("green"), vars.getFloat("blue"));
        }
    }

    // ------------------------------------------------------------------------
    // updateShaders: only recompile when toggle changes
    // ------------------------------------------------------------------------
    void updateShaders(vars::Vars& vars) {
        int toggle = static_cast<int>(vars.getUint32("method.shaderToggle"));
        if (toggle == lastShaderToggle) return; // no change -> keep current program

        // shader toggle changed -> create new program
        if (prg) {
            glDeleteProgram(prg);
            prg = 0;
        }

        lastShaderToggle = toggle;

        if (toggle == 0) { // normal phong shader
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + source),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + phong::phongLightingShader + source),
                });
        }
        else if (toggle == 1) { // fur shader
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + furshader),
                createShader(GL_GEOMETRY_SHADER, "#version 460\n#define GEOMETRY_SHADER\n" + furshader),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + furshader),
                });
        }
        else if (toggle == 2) {
            diffuseTex = loadTexture(getTexturePath("blorbtextures/sackboy_diffuse.png").c_str(), diffuseTex);
            normalTex = loadTexture(getTexturePath("blorbtextures/baked_normals.png").c_str(), normalTex);
            prg = createProgram({ createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + leather),
                                createShader(GL_FRAGMENT_SHADER,   "#version 460\n#define FRAGMENT_SHADER\n" + leather) });
        }
        else if (toggle == 3) {
            GLuint tex = loadTexture(getTexturePath("blorbtextures/sackboy_diffuse.png").c_str(), texID);
            prg = createProgram({ createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + texshader),
                    createShader(GL_FRAGMENT_SHADER,   "#version 460\n#define FRAGMENT_SHADER\n" + texshader) });

        }
        else if (toggle == 4) {
            GLuint tex = loadTexture(getTexturePath("blorbtextures/blorb_dmap.png").c_str(), texID);
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + sourcetexWithShadows),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + phong::phongLightingShader + sourcetexWithShadows),
                });
        }
        else if (toggle == 5) {
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + gltfWithShadows),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + gltfWithShadows),
                });
        }
        else { // fallback simple shader
            prg = createProgram({
                createShader(GL_VERTEX_SHADER,   "#version 460\n#define VERTEX_SHADER\n" + source2),
                createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + source2),
                });
        }

        // cache uniform locations (may be -1 if not used)
        viewUniform = glGetUniformLocation(prg, "view");
        projUniform = glGetUniformLocation(prg, "proj");
        modelUniform = glGetUniformLocation(prg, "model");
    }

    Keyframe interpolateKeyframes(const Keyframe& a,
        const Keyframe& b,
        float time)
    {
        Keyframe result;

        float span = b.time - a.time;
        float t = (span > 0.0001f) ? (time - a.time) / span : 0.0f;
        t = glm::clamp(t, 0.0f, 1.0f);

        result.translation = glm::mix(a.translation, b.translation, t);
        result.rotation = glm::normalize(glm::slerp(a.rotation, b.rotation, t));
        result.scale = glm::mix(a.scale, b.scale, t);

        return result;
    }

    void createMesh(
        Mesh& mesh,
        const BlorbVertex* vertices,
        size_t vertexSize,
        const uint32_t* indices,
        size_t indexCount)
    {
        if (vertices[0].texcoord[0] == 0) {
            computeTangentsAndSphericalUVs(
                blorbVertices,
                5330,
                blorbIndices,
                10594
            );
        }
        //generateUVs(blorbVertices, sizeof(blorbVertices) / sizeof(BlorbVertex));

        glGenVertexArrays(1, &mesh.vao);
        glGenBuffers(1, &mesh.vbo);
        glGenBuffers(1, &mesh.ebo);

        glBindVertexArray(mesh.vao);

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
        glBufferData(GL_ARRAY_BUFFER, vertexSize, vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            indexCount * sizeof(uint32_t),
            indices,
            GL_STATIC_DRAW);

        const GLsizei stride = sizeof(BlorbVertex);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, position));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, texcoord));

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, tan));

        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride,
            (void*)offsetof(BlorbVertex, bitan));

        // bone IDs
        glEnableVertexAttribArray(5);
        glVertexAttribIPointer(
            5, 4, GL_INT,
            sizeof(BlorbVertex),
            (void*)offsetof(BlorbVertex, boneIDs)
        );

        // weights
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(
            6, 4, GL_FLOAT, GL_FALSE,
            sizeof(BlorbVertex),
            (void*)offsetof(BlorbVertex, weights)
        );


        glBindVertexArray(0);

        mesh.indexCount = static_cast<GLsizei>(indexCount);
    }
    void debugBoneMatrices(const Skeleton& skeleton) {
        std::cout << "\n=== BONE DEBUG ===" << std::endl;

        for (size_t i = 0; i < std::min(size_t(5), skeleton.bones.size()); ++i) {
            std::cout << "Bone " << i << ":" << std::endl;
            std::cout << "  Parent: " << skeleton.bones[i].parent << std::endl;

            // Check if inverse bind is identity (would be wrong)
            glm::mat4 inv = skeleton.bones[i].inverseBind;
            bool isIdentity = (inv == glm::mat4(1.0f));
            std::cout << "  Inverse bind is identity: " << (isIdentity ? "YES (BAD)" : "NO (good)") << std::endl;

            // Check local transform
            glm::mat4 local = skeleton.bones[i].localTransform;
            glm::vec3 translation(local[3]);
            std::cout << "  Local translation: (" << translation.x << ", "
                << translation.y << ", " << translation.z << ")" << std::endl;
        }

        std::cout << "==================\n" << std::endl;
    }

    void debugInverseBindMatrices(const Skeleton& skeleton) {
        std::cout << "\n=== INVERSE BIND MATRICES ===" << std::endl;
        for (size_t i = 0; i < std::min(size_t(3), skeleton.bones.size()); ++i) {
            const glm::mat4& inv = skeleton.bones[i].inverseBind;
            std::cout << "Bone " << i << " inverse bind:" << std::endl;
            std::cout << "  [" << inv[0][0] << ", " << inv[1][0] << ", " << inv[2][0] << ", " << inv[3][0] << "]" << std::endl;
            std::cout << "  [" << inv[0][1] << ", " << inv[1][1] << ", " << inv[2][1] << ", " << inv[3][1] << "]" << std::endl;
            std::cout << "  [" << inv[0][2] << ", " << inv[1][2] << ", " << inv[2][2] << ", " << inv[3][2] << "]" << std::endl;
            std::cout << "  [" << inv[0][3] << ", " << inv[1][3] << ", " << inv[2][3] << ", " << inv[3][3] << "]" << std::endl;

            // Extract translation from inverse bind
            glm::vec3 trans(inv[3]);
            std::cout << "  Translation: (" << trans.x << ", " << trans.y << ", " << trans.z << ")" << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
    }

    void buildSkeletonFromGLTF_Fixed(
        const tinygltf::Model& model,
        Skeleton& skeleton
    ) {
        if (model.skins.empty()) {
            std::cerr << "No skins in model" << std::endl;
            return;
        }

        const tinygltf::Skin& skin = model.skins[0];
        size_t jointCount = skin.joints.size();

        skeleton.bones.resize(jointCount);
        skeleton.finalMatrices.resize(jointCount, glm::mat4(1.0f));

        std::cout << "Building skeleton with " << jointCount << " bones" << std::endl;

        // Step 1: Build parent relationships using SKIN joint order
        for (size_t i = 0; i < jointCount; ++i) {
            skeleton.bones[i].parent = -1;
            int jointNode = skin.joints[i];  // This bone's node

            // Find this node's parent in the node hierarchy
            int parentNodeIndex = -1;
            for (size_t n = 0; n < model.nodes.size(); ++n) {
                const auto& node = model.nodes[n];
                for (int child : node.children) {
                    if (child == jointNode) {
                        parentNodeIndex = (int)n;
                        break;
                    }
                }
                if (parentNodeIndex >= 0) break;
            }

            // If we found a parent node, check if it's in the skin's joints
            if (parentNodeIndex >= 0) {
                for (size_t p = 0; p < jointCount; ++p) {
                    if (skin.joints[p] == parentNodeIndex) {
                        skeleton.bones[i].parent = (int)p;
                        break;
                    }
                }
            }

            // Debug print
            if (i < 7) {
                std::cout << "Bone " << i << " (node " << jointNode << "): parent = " << skeleton.bones[i].parent;
                if (skeleton.bones[i].parent >= 0) {
                    std::cout << " (node " << skin.joints[skeleton.bones[i].parent] << ")";
                }
                std::cout << std::endl;
            }
        }

        // Step 2: Load node LOCAL transforms
        for (size_t i = 0; i < jointCount; ++i) {
            int nodeIdx = skin.joints[i];
            const tinygltf::Node& node = model.nodes[nodeIdx];

            glm::mat4 transform = glm::mat4(1.0f);

            if (node.matrix.size() == 16) {
                transform = glm::make_mat4(node.matrix.data());
            }
            else {
                if (node.translation.size() == 3) {
                    transform = glm::translate(transform,
                        glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
                }
                if (node.rotation.size() == 4) {
                    glm::quat rot(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
                    transform = transform * glm::mat4_cast(rot);
                }
                if (node.scale.size() == 3) {
                    transform = glm::scale(transform,
                        glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
                }
            }

            skeleton.bones[i].localTransform = transform;
        }

        // Step 3: Compute GLOBAL bind pose transforms (before loading inverse bind)
        std::vector<glm::mat4> globalBindPose(jointCount);
        for (size_t i = 0; i < jointCount; ++i) {
            if (skeleton.bones[i].parent >= 0) {
                globalBindPose[i] = globalBindPose[skeleton.bones[i].parent] *
                    skeleton.bones[i].localTransform;
            }
            else {
                globalBindPose[i] = skeleton.bones[i].localTransform;
            }
        }

        // Step 4: Load inverse bind matrices
        if (skin.inverseBindMatrices >= 0) {
            const tinygltf::Accessor& acc = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[view.buffer];

            const float* data = reinterpret_cast<const float*>(
                &buffer.data[view.byteOffset + acc.byteOffset]);

            for (size_t i = 0; i < jointCount; ++i) {
                skeleton.bones[i].inverseBind = glm::make_mat4(data + i * 16);
            }
        }
        else {
            std::cerr << "ERROR: No inverse bind matrices!" << std::endl;
            // If no inverse bind matrices, compute them manually
            for (size_t i = 0; i < jointCount; ++i) {
                skeleton.bones[i].inverseBind = glm::inverse(globalBindPose[i]);
            }
        }

        // Step 5: Compute rest pose
        computeRestPose(skeleton);
        recomputeInverseBindMatrices(skeleton);

        std::cout << "Skeleton built successfully\n" << std::endl;
    }

    void findMeshNode(const tinygltf::Model& model) {
        std::cout << "\n=== SEARCHING FOR MESH NODE ===" << std::endl;

        for (size_t i = 0; i < model.nodes.size(); ++i) {
            const tinygltf::Node& node = model.nodes[i];
            if (node.mesh >= 0) {
                std::cout << "Mesh node " << i << " (name: " << node.name << ")" << std::endl;
                std::cout << "  Mesh index: " << node.mesh << std::endl;
                std::cout << "  Skin index: " << node.skin << std::endl;

                if (node.matrix.size() == 16) {
                    glm::mat4 mat = glm::make_mat4(node.matrix.data());
                    glm::vec3 trans(mat[3]);
                    std::cout << "  Matrix translation: (" << trans.x << ", " << trans.y << ", " << trans.z << ")" << std::endl;
                }
                else {
                    if (node.translation.size() == 3) {
                        std::cout << "  Translation: (" << node.translation[0] << ", "
                            << node.translation[1] << ", " << node.translation[2] << ")" << std::endl;
                    }
                }
            }
        }
        std::cout << "================================\n" << std::endl;
    }

    void printAllBoneNames(const tinygltf::Model& model, const Skeleton& skeleton) {
        std::cout << "\n=== ALL BONE NAMES ===" << std::endl;

        if (model.skins.empty()) {
            std::cout << "No skins found!" << std::endl;
            return;
        }

        const tinygltf::Skin& skin = model.skins[0];

        std::cout << "Total bones: " << skin.joints.size() << std::endl;
        std::cout << "Skeleton bones: " << skeleton.bones.size() << std::endl;

        for (size_t i = 0; i < skin.joints.size(); ++i) {
            int nodeIdx = skin.joints[i];

            if (nodeIdx < model.nodes.size()) {
                const tinygltf::Node& node = model.nodes[nodeIdx];

                std::cout << "Bone " << i << ": ";
                std::cout << "\"" << node.name << "\"";

                // Show parent
                if (i < skeleton.bones.size() && skeleton.bones[i].parent >= 0) {
                    int parentNodeIdx = skin.joints[skeleton.bones[i].parent];
                    if (parentNodeIdx < model.nodes.size()) {
                        std::cout << " (parent: \"" << model.nodes[parentNodeIdx].name << "\")";
                    }
                }
                else {
                    std::cout << " (ROOT)";
                }

                std::cout << std::endl;
            }
            else {
                std::cout << "Bone " << i << ": Invalid node index " << nodeIdx << std::endl;
            }
        }

        std::cout << "======================\n" << std::endl;
    }
    // ----------------------
    // lifecycle / callbacks
    // ----------------------
    void onInit(vars::Vars& vars) {
        stbi_set_flip_vertically_on_load(true);
        vars.addFloat("method.sensitivity", 0.1f); //Camera settings
        vars.addFloat("method.near", 0.10f);
        vars.addFloat("method.far", 100.00f);
        vars.addFloat("method.orbit.angleX", 0.50f);
        vars.addFloat("method.orbit.angleY", 0.50f);
        vars.addFloat("method.orbit.distance", 6.00f);
        vars.addFloat("method.orbit.zoomSpeed", 0.10f);
        vars.addUint32("method.shaderToggle", 0);
        addVarsLimitsF(vars, "method.sensitivity", -1.f, +1.f, 0.1f);
        //clothes toggle
        vars.addFloat("red", 1.f); vars.addFloat("green", 1.f); vars.addFloat("blue", 1.f);
        addVarsLimitsF(vars, "red", 0.f, 1.f, 0.01f); addVarsLimitsF(vars, "green", 0.f, 1.f, 0.01f); addVarsLimitsF(vars, "blue", 0.f, 1.f, 0.01f);
        vars.addBool("method.hat", false); //by default witch hat is off
        vars.addBool("method.robe", false);
        vars.addBool("method.winter_hat", false);
        vars.addBool("method.coat", false);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        stbi_set_flip_vertically_on_load(false);
        std::vector<std::string> faces = {
        getTexturePath("Yokohama2/posx.jpg").c_str(),
        getTexturePath("Yokohama2/negx.jpg").c_str(),
        getTexturePath("Yokohama2/posy.jpg").c_str(),
        getTexturePath("Yokohama2/negy.jpg").c_str(),
        getTexturePath("Yokohama2/posz.jpg").c_str(),
        getTexturePath("Yokohama2/negz.jpg").c_str(),
        };

        skyboxTex = loadCubemap(faces);

        // Initialize skeleton before anything else
        glbModel.skeleton.bones.clear();
        glbModel.skeleton.finalMatrices.clear();

        // Load GLB model
        tinygltf::TinyGLTF loader;
        std::string err, warn;

        bool ok = loader.LoadBinaryFromFile(
            &glbModel.model,
            &err,
            &warn,
            getTexturePath("blorbidle.glb").c_str()
        );
        

        if (ok) {
            buildSkeletonFromGLTF_Fixed(glbModel.model, glbModel.skeleton);
            loadAnimationFromGLTF(glbModel.model, glbModel.animation);
            loadMeshFromGLTF(glbModel.model, glbMesh, glbModel.skeleton);

            // Find the head bone
            headBoneIndex = findBoneByName(glbModel.model, glbModel.skeleton, "Bone.002");
            if (headBoneIndex == -1) {
                std::cout << "WARNING: Could not find head bone, trying alternatives..." << std::endl;
                // Try other common names
                headBoneIndex = findBoneByName(glbModel.model, glbModel.skeleton, "Bone.001");
            }

            std::cout << "Head bone index: " << headBoneIndex << std::endl;

            torsoBoneIndex = findBoneByName(glbModel.model, glbModel.skeleton, "Bone.001");


            glbDiffuseTex = loadTexture(
                getTexturePath("blorbtextures/blorb_dmap.png").c_str(),
                glbDiffuseTex
            );
            glbModel.loaded = true;
            findMeshNode(glbModel.model);
        }
        else {
            std::cerr << "Failed to load GLB: " << err << std::endl;
            // Create dummy skeleton with 1 bone
            glbModel.skeleton.bones.resize(1);
            glbModel.skeleton.bones[0].inverseBind = glm::mat4(1.0f);
            glbModel.skeleton.bones[0].localTransform = glm::mat4(1.0f);
            glbModel.skeleton.bones[0].parent = -1;
            glbModel.skeleton.finalMatrices.resize(1, glm::mat4(1.0f));
        }

        // Create shadow map framebuffer
        glGenFramebuffers(1, &shadowFBO);

        // Create depth texture
        glGenTextures(1, &shadowMap);
        glBindTexture(GL_TEXTURE_2D, shadowMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        // Attach depth texture to framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Shadow framebuffer not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Create shadow shader programs
        shadowProgram = createProgram({
            createShader(GL_VERTEX_SHADER, "#version 460\n#define VERTEX_SHADER\n" + shadowVertexShader),
            createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + shadowVertexShader)
            });

        shadowProgramGLTF = createProgram({
            createShader(GL_VERTEX_SHADER, "#version 460\n#define VERTEX_SHADER\n" + shadowVertexShaderGLTF),
            createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + shadowVertexShaderGLTF)
            });

        std::cout << "Shadow mapping initialized" << std::endl;
        // Initialize ground plane
        initGroundPlane();

        // Create ground plane shader
        groundPlaneProgram = createProgram({
            createShader(GL_VERTEX_SHADER, "#version 460\n#define VERTEX_SHADER\n" + groundPlaneShader),
            createShader(GL_FRAGMENT_SHADER, "#version 460\n#define FRAGMENT_SHADER\n" + groundPlaneShader)
            });

        std::cout << "Ground plane shader created" << std::endl;
        glGenBuffers(1, &bonesUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, bonesUBO);
        glBufferData(GL_UNIFORM_BUFFER,
            200 * sizeof(glm::mat4),
            nullptr,
            GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, bonesUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        stbi_set_flip_vertically_on_load(true);
        //build skybox buffers
        glGenVertexArrays(1, &skyboxvao);
        glGenBuffers(1, &skyboxvbo);

        glBindVertexArray(skyboxvao);
        glBindBuffer(GL_ARRAY_BUFFER, skyboxvbo);

        glBufferData(
            GL_ARRAY_BUFFER,
            sizeof(skyboxVertices),
            skyboxVertices,
            GL_STATIC_DRAW
        );
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE,
            3 * sizeof(float),
            (void*)0
        );

        glBindVertexArray(0);
        // build buffers
        createMesh(blorb, blorbVertices, sizeof(blorbVertices), &blorbIndices[0][0], 10594 * 3);
        createMesh(hat, witchVertices, sizeof(witchVertices), &witchIndices[0][0], 416 * 3);
        createMesh(robe, robeVertices, sizeof(robeVertices), &robeIndices[0][0], 696 * 3);
        createMesh(winthat, gorroVertices, sizeof(gorroVertices), &gorroIndices[0][0], 2464 * 3);
        createMesh(coat, coatVertices, sizeof(coatVertices), &coatIndices[0][0], 1312 * 3);
        glbModel.skeleton.finalMatrices.resize(glbModel.skeleton.bones.size(), glm::mat4(1.0f));
        // Hat texture
        hatTex = loadTexture(
            getTexturePath("blorbtextures/hat_dmap.png").c_str(),
            hatTex
        );
        robetex = loadTexture(getTexturePath("blorbtextures/robe_diffuse.png").c_str(),robetex);
        winttex = loadTexture(getTexturePath("blorbtextures/cap_bs.png").c_str(), winttex);
        coattex = loadTexture(getTexturePath("blorbtextures/coat.png").c_str(), coattex);

        // Hat shader program (sourcetex)
        hatPrg = createProgram({
            createShader(GL_VERTEX_SHADER,
                "#version 460\n#define VERTEX_SHADER\n" + sourcetexWithShadows),
            createShader(GL_FRAGMENT_SHADER,
                "#version 460\n#define FRAGMENT_SHADER\n" +
                phong::phongLightingShader + sourcetexWithShadows),
            });
        robeprg = createProgram({
            createShader(GL_VERTEX_SHADER,
                "#version 460\n#define VERTEX_SHADER\n" + sourcetexWithShadows),
            createShader(GL_FRAGMENT_SHADER,
                "#version 460\n#define FRAGMENT_SHADER\n" +
                phong::phongLightingShader + sourcetexWithShadows),
            });
        wintprg = createProgram({
            createShader(GL_VERTEX_SHADER,
                "#version 460\n#define VERTEX_SHADER\n" + sourcetexWithShadows),
            createShader(GL_FRAGMENT_SHADER,
                "#version 460\n#define FRAGMENT_SHADER\n" +
                phong::phongLightingShader + sourcetexWithShadows),
            });
        coatprg = createProgram({
            createShader(GL_VERTEX_SHADER,
                "#version 460\n#define VERTEX_SHADER\n" + sourcetexWithShadows),
            createShader(GL_FRAGMENT_SHADER,
                "#version 460\n#define FRAGMENT_SHADER\n" +
                phong::phongLightingShader + sourcetexWithShadows),
            });
        skyboxProgram = createProgram({
            createShader(GL_VERTEX_SHADER,
                "#version 460\n#define VERTEX_SHADER\n" + cubemap),
            createShader(GL_FRAGMENT_SHADER,
                "#version 460\n#define FRAGMENT_SHADER\n" +
                cubemap),
            });
        computeProjectionMatrix(vars);
    }

    void reactToSensitivityChange(vars::Vars& vars) {
        if (notChanged(vars, "method.all", __FUNCTION__, { "method.sensitivity" })) return;
        std::cerr << "sensitivity was changed to: " << vars.getFloat("method.sensitivity") << std::endl;
    }
    glm::vec3 interpolateVec3(const std::vector<Vec3Key>& keys, float time)
    {
        if (keys.empty())
            return glm::vec3(0.0f);

        if (keys.size() == 1)
            return keys[0].value;

        // Clamp time
        if (time <= keys.front().time)
            return keys.front().value;
        if (time >= keys.back().time)
            return keys.back().value;

        // Find keyframe interval
        for (size_t i = 0; i + 1 < keys.size(); i++) {
            const Vec3Key& k0 = keys[i];
            const Vec3Key& k1 = keys[i + 1];

            if (time >= k0.time && time <= k1.time) {
                float span = k1.time - k0.time;
                float t = (span > 0.0001f) ? (time - k0.time) / span : 0.0f;
                return glm::mix(k0.value, k1.value, t);
            }
        }

        return keys.back().value;
    }
    glm::quat interpolateQuat(const std::vector<QuatKey>& keys, float time)
    {
        if (keys.empty())
            return glm::quat(1, 0, 0, 0);

        if (keys.size() == 1)
            return glm::normalize(keys[0].value);

        if (time <= keys.front().time)
            return glm::normalize(keys.front().value);
        if (time >= keys.back().time)
            return glm::normalize(keys.back().value);

        for (size_t i = 0; i + 1 < keys.size(); i++) {
            const QuatKey& k0 = keys[i];
            const QuatKey& k1 = keys[i + 1];

            if (time >= k0.time && time <= k1.time) {
                float span = k1.time - k0.time;
                float t = (span > 0.0001f) ? (time - k0.time) / span : 0.0f;
                return glm::normalize(glm::slerp(k0.value, k1.value, t));
            }
        }

        return glm::normalize(keys.back().value);
    }

    void debugBone1Transform(const Animation& animation, float time) {
        std::cout << "\n=== BONE 1 DETAILED DEBUG ===" << std::endl;

        // Find Bone 1's animation track
        const BoneAnimation* track = nullptr;
        for (const auto& ba : animation.boneAnimations) {
            if (ba.boneIndex == 1) {
                track = &ba;
                break;
            }
        }

        if (!track) {
            std::cout << "No animation track for Bone 1!" << std::endl;
            return;
        }

        std::cout << "Bone 1 has " << track->keys.size() << " keyframes" << std::endl;

        // Show first few keyframes
        for (size_t i = 0; i < std::min(size_t(5), track->keys.size()); ++i) {
            const Keyframe& k = track->keys[i];
            std::cout << "  Keyframe " << i << " (t=" << k.time << "):" << std::endl;
            std::cout << "    Trans: (" << k.translation.x << ", " << k.translation.y << ", " << k.translation.z << ")" << std::endl;
            std::cout << "    Rot:   (" << k.rotation.x << ", " << k.rotation.y << ", " << k.rotation.z << ", " << k.rotation.w << ")" << std::endl;
            std::cout << "    Scale: (" << k.scale.x << ", " << k.scale.y << ", " << k.scale.z << ")" << std::endl;
        }

        // Now interpolate at current time
        float animTime = fmod(time, animation.duration);
        glm::mat4 interpolated = InterpolateBone(animation, 1, time);
        glm::vec3 trans(interpolated[3]);
        std::cout << "\nInterpolated at time " << animTime << ":" << std::endl;
        std::cout << "  Translation: (" << trans.x << ", " << trans.y << ", " << trans.z << ")" << std::endl;

        // Extract rotation
        glm::quat rot = glm::quat_cast(interpolated);
        std::cout << "  Rotation: (" << rot.x << ", " << rot.y << ", " << rot.z << ", " << rot.w << ")" << std::endl;

        std::cout << "============================\n" << std::endl;
    }
    void updateSkeletonWithRootExtracted(
        Skeleton& skeleton,
        const Animation& animation,
        float time,
        glm::mat4& outRootTransform  // Output the root transform
    ) {
        if (skeleton.bones.empty()) return;

        skeleton.finalMatrices.resize(skeleton.bones.size());
        std::vector<glm::mat4> globalTransforms(skeleton.bones.size());

        // Compute all global transforms
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            glm::mat4 animatedLocal = InterpolateBone(animation, (int)i, time);

            glm::mat4 local;
            if (animatedLocal != glm::mat4(1.0f)) {
                local = animatedLocal;
            }
            else {
                local = skeleton.bones[i].localTransform;
            }

            if (skeleton.bones[i].parent >= 0) {
                globalTransforms[i] = globalTransforms[skeleton.bones[i].parent] * local;
            }
            else {
                globalTransforms[i] = local;
            }
        }

        // Extract root transform for use in model matrix
        outRootTransform = globalTransforms[0];
        glm::mat4 invRoot = glm::inverse(outRootTransform);

        // Make all bones relative to root
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            glm::mat4 relativeToRoot = invRoot * globalTransforms[i];
            skeleton.finalMatrices[i] = relativeToRoot * skeleton.bones[i].inverseBind;
        }
    }

    void updateSkeletonSimple(
        Skeleton& skeleton,
        const Animation& animation,
        float time
    ) {
        if (skeleton.bones.empty()) return;

        skeleton.finalMatrices.resize(skeleton.bones.size());
        std::vector<glm::mat4> globalTransforms(skeleton.bones.size());

        // Compute all global transforms normally
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            glm::mat4 animatedLocal = InterpolateBone(animation, (int)i, time);

            glm::mat4 local;
            if (animatedLocal != glm::mat4(1.0f)) {
                local = animatedLocal;
            }
            else {
                local = skeleton.bones[i].localTransform;
            }

            if (skeleton.bones[i].parent >= 0) {
                globalTransforms[i] = globalTransforms[skeleton.bones[i].parent] * local;
            }
            else {
                globalTransforms[i] = local;
            }
        }

        // Apply inverse bind normally
        for (size_t i = 0; i < skeleton.bones.size(); ++i) {
            skeleton.finalMatrices[i] = globalTransforms[i] * skeleton.bones[i].inverseBind;
        }
    }

    glm::mat4 getBoneLocalTransform(
        const Skeleton& skeleton,
        const Animation& animation,
        int boneIndex,
        float time
    ) {
        if (boneIndex < 0 || boneIndex >= skeleton.bones.size()) {
            return glm::mat4(1.0f);
        }

        glm::mat4 animatedLocal = InterpolateBone(animation, boneIndex, time);

        if (animatedLocal != glm::mat4(1.0f)) {
            return animatedLocal;
        }
        else {
            return skeleton.bones[boneIndex].localTransform;
        }
    }

    glm::mat4 getBoneWorldTransform(
        const Skeleton& skeleton,
        const Animation& animation,
        int boneIndex,
        float time
    ) {
        if (boneIndex < 0 || boneIndex >= skeleton.bones.size()) {
            return glm::mat4(1.0f);
        }

        glm::mat4 localTransform = getBoneLocalTransform(skeleton, animation, boneIndex, time);

        // If this bone has a parent, recursively get parent's world transform
        if (skeleton.bones[boneIndex].parent >= 0) {
            glm::mat4 parentWorld = getBoneWorldTransform(
                skeleton, animation, skeleton.bones[boneIndex].parent, time
            );
            return parentWorld * localTransform;
        }

        // Root bone - just return local
        return localTransform;
    }

    void onDraw(vars::Vars& vars) {
        reactToSensitivityChange(vars);
        updateShaders(vars);

        // Calculate animation time
        animTime += vars.getFloat("event.dt");
        float t = (glbModel.animation.duration > 0.0f)
            ? fmod(animTime, glbModel.animation.duration)
            : 0.0f;

        // Update skeleton animation
        if (glbModel.loaded) {
            if (glbModel.animation.duration > 0.0f && animTime > 0.0f) {
                std::vector<glm::mat4> globalTransforms(glbModel.skeleton.bones.size());

                for (size_t i = 0; i < glbModel.skeleton.bones.size(); ++i) {
                    glm::mat4 animatedLocal = InterpolateBone(glbModel.animation, (int)i, t);

                    glm::mat4 local;
                    if (animatedLocal != glm::mat4(1.0f)) {
                        local = animatedLocal;
                    }
                    else {
                        local = glbModel.skeleton.bones[i].localTransform;
                    }

                    if (glbModel.skeleton.bones[i].parent >= 0) {
                        globalTransforms[i] = globalTransforms[glbModel.skeleton.bones[i].parent] * local;
                    }
                    else {
                        globalTransforms[i] = local;
                    }

                    glbModel.skeleton.finalMatrices[i] = globalTransforms[i] *
                        glbModel.skeleton.bones[i].inverseBind;
                }
            }
            else {
                // Rest pose
                std::vector<glm::mat4> globalTransforms(glbModel.skeleton.bones.size());

                for (size_t i = 0; i < glbModel.skeleton.bones.size(); ++i) {
                    if (glbModel.skeleton.bones[i].parent >= 0) {
                        globalTransforms[i] = globalTransforms[glbModel.skeleton.bones[i].parent] *
                            glbModel.skeleton.bones[i].localTransform;
                    }
                    else {
                        globalTransforms[i] = glbModel.skeleton.bones[i].localTransform;
                    }

                    glbModel.skeleton.finalMatrices[i] = globalTransforms[i] *
                        glbModel.skeleton.bones[i].inverseBind;
                }
            }
        }

        // Get bone transforms for accessories
        glm::mat4 headTransform = glm::mat4(1.0f);
        glm::mat4 torsoTransform = glm::mat4(1.0f);

        if (glbModel.loaded && glbModel.animation.duration > 0.0f) {
            if (headBoneIndex >= 0) {
                headTransform = getBoneWorldTransform(glbModel.skeleton, glbModel.animation, headBoneIndex, t);
            }
            if (torsoBoneIndex >= 0) {
                torsoTransform = getBoneWorldTransform(glbModel.skeleton, glbModel.animation, torsoBoneIndex, t);
            }
        }

        // ===== SHADOW PASS =====
        float near_plane = 1.0f, far_plane = 25.0f;
        glm::mat4 lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
        glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        lightSpaceMatrix = lightProjection * lightView;

        glUseProgram(shadowProgram);
        glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "lightSpaceMatrix"),
            1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
        glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
            1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f))); // Identity matrix
        glBindVertexArray(groundPlaneVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glClear(GL_DEPTH_BUFFER_BIT);

        // Render GLTF model to shadow map
        if (vars.getUint32("method.shaderToggle") == 5 && glbModel.loaded) {
            glUseProgram(shadowProgramGLTF);

            GLint bonesLoc = glGetUniformLocation(shadowProgramGLTF, "bones");
            if (bonesLoc != -1 && !glbModel.skeleton.finalMatrices.empty()) {
                int boneCount = std::min((int)glbModel.skeleton.finalMatrices.size(), 100);
                glUniformMatrix4fv(bonesLoc, boneCount, GL_FALSE,
                    glm::value_ptr(glbModel.skeleton.finalMatrices[0]));
            }

            glm::mat4 rootTransform = glm::mat4(1.0f);
            if (!glbModel.skeleton.bones.empty()) {
                rootTransform = getBoneWorldTransform(glbModel.skeleton, glbModel.animation, 0, t);
            }
            glm::mat4 adjustedModel = model * rootTransform;

            glUniformMatrix4fv(glGetUniformLocation(shadowProgramGLTF, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
            glUniformMatrix4fv(glGetUniformLocation(shadowProgramGLTF, "model"),
                1, GL_FALSE, glm::value_ptr(adjustedModel));

            glBindVertexArray(glbMesh.vao);
            glDrawElements(GL_TRIANGLES, glbMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        }
        else {
            // Render regular blorb to shadow map
            glUseProgram(shadowProgram);

            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
                1, GL_FALSE, glm::value_ptr(model));

            glBindVertexArray(blorb.vao);
            glDrawElements(GL_TRIANGLES, 10594 * 3, GL_UNSIGNED_INT, nullptr);
        }

        // Render accessories to shadow map
        glUseProgram(shadowProgram);

        if (vars.getBool("method.hat")) {
            glm::mat4 hatModel = model * headTransform;
            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
                1, GL_FALSE, glm::value_ptr(hatModel));
            glBindVertexArray(hat.vao);
            glDrawElements(GL_TRIANGLES, 416 * 3, GL_UNSIGNED_INT, nullptr);
        }

        if (vars.getBool("method.winter_hat")) {
            glm::mat4 winterHatModel = model * headTransform;
            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
                1, GL_FALSE, glm::value_ptr(winterHatModel));
            glBindVertexArray(winthat.vao);
            glDrawElements(GL_TRIANGLES, 2464 * 3, GL_UNSIGNED_INT, nullptr);
        }

        if (vars.getBool("method.robe")) {
            glm::mat4 robeTransform = glm::mat4(1.0f);
            robeTransform = glm::rotate(robeTransform, glm::radians(1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            robeTransform = glm::scale(robeTransform, glm::vec3(1.5f, 1.0f, 1.5f));
            glm::mat4 robeOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.8f, 0.0f));

            glm::mat4 robeModel = model * torsoTransform * robeOffset * robeTransform;

            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
                1, GL_FALSE, glm::value_ptr(robeModel));
            glBindVertexArray(robe.vao);
            glDrawElements(GL_TRIANGLES, 696 * 3, GL_UNSIGNED_INT, nullptr);
        }

        if (vars.getBool("method.coat")) {
            glm::mat4 coatModel = model * torsoTransform;
            glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "model"),
                1, GL_FALSE, glm::value_ptr(coatModel));
            glBindVertexArray(coat.vao);
            glDrawElements(GL_TRIANGLES, 1312 * 3, GL_UNSIGNED_INT, nullptr);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ===== NORMAL RENDERING PASS =====
        auto width = vars.getUint32("event.resizeX");
        auto height = vars.getUint32("event.resizeY");
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        computeProjectionMatrix(vars);
        computeViewMatrix(vars);

        // Use main program
        glUseProgram(prg);

        // Bind shadow map
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, shadowMap);
        glUniform1i(glGetUniformLocation(prg, "shadowMap"), 1);

        // Set light space matrix
        glUniformMatrix4fv(glGetUniformLocation(prg, "lightSpaceMatrix"),
            1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

        if (static_cast<int>(vars.getUint32("method.shaderToggle")) == 0) {
            updateColor(vars);
        }

        if (vars.getUint32("method.shaderToggle") == 2) { // texture with normal maps
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseTex);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, normalTex);

            glUniform1i(glGetUniformLocation(prg, "alb"), 0);
            glUniform1i(glGetUniformLocation(prg, "nor"), 1);
            glUniform3f(glGetUniformLocation(prg, "lightPos"), 5.f, 5.f, 5.f);
            glUniform3f(glGetUniformLocation(prg, "lightColor"), 1.f, 1.f, 1.f);
        }

        if (static_cast<int>(vars.getUint32("method.shaderToggle")) == 3) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texID);
            glUniform1i(glGetUniformLocation(prg, "tex"), 0);
        }

        if (static_cast<int>(vars.getUint32("method.shaderToggle")) == 4) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texID);
            glUniform1i(glGetUniformLocation(prg, "tex"), 0);
        }

        if (vars.getUint32("method.shaderToggle") == 5) {
            // Upload bone matrices
            GLint bonesLoc = glGetUniformLocation(prg, "bones");
            if (bonesLoc != -1 && !glbModel.skeleton.finalMatrices.empty()) {
                int boneCount = std::min((int)glbModel.skeleton.finalMatrices.size(), 100);
                glUniformMatrix4fv(bonesLoc, boneCount, GL_FALSE,
                    glm::value_ptr(glbModel.skeleton.finalMatrices[0]));
            }

            glm::mat4 adjustedModel = model;

            glUniformMatrix4fv(glGetUniformLocation(prg, "model"),
                1, GL_FALSE, glm::value_ptr(adjustedModel));

            glUniformMatrix4fv(glGetUniformLocation(prg, "view"),
                1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(prg, "proj"),
                1, GL_FALSE, glm::value_ptr(proj));

            // Set diffuse texture
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, glbDiffuseTex);
            glUniform1i(glGetUniformLocation(prg, "diffuseTexture"), 0);
            glUniform1i(glGetUniformLocation(prg, "useTexture"), 1);

            // IMPORTANT: Bind shadow map to texture unit 1
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowMap);
            glUniform1i(glGetUniformLocation(prg, "shadowMap"), 1);

            // Set light space matrix for shadows
            glUniformMatrix4fv(glGetUniformLocation(prg, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

            // Set base color
            glUniform3f(glGetUniformLocation(prg, "baseColor"),
                vars.getFloat("red"), vars.getFloat("green"), vars.getFloat("blue"));

            // Set lights
            glUniform3f(glGetUniformLocation(prg, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
            glUniform3f(glGetUniformLocation(prg, "lightColor"), 1.f, 1.f, 1.f);
        }

        glUniformMatrix4fv(glGetUniformLocation(prg, "view"),
            1, GL_FALSE, glm::value_ptr(view));

        // Update camera position
        GLint camLoc = glGetUniformLocation(prg, "cameraPos");
        if (camLoc != -1) {
            glm::vec4 camWorld = glm::inverse(view) * glm::vec4(0, 0, 0, 1);
            glUniform3f(camLoc, camWorld.x, camWorld.y, camWorld.z);
        }

        // Set view/proj/model uniforms
        if (viewUniform != -1) glProgramUniformMatrix4fv(prg, viewUniform, 1, GL_FALSE, glm::value_ptr(view));
        if (projUniform != -1) glProgramUniformMatrix4fv(prg, projUniform, 1, GL_FALSE, glm::value_ptr(proj));

        if (modelUniform != -1 && vars.getUint32("method.shaderToggle") != 5) {
            glProgramUniformMatrix4fv(prg, modelUniform, 1, GL_FALSE, glm::value_ptr(model));
        }

        // Render main mesh
        if (vars.getUint32("method.shaderToggle") == 5) {
            glBindVertexArray(glbMesh.vao);
            glDrawElements(GL_TRIANGLES, glbMesh.indexCount, GL_UNSIGNED_INT, nullptr);
        }
        else {
            glBindVertexArray(blorb.vao);
            glDrawElements(GL_TRIANGLES, 10594 * 3, GL_UNSIGNED_INT, nullptr);
        }

        // HAT rendering
        if (vars.getBool("method.hat")) {
            glUseProgram(hatPrg);

            glProgramUniformMatrix4fv(hatPrg, glGetUniformLocation(hatPrg, "view"),
                1, GL_FALSE, glm::value_ptr(view));
            glProgramUniformMatrix4fv(hatPrg, glGetUniformLocation(hatPrg, "proj"),
                1, GL_FALSE, glm::value_ptr(proj));

            glm::mat4 hatOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -2.0f, 0.0f));
            glm::mat4 hatModel = model * headTransform * hatOffset;

            glProgramUniformMatrix4fv(hatPrg, glGetUniformLocation(hatPrg, "model"),
                1, GL_FALSE, glm::value_ptr(hatModel));

            glUniformMatrix4fv(glGetUniformLocation(hatPrg, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, hatTex);
            glUniform1i(glGetUniformLocation(hatPrg, "tex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowMap);
            glUniform1i(glGetUniformLocation(hatPrg, "shadowMap"), 1);

            glBindVertexArray(hat.vao);
            glDrawElements(GL_TRIANGLES, 416 * 3, GL_UNSIGNED_INT, nullptr);
        }

        // WINTER HAT rendering
        if (vars.getBool("method.winter_hat")) {
            glUseProgram(wintprg);

            glProgramUniformMatrix4fv(wintprg, glGetUniformLocation(wintprg, "view"),
                1, GL_FALSE, glm::value_ptr(view));
            glProgramUniformMatrix4fv(wintprg, glGetUniformLocation(wintprg, "proj"),
                1, GL_FALSE, glm::value_ptr(proj));

            glm::mat4 winterHatOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -2.0f, 0.0f));
            glm::mat4 winterHatModel = model * headTransform * winterHatOffset;

            glProgramUniformMatrix4fv(wintprg, glGetUniformLocation(wintprg, "model"),
                1, GL_FALSE, glm::value_ptr(winterHatModel));

            glUniformMatrix4fv(glGetUniformLocation(wintprg, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, winttex);
            glUniform1i(glGetUniformLocation(wintprg, "tex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowMap);
            glUniform1i(glGetUniformLocation(wintprg, "shadowMap"), 1);

            glBindVertexArray(winthat.vao);
            glDrawElements(GL_TRIANGLES, 2464 * 3, GL_UNSIGNED_INT, nullptr);
        }

        // ROBE rendering
        if (vars.getBool("method.robe")) {
            glUseProgram(robeprg);

            glProgramUniformMatrix4fv(robeprg, glGetUniformLocation(robeprg, "view"),
                1, GL_FALSE, glm::value_ptr(view));
            glProgramUniformMatrix4fv(robeprg, glGetUniformLocation(robeprg, "proj"),
                1, GL_FALSE, glm::value_ptr(proj));

            // Create transformation for upside down and wide robe
            glm::mat4 robeTransform = glm::mat4(1.0f);

            // 1. Rotate 180 degrees around X axis (flip upside down)
            robeTransform = glm::rotate(robeTransform, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));

            // 2. Scale to make it wider (X and Z axes) and potentially taller (Y axis)
            robeTransform = glm::scale(robeTransform, glm::vec3(1.2f, 1.0f, 1.2f)); // 2x wider, same height

            // 3. Apply offset (adjust Y position as needed since it's upside down)
            glm::mat4 robeOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.8f, 0.0f));

            // Combine: model * torsoTransform * offset * robeTransform
            glm::mat4 robeModel = model * torsoTransform * robeOffset * robeTransform;

            glProgramUniformMatrix4fv(robeprg, glGetUniformLocation(robeprg, "model"),
                1, GL_FALSE, glm::value_ptr(robeModel));

            glUniformMatrix4fv(glGetUniformLocation(robeprg, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, robetex);
            glUniform1i(glGetUniformLocation(robeprg, "tex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowMap);
            glUniform1i(glGetUniformLocation(robeprg, "shadowMap"), 1);

            glBindVertexArray(robe.vao);
            glDrawElements(GL_TRIANGLES, 696 * 3, GL_UNSIGNED_INT, nullptr);

        }

        // COAT rendering
        if (vars.getBool("method.coat")) {
            glUseProgram(coatprg);

            glProgramUniformMatrix4fv(coatprg, glGetUniformLocation(coatprg, "view"),
                1, GL_FALSE, glm::value_ptr(view));
            glProgramUniformMatrix4fv(coatprg, glGetUniformLocation(coatprg, "proj"),
                1, GL_FALSE, glm::value_ptr(proj));

            glm::mat4 coatOffset = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.85f, 0.0f));
            glm::mat4 coatModel = model * torsoTransform * coatOffset;

            glProgramUniformMatrix4fv(coatprg, glGetUniformLocation(coatprg, "model"),
                1, GL_FALSE, glm::value_ptr(coatModel));

            glUniformMatrix4fv(glGetUniformLocation(coatprg, "lightSpaceMatrix"),
                1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, coattex);
            glUniform1i(glGetUniformLocation(coatprg, "tex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowMap);
            glUniform1i(glGetUniformLocation(coatprg, "shadowMap"), 1);

            glBindVertexArray(coat.vao);
            glDrawElements(GL_TRIANGLES, 1312 * 3, GL_UNSIGNED_INT, nullptr);
        }
        // Render ground plane with shadows
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(groundPlaneProgram);

        glUniformMatrix4fv(glGetUniformLocation(groundPlaneProgram, "view"),
            1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(groundPlaneProgram, "proj"),
            1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(glGetUniformLocation(groundPlaneProgram, "model"),
            1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
        glUniformMatrix4fv(glGetUniformLocation(groundPlaneProgram, "lightSpaceMatrix"),
            1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));

        glUniform3f(glGetUniformLocation(groundPlaneProgram, "lightPos"),
            lightPos.x, lightPos.y, lightPos.z);

        // Bind shadow map
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, shadowMap);
        glUniform1i(glGetUniformLocation(groundPlaneProgram, "shadowMap"), 0);

        glBindVertexArray(groundPlaneVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glDisable(GL_BLEND);

        // Draw skybox
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);

        glUseProgram(skyboxProgram);

        glUniformMatrix4fv(glGetUniformLocation(skyboxProgram, "view"),
            1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(skyboxProgram, "proj"),
            1, GL_FALSE, glm::value_ptr(proj));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxTex);
        glUniform1i(glGetUniformLocation(skyboxProgram, "skybox"), 0);

        glBindVertexArray(skyboxvao);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) std::cerr << "GL ERROR: " << err << std::endl;

        glBindVertexArray(0);
    }

    void onResize(vars::Vars& vars) {
        auto width = vars.getUint32("event.resizeX");
        auto height = vars.getUint32("event.resizeY");
        glViewport(0, 0, width, height);
    }

    void onQuit(vars::Vars& vars) {
        vars.erase("method");
        vars.erase("red");
        vars.erase("green");
        vars.erase("blue");

        if (prg) { glDeleteProgram(prg); prg = 0; }
        if (blorb.vao) { glDeleteVertexArrays(1, &blorb.vao); blorb.vao = 0; }
        if (blorb.vbo) { glDeleteBuffers(1, &blorb.vbo); blorb.vbo = 0; }
        if (blorb.ebo) { glDeleteBuffers(1, &blorb.ebo); blorb.ebo = 0; }
        if (hat.vao) { glDeleteVertexArrays(1, &hat.vao); hat.vao = 0; }
        if (hat.vbo) { glDeleteBuffers(1, &hat.vbo); hat.vbo = 0; }
        if (hat.ebo) { glDeleteBuffers(1, &hat.ebo); hat.ebo = 0; }
        if (bonesUBO) { glDeleteBuffers(1, &bonesUBO); bonesUBO = 0; }
        if (glbMesh.vao) { glDeleteVertexArrays(1, &glbMesh.vao); glbMesh.vao = 0; }
        if (glbMesh.vbo) { glDeleteBuffers(1, &glbMesh.vbo); glbMesh.vbo = 0; }
        if (glbMesh.ebo) { glDeleteBuffers(1, &glbMesh.ebo); glbMesh.ebo = 0; }
        if (shadowFBO) { glDeleteFramebuffers(1, &shadowFBO); shadowFBO = 0; }
        if (shadowMap) { glDeleteTextures(1, &shadowMap); shadowMap = 0; }
        if (shadowProgram) { glDeleteProgram(shadowProgram); shadowProgram = 0; }
        if (shadowProgramGLTF) { glDeleteProgram(shadowProgramGLTF); shadowProgramGLTF = 0; }
        if (groundPlaneVAO) { glDeleteVertexArrays(1, &groundPlaneVAO); groundPlaneVAO = 0; }
        if (groundPlaneVBO) { glDeleteBuffers(1, &groundPlaneVBO); groundPlaneVBO = 0; }
        if (groundPlaneEBO) { glDeleteBuffers(1, &groundPlaneEBO); groundPlaneEBO = 0; }
        if (groundPlaneProgram) { glDeleteProgram(groundPlaneProgram); groundPlaneProgram = 0; }
    }

    void computeProjectionMatrix(vars::Vars& vars) {
        auto width = vars.getUint32("event.resizeX");
        auto height = vars.getUint32("event.resizeY");
        auto _near = vars.getFloat("method.near");
        auto _far = vars.getFloat("method.far");

        float aspect = static_cast<float>(width) / static_cast<float>(height);
        proj = glm::perspective(glm::half_pi<float>(), aspect, _near, _far);
    }

    void computeViewMatrix(vars::Vars& vars) {
        auto angleX = vars.getFloat("method.orbit.angleX");
        auto angleY = vars.getFloat("method.orbit.angleY");
        auto distance = vars.getFloat("method.orbit.distance");
        view =
            glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -distance)) *
            glm::rotate(glm::mat4(1.f), angleX, glm::vec3(1.f, 0.f, 0.f)) *
            glm::rotate(glm::mat4(1.f), angleY, glm::vec3(0.f, 1.f, 0.f));
    }

    void onKeyDown(vars::Vars& vars) {
        auto key = vars.getInt32("event.key");
        auto sensitivity = vars.getFloat("method.sensitivity");
        auto& angleX = vars.getFloat("method.orbit.angleX");
        auto& angleY = vars.getFloat("method.orbit.angleY");
        if (key == SDLK_a) angleY += sensitivity;
        if (key == SDLK_d) angleY -= sensitivity;
        if (key == SDLK_w) angleX += sensitivity;
        if (key == SDLK_s) angleX -= sensitivity;
    }

    void onKeyUp(vars::Vars& vars) {
        (void)vars;
    }

    void onMouseMotion(vars::Vars& vars) {
        auto xrel = vars.getInt32("event.mouse.xrel");
        auto yrel = vars.getInt32("event.mouse.yrel");

        auto sensitivity = vars.getFloat("method.sensitivity");
        auto zoomSpeed = vars.getFloat("method.orbit.zoomSpeed");
        auto& angleX = vars.getFloat("method.orbit.angleX");
        auto& angleY = vars.getFloat("method.orbit.angleY");
        auto& distance = vars.getFloat("method.orbit.distance");

        if (vars.getBool("event.mouse.middle")) {
            angleX += sensitivity * yrel;
            angleY += sensitivity * xrel;

            angleX = glm::clamp(angleX, -glm::half_pi<float>(), glm::half_pi<float>());
        }
        if (vars.getBool("event.mouse.right")) {
            distance += zoomSpeed * yrel;
            distance = glm::clamp(distance, 0.f, 100.f);
        }
    }

    void onUpdate(vars::Vars& vars) {
        (void)vars;
    }

    // register entrypoint
    EntryPoint main = []() {
        methodManager::Callbacks clbs;
        clbs.onDraw = onDraw;
        clbs.onInit = onInit;
        clbs.onQuit = onQuit;
        clbs.onResize = onResize;
        clbs.onKeyDown = onKeyDown;
        clbs.onKeyUp = onKeyUp;
        clbs.onMouseMotion = onMouseMotion;
        clbs.onUpdate = onUpdate;
        MethodRegister::get().manager.registerMethod("student.project", clbs);
        };

} // namespace student::project
