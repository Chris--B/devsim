#version 450

layout(location = 0) in vec2 vTexCoord;

layout(location = 0, index = 0) out vec4 OutFinalColor;

#include "Common.glsl"

void main()
{
    OutFinalColor = vec4(texture(sampler2D(uTextures[0], uSampler), vTexCoord).rgb, 1.0);
}