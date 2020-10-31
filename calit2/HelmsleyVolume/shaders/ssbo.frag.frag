#version 460

out vec4 FragColor;

layout(binding = 0) uniform atomic_uint data;

void main()
{
    FragColor = vec4(1, 0, 0, 1);
    atomicCounterAdd(data, 10);
}